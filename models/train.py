"""
models/train.py
===============
Train all three model tiers described in the implementation plan:

  1. XGBoost       — primary model; calibrated with Platt scaling
  2. Random Forest — nonlinear baseline; calibrated with isotonic regression
  3. Logistic Regression — interpretable linear baseline

Each model is trained on soft probability targets (failure_freq) treated as a
regression/classification problem.  XGBoost is treated as a binary classifier
with soft labels clamped to [0, 1]; RF and LR use the binarised label.

Hyperparameter selection: grid search on validation Brier score.

Usage (CLI):
    python models/train.py --config configs/experiment_config.yaml \
        --key sub05_k30_simple --out models/saved/
"""

import argparse
import os
import sys
import pickle
from typing import Any, Dict, Optional, Tuple

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def train_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,  # continuous failure_freq in [0, 1]
    X_val:   np.ndarray,
    y_val:   np.ndarray,  # continuous failure_freq in [0, 1] — HP tuning only
    X_cal:   np.ndarray,  # independent calibration set (not seen during HP tuning)
    y_cal:   np.ndarray,  # continuous failure_freq in [0, 1]
    cfg:     dict,
    seed:    int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Train XGBoost, Random Forest, and Logistic Regression models.

    y_train / y_val / y_cal are continuous empirical failure frequencies in [0, 1].
    XGBoost and Random Forest regress on these directly (R1 fix: target is the
    expected per-run failure probability, not the majority-fail binary label).
    Logistic Regression uses binarised labels and is a majority-fail discriminator.
    Calibration is fit on the independent X_cal / y_cal split (R6 fix).

    Parameters
    ----------
    X_train, y_train : training features and continuous failure_freq
    X_val,   y_val   : tuning features and labels (HP selection only)
    X_cal,   y_cal   : independent calibration set (never used for HP tuning)
    cfg              : full experiment config dict
    seed             : random seed

    Returns
    -------
    models_dict : {'xgboost': fitted_model, 'random_forest': ..., 'logistic': ...}
                  plus calibration wrappers if available.
    """
    from sklearn.preprocessing import StandardScaler  # type: ignore
    from calibrate import CalibratedModel

    # Binary labels used only for LR (discriminator) and class-balance checks
    y_train_bin = (y_train >= 0.5).astype(int)
    y_val_bin   = (y_val   >= 0.5).astype(int)

    # Check class balance using binarised labels
    pos_frac = float(y_train_bin.mean())
    if verbose:
        print(f"  Class balance: pos_frac={pos_frac:.4f} "
              f"({'IMBALANCED' if pos_frac < 0.05 or pos_frac > 0.95 else 'OK'})")

    # Single-class guard: classifiers require at least 2 classes
    if len(np.unique(y_train_bin)) < 2:
        from sklearn.dummy import DummyClassifier
        from sklearn.preprocessing import StandardScaler
        if verbose:
            label = 'all-negative (no failures)' if y_train_bin[0] == 0 else 'all-positive'
            print(f"  WARNING: single-class labels ({label}) — using DummyClassifier for all models.")
        dummy = DummyClassifier(strategy='most_frequent')
        dummy.fit(X_train, y_train_bin)
        scaler = StandardScaler()
        scaler.fit(X_train)
        dummy_cal = CalibratedModel(dummy, method='none')
        dummy_cal.fit(X_cal, y_cal)
        return {
            'xgboost': dummy_cal,
            'random_forest': dummy_cal,
            'logistic_regression': dummy,
            'logistic_scaler': scaler,
        }

    # Sample weights for LR only (imbalanced binary classification)
    lr_sample_weights = None
    if pos_frac < 0.05 or pos_frac > 0.95:
        n_pos  = y_train_bin.sum()
        n_neg  = len(y_train_bin) - n_pos
        w_pos  = len(y_train_bin) / (2 * max(n_pos, 1))
        w_neg  = len(y_train_bin) / (2 * max(n_neg, 1))
        lr_sample_weights = np.where(y_train_bin == 1, w_pos, w_neg)

    results = {}

    # -- 1. XGBoost — regressor on continuous failure_freq (R1 fix) -----------
    if verbose:
        print("  Training XGBoost ...")
    xgb_model = _train_xgboost(
        X_train, y_train, X_val, y_val,
        cfg['models']['xgboost'], seed, weights=None
    )
    xgb_calibrated = CalibratedModel(xgb_model, method='platt', seed=seed)
    xgb_calibrated.fit(X_cal, y_cal)   # R6 fix: independent calibration set
    results['xgboost'] = xgb_calibrated
    if verbose:
        print(f"    XGBoost fitted")

    # -- 2. Random Forest — regressor on continuous failure_freq (R1 fix) -----
    if verbose:
        print("  Training Random Forest ...")
    rf_model = _train_random_forest(
        X_train, y_train, X_val, y_val,
        cfg['models']['random_forest'], seed, weights=None
    )
    rf_calibrated = CalibratedModel(rf_model, method='isotonic', seed=seed)
    rf_calibrated.fit(X_cal, y_cal)    # R6 fix: independent calibration set
    results['random_forest'] = rf_calibrated
    if verbose:
        print(f"    Random Forest fitted")

    # -- 3. Logistic Regression — binary majority-fail discriminator -----------
    # LR uses binarised labels. Its output is P(majority_fail), not the expected
    # per-run failure probability. Treated as a risk-ranking baseline.
    if verbose:
        print("  Training Logistic Regression ...")
    scaler   = StandardScaler()
    X_tr_s   = scaler.fit_transform(X_train)
    X_val_s  = scaler.transform(X_val)
    lr_model = _train_logistic_regression(
        X_tr_s, y_train_bin, X_val_s, y_val_bin,
        cfg['models']['logistic_regression'], seed, lr_sample_weights
    )
    results['logistic_regression'] = lr_model
    results['logistic_scaler']     = scaler
    if verbose:
        print(f"    Logistic Regression fitted")

    return results


# -- Model-specific trainers ---------------------------------------------------

def _train_xgboost(X_tr, y_tr, X_val, y_val, xgb_cfg, seed, weights=None):
    """Grid search over XGBoost hyperparameters, select on val Brier score.

    Uses XGBRegressor with reg:logistic objective to regress directly on
    continuous failure_freq targets (R1 fix). Brier score is MSE against
    the continuous targets.
    """
    try:
        import xgboost as xgb  # type: ignore
    except ImportError:
        raise ImportError("xgboost required: pip install xgboost")

    best_model, best_brier = None, np.inf

    for depth in xgb_cfg['max_depth']:
        for lr in xgb_cfg['learning_rate']:
            for n_est in xgb_cfg['n_estimators']:
                reg = xgb.XGBRegressor(
                    objective='reg:logistic',
                    max_depth=depth,
                    learning_rate=lr,
                    n_estimators=n_est,
                    subsample=xgb_cfg.get('subsample', 0.8),
                    colsample_bytree=xgb_cfg.get('colsample_bytree', 0.8),
                    eval_metric='logloss',
                    early_stopping_rounds=xgb_cfg.get('early_stopping_rounds', 20),
                    random_state=seed,
                    verbosity=0,
                )
                fit_params = {}
                if weights is not None:
                    fit_params['sample_weight'] = weights

                reg.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                    **fit_params,
                )
                preds = np.clip(reg.predict(X_val), 0.0, 1.0)
                brier = float(np.mean((y_val - preds) ** 2))

                if brier < best_brier:
                    best_brier, best_model = brier, reg

    return best_model


def _train_random_forest(X_tr, y_tr, X_val, y_val, rf_cfg, seed, weights=None):
    """Grid search Random Forest regressor on validation Brier score.

    Uses RandomForestRegressor to regress directly on continuous failure_freq
    targets (R1 fix). Brier score is MSE against the continuous targets.
    """
    from sklearn.ensemble import RandomForestRegressor  # type: ignore

    best_model, best_brier = None, np.inf

    for depth in rf_cfg['max_depth']:
        for min_leaf in rf_cfg['min_samples_leaf']:
            reg = RandomForestRegressor(
                n_estimators=rf_cfg['n_estimators'],
                max_depth=depth,
                min_samples_leaf=min_leaf,
                n_jobs=-1,
                random_state=seed,
            )
            fit_params = {}
            if weights is not None:
                fit_params['sample_weight'] = weights
            reg.fit(X_tr, y_tr, **fit_params)

            preds = np.clip(reg.predict(X_val), 0.0, 1.0)
            brier = float(np.mean((y_val - preds) ** 2))

            if brier < best_brier:
                best_brier, best_model = brier, reg

    return best_model


def _train_logistic_regression(X_tr, y_tr, X_val, y_val, lr_cfg, seed, weights=None):
    """Grid search Logistic Regression on validation Brier score."""
    from sklearn.linear_model import LogisticRegression  # type: ignore
    from sklearn.metrics import brier_score_loss          # type: ignore

    best_model, best_brier = None, np.inf

    for C in lr_cfg['C']:
        clf = LogisticRegression(
            C=C,
            max_iter=lr_cfg['max_iter'],
            solver=lr_cfg.get('solver', 'lbfgs'),
            random_state=seed,
        )
        fit_params = {}
        if weights is not None:
            fit_params['sample_weight'] = weights
        clf.fit(X_tr, y_tr, **fit_params)

        proba = clf.predict_proba(X_val)[:, 1]
        brier = brier_score_loss(y_val, proba)

        if brier < best_brier:
            best_brier, best_model = brier, clf

    return best_model


# -- Persistence helpers ------------------------------------------------------

def save_models(models: dict, out_dir: str, key: str):
    """Pickle all models to the output directory."""
    os.makedirs(out_dir, exist_ok=True)
    for name, model in models.items():
        path = os.path.join(out_dir, f'{key}_{name}.pkl')
        with open(path, 'wb') as f:
            pickle.dump(model, f)


def load_models(out_dir: str, key: str) -> dict:
    """Load pickled models from the output directory."""
    import glob
    models = {}
    for path in glob.glob(os.path.join(out_dir, f'{key}_*.pkl')):
        name = os.path.basename(path).replace(f'{key}_', '').replace('.pkl', '')
        with open(path, 'rb') as f:
            models[name] = pickle.load(f)
    return models


# -- SHAP analysis -------------------------------------------------------------

def compute_shap_values(
    model,
    X: np.ndarray,
    feature_names: list,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute SHAP values for the XGBoost model.

    Returns
    -------
    shap_values   : ndarray (N, n_features)
    importances   : mean |SHAP| per feature
    """
    try:
        import shap  # type: ignore
    except ImportError:
        raise ImportError("shap required: pip install shap")

    # Unwrap calibrated model if necessary
    base_model = model.base_model if hasattr(model, 'base_model') else model
    explainer  = shap.TreeExplainer(base_model)
    shap_vals  = explainer.shap_values(X)
    # For binary classifier, shap returns either ndarray or list[ndarray]
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    importances = np.abs(shap_vals).mean(axis=0)
    return shap_vals, importances


# -- CLI entry point ----------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Train all ML models for one dataset config.')
    parser.add_argument('--config', default='configs/experiment_config.yaml')
    parser.add_argument('--key',    required=True, help='Dataset config key, e.g. sub05_k30_simple')
    parser.add_argument('--out',    default='models/saved/')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    sys.path.insert(0, 'src')
    from dataset_assembler import load_dataset

    print(f"[train] Loading dataset: {args.key}")
    X_tr, X_val, X_te, y_tr, y_val, y_te, feat_names = load_dataset(args.key, cfg)
    print(f"  train={len(y_tr)}, val={len(y_val)}, test={len(y_te)}, features={X_tr.shape[1]}")

    # Split val into tune (HP selection) and cal (calibration fitting) — R6 fix
    rng  = np.random.default_rng(cfg['random_seed'])
    perm = rng.permutation(len(X_val))
    mid  = len(perm) // 2
    tune_idx, cal_idx = perm[:mid], perm[mid:]
    X_tune, y_tune = X_val[tune_idx], y_val[tune_idx]
    X_cal,  y_cal  = X_val[cal_idx],  y_val[cal_idx]
    print(f"  val split -> tune={len(y_tune)}, cal={len(y_cal)}")

    models = train_all_models(X_tr, y_tr, X_tune, y_tune, X_cal, y_cal,
                               cfg, seed=cfg['random_seed'], verbose=True)
    save_models(models, args.out, args.key)
    print(f"[train] Models saved to {args.out}")


if __name__ == '__main__':
    main()
