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
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    cfg:     dict,
    seed:    int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Train XGBoost, Random Forest, and Logistic Regression models.

    Parameters
    ----------
    X_train, y_train : training features and soft labels (failure frequency)
    X_val,   y_val   : validation features and labels
    cfg              : full experiment config dict
    seed             : random seed

    Returns
    -------
    models_dict : {'xgboost': fitted_model, 'random_forest': ..., 'logistic': ...}
                  plus calibration wrappers if available.
    """
    from sklearn.preprocessing import StandardScaler  # type: ignore
    from calibrate import CalibratedModel

    # Binary labels for classification models
    y_train_bin = (y_train >= 0.5).astype(int)
    y_val_bin   = (y_val   >= 0.5).astype(int)

    # Check class balance
    pos_frac = y_train_bin.mean()
    if verbose:
        print(f"  Class balance: pos_frac={pos_frac:.4f} "
              f"({'IMBALANCED — using sample weights' if pos_frac < 0.05 else 'OK'})")

    sample_weights = None
    if pos_frac < 0.05 or pos_frac > 0.95:
        # Inverse-frequency weighting
        n_pos  = y_train_bin.sum()
        n_neg  = len(y_train_bin) - n_pos
        w_pos  = len(y_train_bin) / (2 * max(n_pos, 1))
        w_neg  = len(y_train_bin) / (2 * max(n_neg, 1))
        sample_weights = np.where(y_train_bin == 1, w_pos, w_neg)

    results = {}

    # ── 1. XGBoost ────────────────────────────────────────────────────────────
    if verbose:
        print("  Training XGBoost ...")
    xgb_model = _train_xgboost(
        X_train, y_train_bin, X_val, y_val_bin,
        cfg['models']['xgboost'], seed, sample_weights
    )
    xgb_calibrated = CalibratedModel(xgb_model, method='platt', seed=seed)
    xgb_calibrated.fit(X_val, y_val_bin)
    results['xgboost'] = xgb_calibrated
    if verbose:
        print(f"    XGBoost fitted (best params logged internally)")

    # ── 2. Random Forest ─────────────────────────────────────────────────────
    if verbose:
        print("  Training Random Forest ...")
    rf_model = _train_random_forest(
        X_train, y_train_bin, X_val, y_val_bin,
        cfg['models']['random_forest'], seed, sample_weights
    )
    rf_calibrated = CalibratedModel(rf_model, method='isotonic', seed=seed)
    rf_calibrated.fit(X_val, y_val_bin)
    results['random_forest'] = rf_calibrated
    if verbose:
        print(f"    Random Forest fitted")

    # ── 3. Logistic Regression ────────────────────────────────────────────────
    if verbose:
        print("  Training Logistic Regression ...")
    scaler   = StandardScaler()
    X_tr_s   = scaler.fit_transform(X_train)
    X_val_s  = scaler.transform(X_val)
    lr_model = _train_logistic_regression(
        X_tr_s, y_train_bin, X_val_s, y_val_bin,
        cfg['models']['logistic_regression'], seed, sample_weights
    )
    results['logistic_regression'] = lr_model
    results['logistic_scaler']     = scaler
    if verbose:
        print(f"    Logistic Regression fitted")

    return results


# ── Model-specific trainers ───────────────────────────────────────────────────

def _train_xgboost(X_tr, y_tr, X_val, y_val, xgb_cfg, seed, weights=None):
    """Grid search over XGBoost hyperparameters, select on val Brier score."""
    try:
        import xgboost as xgb  # type: ignore
    except ImportError:
        raise ImportError("xgboost required: pip install xgboost")
    from sklearn.metrics import brier_score_loss  # type: ignore

    best_model, best_brier = None, np.inf

    for depth in xgb_cfg['max_depth']:
        for lr in xgb_cfg['learning_rate']:
            for n_est in xgb_cfg['n_estimators']:
                clf = xgb.XGBClassifier(
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

                clf.fit(
                    X_tr, y_tr,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                    **fit_params,
                )
                proba   = clf.predict_proba(X_val)[:, 1]
                brier   = brier_score_loss(y_val, proba)

                if brier < best_brier:
                    best_brier, best_model = brier, clf

    return best_model


def _train_random_forest(X_tr, y_tr, X_val, y_val, rf_cfg, seed, weights=None):
    """Grid search Random Forest on validation Brier score."""
    from sklearn.ensemble import RandomForestClassifier  # type: ignore
    from sklearn.metrics import brier_score_loss          # type: ignore

    best_model, best_brier = None, np.inf

    for depth in rf_cfg['max_depth']:
        for min_leaf in rf_cfg['min_samples_leaf']:
            clf = RandomForestClassifier(
                n_estimators=rf_cfg['n_estimators'],
                max_depth=depth,
                min_samples_leaf=min_leaf,
                n_jobs=-1,
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


# ── Persistence helpers ──────────────────────────────────────────────────────

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


# ── SHAP analysis ─────────────────────────────────────────────────────────────

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


# ── CLI entry point ──────────────────────────────────────────────────────────

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

    models = train_all_models(X_tr, y_tr, X_val, y_val, cfg, seed=cfg['random_seed'], verbose=True)
    save_models(models, args.out, args.key)
    print(f"[train] Models saved to {args.out}")


if __name__ == '__main__':
    main()
