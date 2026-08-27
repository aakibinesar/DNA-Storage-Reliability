"""
analysis/shap_stability.py
===========================
R13 mitigation: bootstrap stability analysis of SHAP feature importances.

Background
----------
Figure 3 of the paper reports SHAP-derived feature importance rankings (e.g.
"GC deviation and homopolymer length are the dominant predictors").  The R13
concern is that a single-run SHAP importance vector may be specific to the
particular test-set sample drawn, and that feature rankings could shift if the
test set were re-sampled — making claims about dominant features fragile.

This script quantifies that fragility by repeating the SHAP computation on
N_BOOTSTRAP independent bootstrap samples of the test set, computing:

  1. Kendall's tau between each bootstrap's importance ranking and the
     reference (full test set) ranking.  High tau (close to 1.0) indicates
     the ranking is reproducible.
  2. Per-feature rank standard deviation across bootstraps.  A small rank_std
     means the feature always lands near the same position regardless of
     which test sequences are sampled.
  3. Top-K consistency: fraction of bootstrap runs in which each of the
     reference top-K features appears among the bootstrap top-K.

A feature is labelled STABLE if its rank_std <= STABLE_RANK_STD (default 3.0),
meaning its rank moves less than ±3 positions across bootstraps.

The analysis uses the *informative-regime* test sequences only (R11 fix):
sequences in the under_failure and saturated regimes have near-0 or near-1
SHAP values that add noise without reflecting decision-critical importance.

Outputs
-------
  <out>/shap_stability_{key}.csv   — one row per feature; mean_importance,
                                     std_importance, cv_importance,
                                     ref_rank, mean_rank, std_rank, is_stable,
                                     top_k_consistency
  <out>/shap_tau_{key}.csv         — scalar summary: mean_tau, std_tau, CI95,
                                     n_stable_features, frac_stable

Usage:
    python analysis/shap_stability.py \\
        --config  configs/experiment_config.yaml \\
        --key     sub09_k5_simple \\
        --models-dir models/saved/ \\
        --out     results/shap_stability/ \\
        [--n-bootstrap 200] [--top-k 10]
"""

import argparse
import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))

N_BOOTSTRAP    = 200
TOP_K          = 10      # features to track for "top-K consistency"
STABLE_RANK_STD = 3.0   # rank std threshold for "stable" label


# -- SHAP helpers --------------------------------------------------------------

def _shap_importances(model, X: np.ndarray) -> np.ndarray:
    """Mean |SHAP| per feature.  Unwraps CalibratedModel if necessary."""
    try:
        import shap  # type: ignore
    except ImportError:
        raise ImportError("shap required: pip install shap")

    base = model.base_model if hasattr(model, 'base_model') else model
    explainer = shap.TreeExplainer(base)
    vals = explainer.shap_values(X)
    if isinstance(vals, list):
        vals = vals[1]
    return np.abs(vals).mean(axis=0)


def _ranks(importances: np.ndarray) -> np.ndarray:
    """Return rank array (rank 1 = highest importance).  Ties broken by index."""
    order = np.argsort(-importances, kind='stable')
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(order) + 1)
    return ranks


# -- Regime filter -------------------------------------------------------------

def _informative_mask(failure_freq: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Boolean mask: True for sequences in the informative regime."""
    return (failure_freq >= lo) & (failure_freq <= hi)


# -- Core analysis -------------------------------------------------------------

def run_shap_stability(
    model,
    X_test:       np.ndarray,
    failure_freq: np.ndarray,
    feat_names:   List[str],
    cfg:          dict,
    n_bootstrap:  int = N_BOOTSTRAP,
    top_k:        int = TOP_K,
    seed:         int = 42,
    verbose:      bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Bootstrap SHAP stability analysis.

    Parameters
    ----------
    model        : calibrated XGBoost (CalibratedModel wrapping XGBRegressor)
    X_test       : test feature matrix (N, n_features)
    failure_freq : continuous failure_freq for each test sequence (N,)
    feat_names   : feature name list (length n_features)
    cfg          : experiment config dict (for regime thresholds)
    n_bootstrap  : number of bootstrap samples
    top_k        : track which top-k features are consistently top-k

    Returns
    -------
    feat_df : per-feature stability DataFrame
    tau_df  : Kendall's tau summary DataFrame
    """
    from scipy.stats import kendalltau  # type: ignore

    lo = cfg.get('evaluation', {}).get('regime_lo', 0.15)
    hi = cfg.get('evaluation', {}).get('regime_hi', 0.85)

    # Restrict to informative regime (R11 fix)
    mask = _informative_mask(failure_freq, lo, hi)
    X_info = X_test[mask]
    n_info = mask.sum()

    if n_info < 20:
        raise ValueError(
            f"Only {n_info} informative-regime sequences — "
            f"SHAP stability requires at least 20."
        )

    if verbose:
        print(f"  [shap_stability] informative subset: {n_info}/{len(X_test)} sequences")

    # Reference importances on full informative subset
    ref_imp   = _shap_importances(model, X_info)
    ref_ranks = _ranks(ref_imp)
    top_k_set = set(np.argsort(-ref_imp)[:top_k])

    if verbose:
        print(f"  [shap_stability] Reference top-{top_k}: "
              f"{[feat_names[i] for i in np.argsort(-ref_imp)[:top_k]]}")
        print(f"  [shap_stability] Running {n_bootstrap} bootstraps ...")

    rng = np.random.default_rng(seed)
    tau_vals            = np.empty(n_bootstrap)
    boot_importances    = np.empty((n_bootstrap, len(feat_names)))
    boot_top_k_hits     = np.zeros(len(feat_names), dtype=int)

    for b in range(n_bootstrap):
        idx     = rng.integers(0, n_info, size=n_info)
        X_boot  = X_info[idx]
        imp     = _shap_importances(model, X_boot)
        boot_importances[b] = imp

        tau, _ = kendalltau(ref_imp, imp)
        tau_vals[b] = tau

        # Track top-K hits
        boot_top = set(np.argsort(-imp)[:top_k])
        for i in top_k_set:
            if i in boot_top:
                boot_top_k_hits[i] += 1

    # Per-feature statistics
    mean_imp = boot_importances.mean(axis=0)
    std_imp  = boot_importances.std(axis=0)
    cv_imp   = np.where(mean_imp > 1e-10, std_imp / mean_imp, np.nan)

    boot_ranks = np.apply_along_axis(_ranks, axis=1, arr=boot_importances)
    mean_rank  = boot_ranks.mean(axis=0)
    std_rank   = boot_ranks.std(axis=0)

    top_k_consistency = np.where(
        np.array([i in top_k_set for i in range(len(feat_names))]),
        boot_top_k_hits / n_bootstrap,
        np.nan,
    )

    feat_df = pd.DataFrame({
        'feature'          : feat_names,
        'ref_importance'   : ref_imp,
        'ref_rank'         : ref_ranks,
        'mean_importance'  : mean_imp,
        'std_importance'   : std_imp,
        'cv_importance'    : cv_imp,
        'mean_rank'        : mean_rank,
        'std_rank'         : std_rank,
        'is_stable'        : std_rank <= STABLE_RANK_STD,
        'top_k_consistency': top_k_consistency,
    }).sort_values('ref_rank').reset_index(drop=True)

    mean_tau = float(np.nanmean(tau_vals))
    std_tau  = float(np.nanstd(tau_vals))
    n_stable = int((std_rank <= STABLE_RANK_STD).sum())

    tau_df = pd.DataFrame([{
        'n_bootstrap'        : n_bootstrap,
        'n_info_sequences'   : int(n_info),
        'regime_lo'          : lo,
        'regime_hi'          : hi,
        'top_k'              : top_k,
        'stable_rank_std_thr': STABLE_RANK_STD,
        'mean_tau'           : mean_tau,
        'std_tau'            : std_tau,
        'ci95_lo'            : float(np.percentile(tau_vals, 2.5)),
        'ci95_hi'            : float(np.percentile(tau_vals, 97.5)),
        'n_stable_features'  : n_stable,
        'frac_stable_features': n_stable / len(feat_names),
    }])

    if verbose:
        print(f"  [shap_stability] Mean Kendall tau = {mean_tau:.4f} "
              f"(95% CI: [{tau_df['ci95_lo'].values[0]:.4f}, "
              f"{tau_df['ci95_hi'].values[0]:.4f}])")
        print(f"  [shap_stability] Stable features (rank_std <= {STABLE_RANK_STD}): "
              f"{n_stable}/{len(feat_names)}")
        print(f"  [shap_stability] Top-{top_k} by reference importance:")
        top_rows = feat_df[feat_df['ref_rank'] <= top_k][
            ['feature', 'ref_rank', 'std_rank', 'is_stable', 'top_k_consistency']
        ]
        for _, r in top_rows.iterrows():
            stable_tag = 'STABLE' if r['is_stable'] else 'UNSTABLE'
            consistency = (f"{r['top_k_consistency']:.0%}"
                           if not np.isnan(r['top_k_consistency']) else '—')
            print(f"    #{int(r['ref_rank']):>2}  {r['feature']:<35}  "
                  f"rank_std={r['std_rank']:.2f}  [{stable_tag}]  "
                  f"top-{top_k} in {consistency} of boots")

    return feat_df, tau_df


# -- CLI -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='R13: bootstrap SHAP stability analysis.'
    )
    parser.add_argument('--config',      default='configs/experiment_config.yaml')
    parser.add_argument('--key',         required=True)
    parser.add_argument('--models-dir',  default='models/saved/')
    parser.add_argument('--out',         default='results/shap_stability/')
    parser.add_argument('--n-bootstrap', type=int, default=N_BOOTSTRAP)
    parser.add_argument('--top-k',       type=int, default=TOP_K)
    parser.add_argument('--seed',        type=int, default=42)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    from dataset_assembler import load_dataset
    from train import load_models

    print(f"[shap_stability] {args.key}")
    X_tr, X_val, X_te, y_tr, y_val, y_te, feat_names = load_dataset(args.key, cfg)
    models  = load_models(args.models_dir, args.key)
    xgb_cal = models.get('xgboost')
    if xgb_cal is None:
        print(f"  [ERROR] No XGBoost model found for {args.key}")
        sys.exit(1)

    failure_freq = np.asarray(y_te, dtype=float)

    try:
        feat_df, tau_df = run_shap_stability(
            model=xgb_cal,
            X_test=X_te,
            failure_freq=failure_freq,
            feat_names=list(feat_names),
            cfg=cfg,
            n_bootstrap=args.n_bootstrap,
            top_k=args.top_k,
            seed=args.seed,
            verbose=True,
        )
    except ImportError as e:
        print(f"  [ERROR] {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"  [SKIP] {e}")
        sys.exit(0)

    os.makedirs(args.out, exist_ok=True)
    feat_path = os.path.join(args.out, f'shap_stability_{args.key}.csv')
    tau_path  = os.path.join(args.out, f'shap_tau_{args.key}.csv')
    feat_df.to_csv(feat_path, index=False, float_format='%.6f')
    tau_df.to_csv(tau_path,  index=False, float_format='%.6f')
    print(f"  [shap_stability] Saved feature stats  -> {feat_path}")
    print(f"  [shap_stability] Saved tau summary    -> {tau_path}")


if __name__ == '__main__':
    main()
