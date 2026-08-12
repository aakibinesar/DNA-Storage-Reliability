"""
analysis/threshold_sensitivity.py
==================================
R8 mitigation: quantify label uncertainty due to the binary 0.5 threshold and
demonstrate that model quality metrics are not highly sensitive to threshold choice.

Background
----------
With M=30 Monte Carlo runs a sequence's failure_freq has 95% Wilson CI ≈ ±0.18
at p=0.5, so sequences with true failure probability in (0.32, 0.68) may be
mislabelled by the binary rule.  This script:

  1. Reports what fraction of sequences in the test set are "near-threshold"
     (their CI spans 0.5), making the label statistically uncertain.

  2. Sweeps the decision threshold from 0.10 to 0.90 in steps of 0.05 and
     reports precision, recall, F1, accuracy at each value, while noting that
     AUROC and PR-AUC are threshold-free and stay constant.

  3. Identifies the threshold that maximises F1 on the test set, so authors
     can either justify 0.5 or switch to the F1-optimal threshold.

Outputs
-------
  <out>/<key>_threshold_sensitivity.csv      — metric sweep table
  <out>/<key>_near_threshold_stats.csv       — label uncertainty summary

Usage (CLI):
    python analysis/threshold_sensitivity.py \\
        --config configs/experiment_config.yaml \\
        --key sub09_k5_simple \\
        --models-dir models/saved/ \\
        --out results/threshold_sensitivity/
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))


# ── Core analysis functions (used by both CLI and pipeline stage) ─────────────

def run_threshold_sensitivity(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    failure_freq: np.ndarray,
    n_runs: int,
    thresholds: np.ndarray = None,
    threshold: float = 0.5,
    alpha: float = 0.05,
    verbose: bool = True,
) -> tuple:
    """Compute threshold sweep table and near-threshold stats.

    Parameters
    ----------
    y_true       : binary labels (0/1) from the test set (based on the 0.5 rule)
    y_prob       : model predicted probabilities
    failure_freq : continuous failure_freq values for the test set
    n_runs       : M used when generating labels (for CI computation)
    thresholds   : thresholds to sweep (default 0.10–0.90 in 0.05 steps)
    threshold    : reference threshold used for the binary labels (default 0.5)

    Returns
    -------
    (sweep_df, nt_stats) where sweep_df is a DataFrame of metrics per threshold
    and nt_stats is a dict of near-threshold statistics.
    """
    from evaluate import threshold_sensitivity
    from label_generator import near_threshold_stats

    if thresholds is None:
        thresholds = np.arange(0.10, 0.91, 0.05)

    rows   = threshold_sensitivity(failure_freq, y_prob, thresholds)
    sweep_df = pd.DataFrame(rows)

    # Identify threshold that maximises F1
    best_idx = sweep_df['f1'].idxmax()
    best_t   = float(sweep_df.loc[best_idx, 'threshold'])
    best_f1  = float(sweep_df.loc[best_idx, 'f1'])

    nt = near_threshold_stats(failure_freq, n_runs, threshold=threshold, alpha=alpha)

    if verbose:
        print(f"  AUROC = {sweep_df['auroc'].iloc[0]:.4f}  "
              f"PR-AUC = {sweep_df['pr_auc'].iloc[0]:.4f}  "
              f"(threshold-free)")
        print(f"  F1 at threshold=0.50: "
              f"{sweep_df.loc[(sweep_df['threshold'] - 0.5).abs().idxmin(), 'f1']:.4f}")
        print(f"  Best F1 at threshold={best_t:.2f}: {best_f1:.4f}")
        print(f"  Near-threshold sequences: "
              f"{nt['n_near_threshold']}/{nt['n_total']} "
              f"({nt['frac_near']:.1%}), mean CI width = {nt['ci_width_mean']:.3f}")

    return sweep_df, nt


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='R8: binary threshold sensitivity and label uncertainty analysis.'
    )
    parser.add_argument('--config',     default='configs/experiment_config.yaml')
    parser.add_argument('--key',        required=True,
                        help='Dataset key, e.g. sub09_k5_simple')
    parser.add_argument('--models-dir', default='models/saved/')
    parser.add_argument('--out',        default='results/threshold_sensitivity/')
    parser.add_argument('--alpha',      type=float, default=0.05,
                        help='CI significance level (default 0.05 → 95%% CI)')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    from dataset_assembler import load_dataset
    from train import load_models

    parts    = args.key.split('_')
    sub_rate = int(parts[0].replace('sub', '')) / 100
    coverage = int(parts[1].replace('k', ''))

    print(f"[threshold_sensitivity] {args.key}  sub_rate={sub_rate}  coverage={coverage}")

    X_tr, X_val, X_te, y_tr, y_val, y_te, feat_names = load_dataset(args.key, cfg)
    models  = load_models(args.models_dir, args.key)
    xgb_cal = models.get('xgboost')

    if xgb_cal is None:
        print("[threshold_sensitivity] No XGBoost model found — skipping.")
        return

    y_prob = xgb_cal.predict_proba(X_te)

    # Load continuous failure_freq for the test set (not just binary labels)
    data_path = os.path.join(cfg['paths']['datasets_dir'], f'{args.key}.parquet')
    splits_path = os.path.join(cfg['paths']['splits_dir'], f'{args.key}_splits.parquet')
    df     = pd.read_parquet(data_path)
    splits = pd.read_parquet(splits_path)

    test_idx     = splits[splits['split'] == 'test']['index'].values
    failure_freq = df['failure_freq'].values[test_idx]
    n_runs       = int(df.attrs.get('n_runs', cfg['allocation']['n_monte_carlo_runs']))

    sweep_df, nt = run_threshold_sensitivity(
        y_true=y_te,
        y_prob=y_prob,
        failure_freq=failure_freq,
        n_runs=n_runs,
        threshold=0.5,
        alpha=args.alpha,
        verbose=True,
    )

    os.makedirs(args.out, exist_ok=True)

    sweep_path = os.path.join(args.out, f'{args.key}_threshold_sensitivity.csv')
    sweep_df.to_csv(sweep_path, index=False, float_format='%.6f')
    print(f"\n[threshold_sensitivity] Saved sweep → {sweep_path}")
    print(sweep_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))

    nt_path = os.path.join(args.out, f'{args.key}_near_threshold_stats.csv')
    pd.DataFrame([nt]).to_csv(nt_path, index=False, float_format='%.6f')
    print(f"\n[threshold_sensitivity] Saved near-threshold stats → {nt_path}")


if __name__ == '__main__':
    main()
