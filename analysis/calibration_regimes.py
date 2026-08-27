"""
analysis/calibration_regimes.py
================================
R9 mitigation: report calibration quality (ECE, Brier) stratified by
failure-frequency regime, with bootstrap 95% confidence intervals and
class prevalence per stratum.

Background
----------
Aggregate ECE reported over the full test set is dominated by the
under-failure and saturated regimes, where calibration is trivially good:

  • under_failure (ff < lo): model outputs near-0, labels ~= 0 -> small ECE
    regardless of discrimination ability.
  • saturated (ff > hi): symmetric — model outputs near-1, labels ~= 1.
  • informative (lo <= ff <= hi): the regime where the model's predictions
    actually matter for downstream allocation decisions.

Reporting only aggregate ECE can therefore hide poor calibration in the
informative regime.  This script exposes the delta:

    Delta_ECE = ECE_informative - ECE_aggregate

and provides bootstrap 95% CIs so the magnitude is statistically bounded.

Regime thresholds
-----------------
Default: lo=0.15, hi=0.85.  Sequences at exactly lo or hi are included
in the informative regime.  Override with --lo and --hi.

Outputs
-------
  <out>/<key>_calibration_regimes.csv    — per-regime ECE/Brier table
  <out>/<key>_calibration_delta.csv      — Delta_ECE and Delta_Brier summary

Usage (CLI):
    python analysis/calibration_regimes.py \\
        --config configs/experiment_config.yaml \\
        --key sub09_k5_simple \\
        --models-dir models/saved/ \\
        --out results/calibration_regimes/
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))


# -- Core analysis (used by CLI and pipeline stage) ----------------------------

def run_calibration_regimes(
    failure_freq: np.ndarray,
    y_prob: np.ndarray,
    lo: float = 0.15,
    hi: float = 0.85,
    n_bootstrap: int = 500,
    alpha: float = 0.05,
    seed: int = 0,
    verbose: bool = True,
) -> tuple:
    """Compute regime-stratified calibration table and delta summary.

    Parameters
    ----------
    failure_freq : continuous failure_freq values for the test set (N,)
    y_prob       : model predicted probabilities (N,)

    Returns
    -------
    (regime_df, delta_df)
      regime_df : DataFrame with ECE/Brier per regime (+ bootstrap CIs)
      delta_df  : DataFrame summarising Delta_ECE and Delta_Brier
    """
    from evaluate import calibration_by_regime

    rows = calibration_by_regime(
        failure_freq, y_prob,
        regime_thresholds=(lo, hi),
        n_bootstrap=n_bootstrap,
        alpha=alpha,
        seed=seed,
    )
    regime_df = pd.DataFrame(rows)

    # Compute calibration deltas: informative vs aggregate
    def _get(df, regime, col):
        row = df[df['regime'] == regime]
        return float(row[col].values[0]) if len(row) > 0 else float('nan')

    ece_agg  = _get(regime_df, 'aggregate',  'ece')
    ece_inf  = _get(regime_df, 'informative', 'ece')
    ece_lo   = _get(regime_df, 'informative', 'ece_ci_lo')
    ece_hi_  = _get(regime_df, 'informative', 'ece_ci_hi')

    brier_agg = _get(regime_df, 'aggregate',  'brier')
    brier_inf = _get(regime_df, 'informative', 'brier')
    brier_lo  = _get(regime_df, 'informative', 'brier_ci_lo')
    brier_hi_ = _get(regime_df, 'informative', 'brier_ci_hi')

    delta_df = pd.DataFrame([{
        'ece_aggregate'              : ece_agg,
        'ece_informative'            : ece_inf,
        'ece_informative_ci_lo'      : ece_lo,
        'ece_informative_ci_hi'      : ece_hi_,
        'delta_ece'                  : ece_inf - ece_agg,
        'brier_aggregate'            : brier_agg,
        'brier_informative'          : brier_inf,
        'brier_informative_ci_lo'    : brier_lo,
        'brier_informative_ci_hi'    : brier_hi_,
        'delta_brier'                : brier_inf - brier_agg,
        'regime_lo'                  : lo,
        'regime_hi'                  : hi,
        'n_bootstrap'                : n_bootstrap,
        'alpha'                      : alpha,
    }])

    if verbose:
        print(f"  Regime thresholds: lo={lo}  hi={hi}")
        print(f"  {'Regime':<15} {'N':>6} {'frac':>6} {'class_prev':>11} "
              f"{'ECE':>7} {'95% CI':>16} {'Brier':>7} {'95% CI':>16}")
        print(f"  {'-'*88}")
        for _, r in regime_df.iterrows():
            n      = int(r['n'])
            frac   = f"{r['frac_total']:.2%}" if n > 0 else "—"
            prev   = f"{r['class_prevalence']:.3f}"  if n > 0 else "—"
            ece_s  = f"{r['ece']:.4f}" if n > 0 else "—"
            ece_ci = (f"[{r['ece_ci_lo']:.4f}, {r['ece_ci_hi']:.4f}]"
                      if n >= 5 else "—")
            brier_s  = f"{r['brier']:.4f}" if n > 0 else "—"
            brier_ci = (f"[{r['brier_ci_lo']:.4f}, {r['brier_ci_hi']:.4f}]"
                        if n >= 5 else "—")
            print(f"  {r['regime']:<15} {n:>6} {frac:>6} {prev:>11} "
                  f"{ece_s:>7} {ece_ci:>16} {brier_s:>7} {brier_ci:>16}")
        print(f"\n  Delta_ECE   (informative - aggregate): "
              f"{delta_df['delta_ece'].values[0]:+.4f}")
        print(f"  Delta_Brier (informative - aggregate): "
              f"{delta_df['delta_brier'].values[0]:+.4f}")

    return regime_df, delta_df


# -- CLI entry point -----------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='R9: regime-stratified calibration with bootstrap CIs.'
    )
    parser.add_argument('--config',      default='configs/experiment_config.yaml')
    parser.add_argument('--key',         required=True,
                        help='Dataset key, e.g. sub09_k5_simple')
    parser.add_argument('--models-dir',  default='models/saved/')
    parser.add_argument('--out',         default='results/calibration_regimes/')
    parser.add_argument('--lo',          type=float, default=0.15,
                        help='Lower regime threshold (default 0.15)')
    parser.add_argument('--hi',          type=float, default=0.85,
                        help='Upper regime threshold (default 0.85)')
    parser.add_argument('--n-bootstrap', type=int, default=500,
                        help='Bootstrap replicates for CI (default 500; 0 = skip CIs)')
    parser.add_argument('--alpha',       type=float, default=0.05,
                        help='CI significance level (default 0.05 -> 95%%)')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    from dataset_assembler import load_dataset
    from train import load_models

    parts    = args.key.split('_')
    sub_rate = int(parts[0].replace('sub', '')) / 100
    coverage = int(parts[1].replace('k', ''))

    print(f"[calibration_regimes] {args.key}  sub_rate={sub_rate}  coverage={coverage}")

    X_tr, X_val, X_te, y_tr, y_val, y_te, feat_names = load_dataset(args.key, cfg)
    models  = load_models(args.models_dir, args.key)
    xgb_cal = models.get('xgboost')

    if xgb_cal is None:
        print("[calibration_regimes] No XGBoost model found — skipping.")
        return

    y_prob = xgb_cal.predict_proba(X_te)

    # Load continuous failure_freq for the test set (y_te IS failure_freq after R1)
    # Use y_te directly — it is the regression target = failure_freq.
    failure_freq = np.asarray(y_te, dtype=float)

    seed = cfg.get('random_seed', 0)

    regime_df, delta_df = run_calibration_regimes(
        failure_freq=failure_freq,
        y_prob=y_prob,
        lo=args.lo,
        hi=args.hi,
        n_bootstrap=args.n_bootstrap,
        alpha=args.alpha,
        seed=seed,
        verbose=True,
    )

    os.makedirs(args.out, exist_ok=True)

    regime_path = os.path.join(args.out, f'{args.key}_calibration_regimes.csv')
    regime_df.to_csv(regime_path, index=False, float_format='%.6f')
    print(f"\n[calibration_regimes] Saved regime table -> {regime_path}")

    delta_path = os.path.join(args.out, f'{args.key}_calibration_delta.csv')
    delta_df.to_csv(delta_path, index=False, float_format='%.6f')
    print(f"[calibration_regimes] Saved delta summary -> {delta_path}")


if __name__ == '__main__':
    main()
