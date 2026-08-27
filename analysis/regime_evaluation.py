"""
analysis/regime_evaluation.py
==============================
R11 mitigation: report all Layer 1 evaluation metrics stratified by
failure-frequency regime, across all 28 dataset configurations.

Background
----------
Aggregate AUROC, F1, ECE and Brier reported over the full test set are
dominated by the under-failure and saturated regimes:

  • under_failure (ff < lo): model outputs near-0 for near-0 labels —
    accuracy is trivially high; AUROC is degenerate (all-negative).
  • saturated (ff > hi): symmetric — model outputs near-1 for near-1 labels.
  • informative (lo <= ff <= hi): the regime where allocation decisions are
    actually made and where model quality genuinely matters.

Aggregate metrics across all three regimes therefore hide whether the model
is useful precisely where it needs to be.

This script produces per-regime metrics for every config so the paper can
report honest performance numbers: "AUROC in the informative regime = X".
It also warns when the informative regime is small relative to the test set.

Regime thresholds
-----------------
Read from cfg['evaluation']['regime_lo'] (default 0.15) and
cfg['evaluation']['regime_hi'] (default 0.85).  These are set in
configs/experiment_config.yaml (R11 addition).

Outputs
-------
  <out>/regime_evaluation_all.csv     — all configs × all regimes in one table
  <out>/<key>_regime_eval.csv         — per-key per-regime metric table

Usage (CLI):
    python analysis/regime_evaluation.py \\
        --config configs/experiment_config.yaml \\
        --models-dir models/saved/ \\
        --out results/regime_evaluation/
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))

_INFORMATIVE_WARNING_THRESHOLD = 0.10  # warn if informative < 10% of test set


# -- Core analysis (used by CLI and pipeline stage) ----------------------------

def run_regime_evaluation(
    failure_freq: np.ndarray,
    y_prob: np.ndarray,
    cfg: dict,
    key: str,
    model_name: str = 'xgb_cal',
    verbose: bool = True,
) -> pd.DataFrame:
    """Stratified Layer 1 evaluation for one config key.

    Parameters
    ----------
    failure_freq : continuous failure_freq values for the test set (N,)
    y_prob       : model predicted probabilities (N,)
    cfg          : experiment config dict (regime thresholds read from here)
    key          : dataset config key, e.g. 'sub09_k5_simple'
    model_name   : label column value

    Returns
    -------
    DataFrame with per-regime rows, prepended with 'key' and 'model' columns.
    """
    from evaluate import stratified_evaluation

    lo     = cfg.get('evaluation', {}).get('regime_lo', 0.15)
    hi     = cfg.get('evaluation', {}).get('regime_hi', 0.85)
    n_bins = cfg.get('evaluation', {}).get('ece_n_bins', 10)

    rows = stratified_evaluation(
        failure_freq, y_prob,
        regime_thresholds=(lo, hi),
        n_bins=n_bins,
    )

    df = pd.DataFrame(rows)
    df.insert(0, 'key', key)
    df.insert(1, 'model', model_name)
    df['regime_lo'] = lo
    df['regime_hi'] = hi

    if verbose:
        n_total = int(df.loc[df['regime'] == 'aggregate', 'n'].values[0])
        info_row = df[df['regime'] == 'informative']
        n_info   = int(info_row['n'].values[0]) if len(info_row) else 0
        pct_info = n_info / n_total * 100 if n_total > 0 else 0.0

        if pct_info < _INFORMATIVE_WARNING_THRESHOLD * 100:
            print(f"  [WARN] Informative regime is only {pct_info:.1f}% of the test set — "
                  f"aggregate metrics are dominated by degenerate sequences.")

        _print_regime_table(df, key)

    return df


def _print_regime_table(df: pd.DataFrame, key: str):
    """Print a compact per-regime metric summary."""
    hdr = (f"  {'Regime':<15} {'N':>6} {'prev':>6} "
           f"{'AUROC':>7} {'F1':>7} {'ECE':>7} {'Brier':>7} {'degen':>6}")
    print(hdr)
    print(f"  {'-' * 66}")
    for _, r in df.iterrows():
        n = int(r['n'])
        def _fmt(v):
            return f"{v:.4f}" if (n > 0 and np.isfinite(v)) else "—"
        degen = "YES" if (n > 0 and r.get('is_degenerate', False)) else ("—" if n == 0 else "no")
        print(f"  {r['regime']:<15} {n:>6} {_fmt(r['class_prevalence']):>6} "
              f"{_fmt(r['auroc']):>7} {_fmt(r['f1']):>7} "
              f"{_fmt(r['ece']):>7} {_fmt(r['brier']):>7} {degen:>6}")


# -- CLI entry point -----------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='R11: stratified Layer 1 evaluation across all configs.'
    )
    parser.add_argument('--config',     default='configs/experiment_config.yaml')
    parser.add_argument('--models-dir', default='models/saved/')
    parser.add_argument('--out',        default='results/regime_evaluation/')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    from dataset_assembler import load_dataset
    from train import load_models

    sub_rates = cfg['channel']['substitution_rates']
    coverages = cfg['coverage_depths']
    encodings = cfg['sequence']['encoding_schemes']
    all_keys  = [
        f'sub{int(s * 100):02d}_k{k}_{e}'
        for s in sub_rates for k in coverages for e in encodings
    ]

    os.makedirs(args.out, exist_ok=True)
    all_dfs = []

    for key in all_keys:
        model_path = os.path.join(args.models_dir, f'{key}_xgboost.pkl')
        if not os.path.exists(os.path.join(
                os.path.dirname(__file__), '..', model_path)):
            # also check relative to cwd
            if not os.path.exists(model_path):
                print(f"[regime_evaluation] {key}: model not found — skipping.")
                continue

        print(f"[regime_evaluation] {key}")
        try:
            X_tr, X_val, X_te, y_tr, y_val, y_te, feat_names = load_dataset(key, cfg)
            models  = load_models(args.models_dir, key)
            xgb_cal = models.get('xgboost')
            if xgb_cal is None:
                print(f"  [WARN] XGBoost model missing for {key}")
                continue

            y_prob       = xgb_cal.predict_proba(X_te)
            failure_freq = np.asarray(y_te, dtype=float)  # IS failure_freq after R1

            df_key = run_regime_evaluation(
                failure_freq=failure_freq,
                y_prob=y_prob,
                cfg=cfg,
                key=key,
                verbose=True,
            )

            per_key_path = os.path.join(args.out, f'{key}_regime_eval.csv')
            df_key.to_csv(per_key_path, index=False, float_format='%.6f')
            all_dfs.append(df_key)

        except Exception as e:
            import traceback
            print(f"  [ERROR] {key}: {e}")
            traceback.print_exc()

    if not all_dfs:
        print("[regime_evaluation] No results produced — are models trained?")
        return

    combined  = pd.concat(all_dfs, ignore_index=True)
    out_path  = os.path.join(args.out, 'regime_evaluation_all.csv')
    combined.to_csv(out_path, index=False, float_format='%.6f')
    print(f"\n[regime_evaluation] Saved combined table "
          f"({len(combined)} rows × {len(combined.columns)} cols) -> {out_path}")

    # Cross-config summary
    info_df = combined[combined['regime'] == 'informative']
    agg_df  = combined[combined['regime'] == 'aggregate']

    def _stat(col, df):
        vals = df[col].dropna()
        return f"{vals.mean():.4f} ± {vals.std():.4f}" if len(vals) else "—"

    print(f"\n  Cross-config summary (N={len(info_df)} configs with informative regime):")
    print(f"  {'Metric':<12} {'Informative':>18} {'Aggregate':>18}")
    print(f"  {'-' * 50}")
    for metric in ('auroc', 'f1', 'ece', 'brier'):
        print(f"  {metric:<12} {_stat(metric, info_df):>18} {_stat(metric, agg_df):>18}")

    # Flag configs where informative fraction is below warning threshold
    n_warn = int((info_df['frac_total'] < _INFORMATIVE_WARNING_THRESHOLD).sum())
    if n_warn:
        print(f"\n  [WARN] {n_warn} configs have informative fraction < "
              f"{_INFORMATIVE_WARNING_THRESHOLD:.0%} — aggregate reporting is misleading "
              f"for these configs.")


if __name__ == '__main__':
    main()
