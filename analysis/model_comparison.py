"""
analysis/model_comparison.py
=============================
Unified comparison of the three trained model tiers -- XGBoost, Random
Forest, Logistic Regression -- on the same footing: the regime-stratified
Layer 1 metric suite already used for XGBoost alone in
`analysis/regime_evaluation.py` (see that module for why aggregate metrics
over- or under-state genuine model quality).

XGBoost and Random Forest are calibrated regressors on continuous
failure_freq; Logistic Regression is a binary majority-fail discriminator
trained on the binarised label. All three nonetheless produce a probability
in [0, 1], so `stratified_evaluation` (regime assignment from failure_freq,
binary labels from failure_freq >= 0.5) applies identically to all three --
this is what makes the comparison fair rather than apples-to-oranges.

Outputs
-------
  <out>/model_comparison_all.csv   -- all configs x all models x all regimes
  <out>/model_comparison_summary.csv -- per-model informative-regime summary

Usage (CLI):
    python analysis/model_comparison.py \\
        --config configs/experiment_config.yaml \\
        --models-dir models/saved/ \\
        --out results/model_comparison/
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))

MODEL_LABELS = {
    'xgboost': 'XGBoost',
    'random_forest': 'Random Forest',
    'logistic_regression': 'Logistic Regression',
}


def _lr_predict_proba(lr_model, lr_scaler, X_te: np.ndarray) -> np.ndarray:
    """Same fallback pattern gate_check.py uses for a degenerate LR model."""
    X_s = lr_scaler.transform(X_te)
    try:
        return lr_model.predict_proba(X_s)[:, 1]
    except (IndexError, ValueError):
        classes = getattr(lr_model, 'classes_', [None])
        return np.full(X_te.shape[0], 1.0 if classes[0] == 1 else 0.0)


def run_model_comparison(cfg: dict, models_dir: str, out_dir: str) -> pd.DataFrame:
    from dataset_assembler import load_dataset
    from train import load_models
    from evaluate import stratified_evaluation

    lo     = cfg.get('evaluation', {}).get('regime_lo', 0.15)
    hi     = cfg.get('evaluation', {}).get('regime_hi', 0.85)
    n_bins = cfg.get('evaluation', {}).get('ece_n_bins', 10)

    sub_rates = cfg['channel']['substitution_rates']
    coverages = cfg['coverage_depths']
    encodings = cfg['sequence']['encoding_schemes']
    all_keys  = [
        f'sub{int(s * 100):02d}_k{k}_{e}'
        for s in sub_rates for k in coverages for e in encodings
    ]

    os.makedirs(out_dir, exist_ok=True)
    all_rows = []

    for key in all_keys:
        try:
            X_tr, X_val, X_te, y_tr, y_val, y_te, feat_names = load_dataset(key, cfg)
            models = load_models(models_dir, key)
        except Exception as e:
            print(f"[model_comparison] {key}: could not load — {e}")
            continue

        failure_freq = np.asarray(y_te, dtype=float)

        probs = {}
        xgb_cal = models.get('xgboost')
        rf_cal  = models.get('random_forest')
        lr_model  = models.get('logistic_regression')
        lr_scaler = models.get('logistic_scaler')

        if xgb_cal is not None:
            probs['xgboost'] = xgb_cal.predict_proba(X_te)
        if rf_cal is not None:
            probs['random_forest'] = rf_cal.predict_proba(X_te)
        if lr_model is not None and lr_scaler is not None:
            probs['logistic_regression'] = _lr_predict_proba(lr_model, lr_scaler, X_te)

        if not probs:
            print(f"[model_comparison] {key}: no models found — skipping.")
            continue

        print(f"[model_comparison] {key}: {list(probs.keys())}")
        for model_name, y_prob in probs.items():
            rows = stratified_evaluation(failure_freq, y_prob, (lo, hi), n_bins)
            for r in rows:
                r['key'] = key
                r['model'] = model_name
            all_rows.extend(rows)

    combined = pd.DataFrame(all_rows)
    cols = ['key', 'model', 'regime'] + [c for c in combined.columns if c not in ('key', 'model', 'regime')]
    combined = combined[cols]

    out_path = os.path.join(out_dir, 'model_comparison_all.csv')
    combined.to_csv(out_path, index=False, float_format='%.6f')
    print(f"\n[model_comparison] Saved combined table "
          f"({len(combined)} rows x {len(combined.columns)} cols) -> {out_path}")

    summary = _build_summary(combined)
    summary_path = os.path.join(out_dir, 'model_comparison_summary.csv')
    summary.to_csv(summary_path, index=False, float_format='%.6f')
    print(f"[model_comparison] Saved summary -> {summary_path}")
    _print_summary(summary, combined)

    return combined


def _build_summary(combined: pd.DataFrame) -> pd.DataFrame:
    """Per-model informative-regime summary: mean/median metrics over the
    subset of configs where that model's informative regime is non-degenerate
    (i.e. AUROC is actually defined)."""
    info = combined[combined['regime'] == 'informative']
    rows = []
    for model_name, grp in info.groupby('model'):
        usable = grp[grp['auroc'].notna()]
        rows.append({
            'model': model_name,
            'n_configs_total': grp['key'].nunique(),
            'n_configs_usable': usable['key'].nunique(),
            'mean_auroc': usable['auroc'].mean(),
            'median_auroc': usable['auroc'].median(),
            'mean_f1': usable['f1'].mean(),
            'mean_ece': usable['ece'].mean(),
            'mean_brier': usable['brier'].mean(),
        })
    return pd.DataFrame(rows)


def _print_summary(summary: pd.DataFrame, combined: pd.DataFrame):
    print("\n  Informative-regime summary (usable configs only, i.e. non-degenerate AUROC):")
    print(f"  {'Model':<22} {'n_usable':>9} {'AUROC(mean)':>12} {'AUROC(med)':>11} "
          f"{'F1':>7} {'ECE':>7} {'Brier':>7}")
    print(f"  {'-' * 82}")
    for _, r in summary.iterrows():
        label = MODEL_LABELS.get(r['model'], r['model'])
        print(f"  {label:<22} {int(r['n_configs_usable']):>9} {r['mean_auroc']:>12.4f} "
              f"{r['median_auroc']:>11.4f} {r['mean_f1']:>7.4f} {r['mean_ece']:>7.4f} "
              f"{r['mean_brier']:>7.4f}")

    # Head-to-head: for configs where >=2 models have a usable informative AUROC,
    # which model wins?
    info = combined[(combined['regime'] == 'informative') & combined['auroc'].notna()]
    pivot = info.pivot(index='key', columns='model', values='auroc')
    contested = pivot.dropna(thresh=2)
    if len(contested):
        wins = contested.idxmax(axis=1).value_counts()
        print(f"\n  Head-to-head best-AUROC wins ({len(contested)} configs with >=2 usable models):")
        for model_name, count in wins.items():
            print(f"    {MODEL_LABELS.get(model_name, model_name):<22} {count} / {len(contested)}")


def main():
    parser = argparse.ArgumentParser(
        description='Unified XGBoost / Random Forest / Logistic Regression comparison.'
    )
    parser.add_argument('--config',     default='configs/experiment_config.yaml')
    parser.add_argument('--models-dir', default='models/saved/')
    parser.add_argument('--out',        default='results/model_comparison/')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_model_comparison(cfg, args.models_dir, args.out)


if __name__ == '__main__':
    main()
