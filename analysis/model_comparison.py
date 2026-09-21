"""
analysis/model_comparison.py
=============================
Unified comparison of the trained model tiers -- XGBoost, Random Forest,
Logistic Regression, and (optionally, if --cnn-models-dir is given and
{key}_cnn.pkl files exist there, see models/train_cnn.py) a small 1D CNN
over raw sequence -- on the same footing: the regime-stratified Layer 1
metric suite already used for XGBoost alone in `analysis/regime_evaluation.py`
(see that module for why aggregate metrics over- or under-state genuine
model quality).

XGBoost, Random Forest, and the CNN are calibrated regressors on continuous
failure_freq; Logistic Regression is a binary majority-fail discriminator
trained on the binarised label. All of them nonetheless produce a probability
in [0, 1], so `stratified_evaluation` (regime assignment from failure_freq,
binary labels from failure_freq >= 0.5) applies identically to all -- this is
what makes the comparison fair rather than apples-to-oranges.

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

MIN_CLASS_N = 10  # a config's informative regime is only "usable" for AUROC
# reporting if BOTH classes have at least this many examples. A regime that
# merely satisfies n_total >= 20 (the Failure Regime Map's threshold) can
# still have as few as 1 positive example -- an AUROC computed from that is
# closer to noise than to evidence, regardless of its numeric value (e.g.
# sub12_k3_constrained: n=251 total but only 1 positive -> AUROC=1.000 is a
# small-sample artifact, not a real result).

MODEL_LABELS = {
    'xgboost': 'XGBoost',
    'random_forest': 'Random Forest',
    'logistic_regression': 'Logistic Regression',
    'cnn': 'CNN (sequence)',
    'cnn_v2': 'BigCNN (deeper, attention)',
}


def _lr_predict_proba(lr_model, lr_scaler, X_te: np.ndarray) -> np.ndarray:
    """Same fallback pattern gate_check.py uses for a degenerate LR model."""
    X_s = lr_scaler.transform(X_te)
    try:
        return lr_model.predict_proba(X_s)[:, 1]
    except (IndexError, ValueError):
        classes = getattr(lr_model, 'classes_', [None])
        return np.full(X_te.shape[0], 1.0 if classes[0] == 1 else 0.0)


def run_model_comparison(
    cfg: dict, models_dir: str, out_dir: str, cnn_models_dir: str = None,
    cnn_v2_models_dir: str = None,
) -> pd.DataFrame:
    from dataset_assembler import load_dataset, load_dataset_sequences
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

        if cnn_models_dir is not None:
            cnn_path = os.path.join(cnn_models_dir, f'{key}_cnn.pkl')
            if os.path.exists(cnn_path):
                import pickle
                from train_cnn import one_hot_encode
                with open(cnn_path, 'rb') as f:
                    cnn_model = pickle.load(f)
                _, _, seq_te, _, _, _ = load_dataset_sequences(key, cfg)
                X_te_onehot = one_hot_encode(seq_te, cfg['sequence']['seq_len_bases'])
                probs['cnn'] = cnn_model.predict_proba(X_te_onehot)

        if cnn_v2_models_dir is not None:
            cnn_v2_path = os.path.join(cnn_v2_models_dir, f'{key}_cnn_v2.pkl')
            if os.path.exists(cnn_v2_path):
                import pickle
                from train_cnn import one_hot_encode
                with open(cnn_v2_path, 'rb') as f:
                    cnn_v2_model = pickle.load(f)
                _, _, seq_te, _, _, _ = load_dataset_sequences(key, cfg)
                X_te_onehot = one_hot_encode(seq_te, cfg['sequence']['seq_len_bases'])
                probs['cnn_v2'] = cnn_v2_model.predict_proba(X_te_onehot)

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
    n_pos = (combined['n'] * combined['class_prevalence']).round()
    combined['min_class_n'] = np.minimum(n_pos, combined['n'] - n_pos)
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
    subset of configs where AUROC is both defined AND statistically
    trustworthy (min_class_n >= MIN_CLASS_N) -- a defined-but-tiny-minority-
    class AUROC (e.g. 1 positive example) is not usable evidence just
    because it happens to be a number."""
    info = combined[combined['regime'] == 'informative']
    rows = []
    for model_name, grp in info.groupby('model'):
        usable = grp[grp['auroc'].notna() & (grp['min_class_n'] >= MIN_CLASS_N)]
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
    print(f"\n  Informative-regime summary (usable = AUROC defined AND "
          f"min_class_n >= {MIN_CLASS_N}):")
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
    info = combined[(combined['regime'] == 'informative') & combined['auroc'].notna()
                     & (combined['min_class_n'] >= MIN_CLASS_N)]
    pivot = info.pivot(index='key', columns='model', values='auroc')
    contested = pivot.dropna(thresh=2)
    if len(contested):
        wins = contested.idxmax(axis=1).value_counts()
        print(f"\n  Head-to-head best-AUROC wins ({len(contested)} configs with >=2 usable models):")
        for model_name, count in wins.items():
            print(f"    {MODEL_LABELS.get(model_name, model_name):<22} {count} / {len(contested)}")


N_BOOTSTRAP = 2000


def _collect_informative_predictions(cfg: dict, models_dir: str) -> dict:
    """Per-config informative-regime binary labels + XGBoost/RF probabilities,
    restricted to configs where BOTH models have a non-degenerate (mixed-class)
    informative regime -- i.e. the same 12/28 'usable' configs used elsewhere.
    """
    from dataset_assembler import load_dataset
    from train import load_models

    lo = cfg.get('evaluation', {}).get('regime_lo', 0.15)
    hi = cfg.get('evaluation', {}).get('regime_hi', 0.85)

    sub_rates = cfg['channel']['substitution_rates']
    coverages = cfg['coverage_depths']
    encodings = cfg['sequence']['encoding_schemes']
    all_keys  = [
        f'sub{int(s * 100):02d}_k{k}_{e}'
        for s in sub_rates for k in coverages for e in encodings
    ]

    out = {}
    for key in all_keys:
        try:
            X_tr, X_val, X_te, y_tr, y_val, y_te, feat_names = load_dataset(key, cfg)
            models = load_models(models_dir, key)
        except Exception:
            continue
        xgb_cal = models.get('xgboost')
        rf_cal  = models.get('random_forest')
        if xgb_cal is None or rf_cal is None:
            continue

        ff = np.asarray(y_te, dtype=float)
        mask = (ff >= lo) & (ff <= hi)
        if mask.sum() < 2:
            continue
        y_bin = (ff[mask] >= 0.5).astype(int)
        n_pos, n_neg = int(y_bin.sum()), int((1 - y_bin).sum())
        if min(n_pos, n_neg) < MIN_CLASS_N:
            continue  # AUROC undefined, or defined but statistically untrustworthy

        out[key] = {
            'y_bin': y_bin,
            'xgboost': np.clip(xgb_cal.predict_proba(X_te)[mask], 0.0, 1.0),
            'random_forest': np.clip(rf_cal.predict_proba(X_te)[mask], 0.0, 1.0),
        }
    return out


def bootstrap_model_comparison(
    cfg: dict, models_dir: str, out_dir: str,
    n_bootstrap: int = N_BOOTSTRAP, seed: int = 42,
) -> pd.DataFrame:
    """Reviewer-grade statistical comparison of RF vs. XGBoost informative-
    regime AUROC: per-config bootstrap CIs, plus a two-level (config +
    within-config) cluster bootstrap for the overall mean difference.

    A paired Wilcoxon test on 12 config-level point estimates (used in the
    initial pass at this comparison) is fair but low-powered by construction
    -- it throws away every within-config sample and treats 12 configs as
    the entire sample size. Bootstrapping the actual test sequences (not
    just the config means) uses the real amount of data actually available
    and gives a proper confidence interval instead of a single p-value.
    """
    from sklearn.metrics import roc_auc_score

    os.makedirs(out_dir, exist_ok=True)
    data = _collect_informative_predictions(cfg, models_dir)
    keys = sorted(data.keys())
    print(f"[model_comparison] Bootstrap comparison over {len(keys)} usable configs, "
          f"n_bootstrap={n_bootstrap}")

    rng = np.random.default_rng(seed)

    # -- Per-config bootstrap CIs --------------------------------------------
    rows = []
    per_config_diffs = {}  # key -> array of bootstrap AUROC diffs (for the cluster step)
    for key in keys:
        d = data[key]
        y_bin, p_xgb, p_rf = d['y_bin'], d['xgboost'], d['random_forest']
        n = len(y_bin)
        diffs = np.full(n_bootstrap, np.nan)
        for b in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            yb = y_bin[idx]
            if len(np.unique(yb)) < 2:
                continue
            auc_rf  = roc_auc_score(yb, p_rf[idx])
            auc_xgb = roc_auc_score(yb, p_xgb[idx])
            diffs[b] = auc_rf - auc_xgb
        valid = diffs[~np.isnan(diffs)]
        per_config_diffs[key] = valid
        point_rf  = roc_auc_score(y_bin, p_rf)
        point_xgb = roc_auc_score(y_bin, p_xgb)
        lo_ci, hi_ci = np.percentile(valid, [2.5, 97.5]) if len(valid) else (np.nan, np.nan)
        rows.append({
            'key': key, 'n': n,
            'auroc_rf': point_rf, 'auroc_xgb': point_xgb,
            'diff_rf_minus_xgb': point_rf - point_xgb,
            'boot_diff_ci_lo': lo_ci, 'boot_diff_ci_hi': hi_ci,
            'ci_excludes_zero': bool(lo_ci > 0 or hi_ci < 0) if np.isfinite(lo_ci) else False,
            'n_valid_bootstraps': len(valid),
        })
    per_config_df = pd.DataFrame(rows)
    per_config_path = os.path.join(out_dir, 'model_comparison_bootstrap_per_config.csv')
    per_config_df.to_csv(per_config_path, index=False, float_format='%.6f')

    # -- Two-level cluster bootstrap for the overall mean difference ---------
    # Resample configs with replacement, and independently resample sequences
    # within each drawn config, so both between-config and within-config
    # uncertainty are propagated into the final interval.
    overall_diffs = np.full(n_bootstrap, np.nan)
    for b in range(n_bootstrap):
        drawn_keys = rng.choice(keys, size=len(keys), replace=True)
        iter_diffs = []
        for key in drawn_keys:
            d = data[key]
            n = len(d['y_bin'])
            idx = rng.integers(0, n, size=n)
            yb = d['y_bin'][idx]
            if len(np.unique(yb)) < 2:
                continue
            auc_rf  = roc_auc_score(yb, d['random_forest'][idx])
            auc_xgb = roc_auc_score(yb, d['xgboost'][idx])
            iter_diffs.append(auc_rf - auc_xgb)
        if iter_diffs:
            overall_diffs[b] = np.mean(iter_diffs)
    valid_overall = overall_diffs[~np.isnan(overall_diffs)]
    overall_lo, overall_hi = np.percentile(valid_overall, [2.5, 97.5])
    overall_mean = valid_overall.mean()
    frac_positive = float((valid_overall > 0).mean())

    summary_row = {
        'n_configs': len(keys),
        'n_bootstrap': n_bootstrap,
        'mean_diff_rf_minus_xgb': overall_mean,
        'ci_95_lo': overall_lo,
        'ci_95_hi': overall_hi,
        'ci_excludes_zero': bool(overall_lo > 0 or overall_hi < 0),
        'frac_bootstraps_rf_better': frac_positive,
    }
    pd.DataFrame([summary_row]).to_csv(
        os.path.join(out_dir, 'model_comparison_bootstrap_overall.csv'),
        index=False, float_format='%.6f',
    )

    n_excl = int(per_config_df['ci_excludes_zero'].sum())
    print(f"\n  Per-config bootstrap CIs saved -> {per_config_path}")
    print(f"  {n_excl} / {len(keys)} configs have a 95% CI excluding zero (RF vs. XGBoost AUROC)")
    print(f"\n  Cluster bootstrap (configs + within-config resampled together):")
    print(f"    mean diff (RF - XGBoost) = {overall_mean:+.4f}")
    print(f"    95% CI = [{overall_lo:+.4f}, {overall_hi:+.4f}]  "
          f"({'excludes zero' if summary_row['ci_excludes_zero'] else 'includes zero'})")
    print(f"    fraction of bootstrap draws favoring RF: {frac_positive:.1%}")

    return per_config_df


def main():
    parser = argparse.ArgumentParser(
        description='Unified XGBoost / Random Forest / Logistic Regression comparison.'
    )
    parser.add_argument('--config',     default='configs/experiment_config.yaml')
    parser.add_argument('--models-dir', default='models/saved/')
    parser.add_argument('--cnn-models-dir', default=None,
                         help='If given, also score any {key}_cnn.pkl found here as a 4th model.')
    parser.add_argument('--cnn-v2-models-dir', default=None,
                         help='If given, also score any {key}_cnn_v2.pkl (BigCNN) found here as a 5th model.')
    parser.add_argument('--out',        default='results/model_comparison/')
    parser.add_argument('--bootstrap',  action='store_true',
                         help='Also run the RF-vs-XGBoost bootstrap CI comparison.')
    parser.add_argument('--n-bootstrap', type=int, default=N_BOOTSTRAP)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_model_comparison(cfg, args.models_dir, args.out, args.cnn_models_dir, args.cnn_v2_models_dir)

    if args.bootstrap:
        bootstrap_model_comparison(cfg, args.models_dir, args.out, args.n_bootstrap)


if __name__ == '__main__':
    main()
