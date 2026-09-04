"""
analysis/risk_marginal_benefit_correlation.py
================================================
Diagnoses why the deployed model (xgb_cal) underperforms the oracle allocation
at higher delta values (see results/allocation_significance/).

Background
----------
allocation/significance.py found that xgb_cal-driven allocation is
significantly *worse* than uniform in 29/84 config x delta combinations
(vs. 15/84 significantly better), concentrated at delta in {2, 4} -- the
opposite pattern from the oracle, whose rare losses concentrate at delta=1.

Hypothesis: xgb_cal is trained/calibrated to predict raw failure probability
at the *default* parity level (see allocation/experiment.py:474,
`calibrated_risk = xgb_cal.predict_proba(X_te)`). The oracle instead ranks
sequences by true marginal_benefit - marginal_harm -- the actual change in
failure probability from moving parity by delta. These are conceptually
different targets. This module checks, per config, how well calibrated_risk
(what the model predicts) tracks marginal_benefit - marginal_harm (what the
allocation decision actually needs) via Spearman rank correlation, alongside
a sanity-check correlation against the raw failure label (to confirm the
model discriminates the label fine and the gap is specific to the marginal-
benefit target, not a broken model).

Scope note: this run uses a single representative delta (2) per config,
28 combinations total, rather than the full 84 config x delta grid --
a faster first pass. marginal_benefit/marginal_harm are delta-dependent
(they measure the effect of a delta-sized parity change), so this captures
the target-mismatch story at one delta rather than its full delta-dependence;
rerun with --deltas 1 2 4 for the complete 84-combination picture.

Outputs
-------
  <out>/risk_marginal_benefit_correlation.csv -- one row per (key, delta)

Usage:
    python analysis/risk_marginal_benefit_correlation.py \\
        --config configs/experiment_config.yaml \\
        --deltas 2 \\
        --workers 4 \\
        --out results/risk_marginal_benefit_correlation/
"""

import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'allocation'))


def _process_one(args):
    key, delta, config_path, models_dir = args
    import yaml as _yaml
    from dataset_assembler import load_dataset
    from channel_model import build_channel_from_config
    from train import load_models
    from experiment import estimate_paired_marginal_effects, _ORACLE_SEED_OFFSET

    with open(config_path) as f:
        cfg = _yaml.safe_load(f)

    base_seed = cfg['random_seed']
    l_rs_default = cfg['sequence']['l_rs_default']
    l_rs_min = cfg['rs']['l_rs_min']
    l_rs_max = cfg['rs']['l_rs_max']
    n_runs = cfg['allocation']['n_monte_carlo_runs']

    parts = key.split('_')
    sub_rate = int(parts[0].replace('sub', '')) / 100
    coverage = int(parts[1].replace('k', ''))

    t0 = time.time()
    try:
        X_tr, X_val, X_te, y_tr, y_val, y_te, feat_names = load_dataset(key, cfg)
        models = load_models(models_dir, key)
        xgb_cal = models.get('xgboost')
        if xgb_cal is None:
            return {'key': key, 'delta': delta, 'status': 'no_xgb_model'}
        calibrated_risk = xgb_cal.predict_proba(X_te)

        channel = build_channel_from_config(cfg, sub_rate, seed=base_seed)

        data_path = os.path.join(cfg['paths']['datasets_dir'], f'{key}.parquet')
        df = pd.read_parquet(data_path)
        splits = pd.read_parquet(
            os.path.join(cfg['paths']['splits_dir'], f'{key}_splits.parquet')
        )
        test_idx = splits[splits['split'] == 'test']['index'].values
        sequences = df['dna_sequence'].values[test_idx].tolist()

        marginal_benefits, marginal_harm = estimate_paired_marginal_effects(
            sequences, channel, coverage, l_rs_default, delta, n_runs,
            base_seed=base_seed, seed_offset=_ORACLE_SEED_OFFSET,
            l_rs_min=l_rs_min, l_rs_max=l_rs_max,
        )
        true_value = marginal_benefits - marginal_harm

        rho_benefit, p_benefit = spearmanr(calibrated_risk, true_value)
        rho_label, p_label = spearmanr(calibrated_risk, y_te)

        elapsed = time.time() - t0
        return {
            'key': key, 'delta': delta, 'status': 'ok',
            'n': len(y_te), 'mean_failure_freq': float(np.mean(y_te)),
            'spearman_risk_vs_marginal_benefit': float(rho_benefit),
            'p_marginal_benefit': float(p_benefit),
            'spearman_risk_vs_label': float(rho_label),
            'p_label': float(p_label),
            'elapsed_sec': round(elapsed, 1),
        }
    except Exception as e:
        return {'key': key, 'delta': delta, 'status': f'error: {e}'}


def main():
    parser = argparse.ArgumentParser(
        description='Spearman correlation between model risk score and true '
                    'marginal benefit-of-parity, per config.'
    )
    parser.add_argument('--config', default='configs/experiment_config.yaml')
    parser.add_argument('--models-dir', default='models/saved/')
    parser.add_argument('--deltas', type=int, nargs='+', default=[2])
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--out', default='results/risk_marginal_benefit_correlation/')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    sub_rates = cfg['channel']['substitution_rates']
    coverages = cfg['coverage_depths']
    encodings = cfg['sequence']['encoding_schemes']
    keys = [
        f'sub{int(s*100):02d}_k{k}_{e}'
        for s in sub_rates for k in coverages for e in encodings
    ]

    jobs = [(key, delta, args.config, args.models_dir)
            for key in keys for delta in args.deltas]

    print(f"[risk_marginal_benefit_correlation] {len(jobs)} jobs "
          f"({len(keys)} keys x {len(args.deltas)} delta(s)), {args.workers} workers ...")

    t_start = time.time()
    with Pool(processes=args.workers) as pool:
        results = []
        for i, res in enumerate(pool.imap_unordered(_process_one, jobs)):
            results.append(res)
            status = res.get('status')
            tag = 'OK' if status == 'ok' else f'FAIL ({status})'
            print(f"  [{i+1}/{len(jobs)}] {res['key']} delta={res['delta']}: {tag}")

    elapsed = time.time() - t_start
    print(f"\n[risk_marginal_benefit_correlation] Done in {elapsed/60:.1f} min")

    os.makedirs(args.out, exist_ok=True)
    df = pd.DataFrame(results)
    out_path = os.path.join(args.out, 'risk_marginal_benefit_correlation.csv')
    df.to_csv(out_path, index=False, float_format='%.6f')
    print(f"[risk_marginal_benefit_correlation] Saved {len(df)} rows -> {out_path}")

    ok = df[df['status'] == 'ok']
    if len(ok):
        print(f"\n  Spearman(risk, marginal_benefit) across {len(ok)} configs: "
              f"mean={ok['spearman_risk_vs_marginal_benefit'].mean():.3f}, "
              f"median={ok['spearman_risk_vs_marginal_benefit'].median():.3f}")
        print(f"  Spearman(risk, raw label) across {len(ok)} configs: "
              f"mean={ok['spearman_risk_vs_label'].mean():.3f}, "
              f"median={ok['spearman_risk_vs_label'].median():.3f}")
        n_sig_benefit = int((ok['p_marginal_benefit'] < 0.05).sum())
        print(f"  Configs with significant risk-vs-marginal-benefit correlation "
              f"(p<0.05, uncorrected): {n_sig_benefit}/{len(ok)}")


if __name__ == '__main__':
    main()
