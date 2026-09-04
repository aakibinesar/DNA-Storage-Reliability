"""
analysis/validate_benefit_model.py
====================================
Part B validation: for every config, compares the new benefit-aware model
against the old failure-risk model, both scored against the true (hard-
threshold, operationally meaningful) marginal benefit on the held-out test
set -- the test set was never used for either model's training or labeling.

Usage:
    python analysis/validate_benefit_model.py \\
        --config configs/experiment_config.yaml \\
        --delta 2 --workers 5 \\
        --out results/benefit_model_validation/
"""

import argparse
import os
import pickle
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
    key, delta, config_path, benefit_models_dir = args
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

    try:
        X_tr, X_val, X_te, y_tr, y_val, y_te, feat_names = load_dataset(key, cfg)

        models = load_models('models/saved/', key)
        xgb_cal = models.get('xgboost')
        if xgb_cal is None:
            return {'key': key, 'delta': delta, 'status': 'no_xgb_model'}
        pred_risk_model = xgb_cal.predict_proba(X_te)

        benefit_model_path = os.path.join(
            benefit_models_dir, f'{key}_delta{delta}_benefit_model.pkl')
        if not os.path.exists(benefit_model_path):
            return {'key': key, 'delta': delta, 'status': 'no_benefit_model'}
        with open(benefit_model_path, 'rb') as f:
            benefit_model = pickle.load(f)
        pred_benefit_model = benefit_model.predict(X_te)

        channel = build_channel_from_config(cfg, sub_rate, seed=base_seed)
        data_path = os.path.join(cfg['paths']['datasets_dir'], f'{key}.parquet')
        df = pd.read_parquet(data_path)
        splits = pd.read_parquet(
            os.path.join(cfg['paths']['splits_dir'], f'{key}_splits.parquet'))
        test_idx = splits[splits['split'] == 'test']['index'].values
        sequences = df['dna_sequence'].values[test_idx].tolist()

        benefit_true, harm_true = estimate_paired_marginal_effects(
            sequences, channel, coverage, l_rs_default, delta, n_runs,
            base_seed=base_seed, seed_offset=_ORACLE_SEED_OFFSET,
            l_rs_min=l_rs_min, l_rs_max=l_rs_max,
        )
        y_true = benefit_true - harm_true

        rho_risk, p_risk = spearmanr(pred_risk_model, y_true)
        rho_benefit, p_benefit = spearmanr(pred_benefit_model, y_true)

        return {
            'key': key, 'delta': delta, 'status': 'ok', 'n': len(y_te),
            'spearman_old_risk_model': float(rho_risk),
            'p_old_risk_model': float(p_risk) if np.isfinite(p_risk) else np.nan,
            'spearman_new_benefit_model': float(rho_benefit),
            'p_new_benefit_model': float(p_benefit) if np.isfinite(p_benefit) else np.nan,
        }
    except Exception as e:
        return {'key': key, 'delta': delta, 'status': f'error: {e}'}


def main():
    parser = argparse.ArgumentParser(
        description='Validate the benefit-aware model against the old '
                    'failure-risk model, scored against true test-set benefit.'
    )
    parser.add_argument('--config', default='configs/experiment_config.yaml')
    parser.add_argument('--delta', type=int, default=2)
    parser.add_argument('--workers', type=int, default=5)
    parser.add_argument('--benefit-models-dir', default='models/saved_benefit/')
    parser.add_argument('--out', default='results/benefit_model_validation/')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    sub_rates = cfg['channel']['substitution_rates']
    coverages = cfg['coverage_depths']
    encodings = cfg['sequence']['encoding_schemes']
    keys = [f'sub{int(s*100):02d}_k{k}_{e}'
            for s in sub_rates for k in coverages for e in encodings]

    jobs = [(key, args.delta, args.config, args.benefit_models_dir) for key in keys]

    print(f"[validate_benefit_model] {len(jobs)} configs, {args.workers} workers ...")
    t_start = time.time()
    results = []
    with Pool(processes=args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(_process_one, jobs)):
            results.append(res)
            status = res.get('status')
            tag = 'OK' if status == 'ok' else f'FAIL ({status})'
            print(f"  [{i+1}/{len(jobs)}] {res['key']}: {tag}")

    print(f"\n[validate_benefit_model] Done in {(time.time()-t_start)/60:.1f} min")

    os.makedirs(args.out, exist_ok=True)
    df = pd.DataFrame(results)
    out_path = os.path.join(args.out, 'benefit_model_validation.csv')
    df.to_csv(out_path, index=False, float_format='%.6f')
    print(f"[validate_benefit_model] Saved {len(df)} rows -> {out_path}")

    ok = df[df['status'] == 'ok']
    if len(ok):
        n_improved = int((ok['spearman_new_benefit_model'] > ok['spearman_old_risk_model']).sum())
        print(f"\n  Old risk model:    mean={ok['spearman_old_risk_model'].mean():.3f}, "
              f"median={ok['spearman_old_risk_model'].median():.3f}")
        print(f"  New benefit model: mean={ok['spearman_new_benefit_model'].mean():.3f}, "
              f"median={ok['spearman_new_benefit_model'].median():.3f}")
        print(f"  Benefit model improves on {n_improved}/{len(ok)} configs")


if __name__ == '__main__':
    main()
