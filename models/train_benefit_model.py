"""
models/train_benefit_model.py
==============================
Part B: trains a "benefit model" -- a regressor aimed directly at the
quantity the allocation decision actually needs (marginal_benefit -
marginal_harm at a representative delta), rather than raw failure risk.

Background
----------
analysis/risk_marginal_benefit_correlation.py showed that the existing
failure-risk classifier (xgb_cal) has almost no relationship with true
marginal benefit of added parity (mean Spearman ~0.08 across configs),
despite discriminating raw failure risk well (~0.54). The model was never
trained on the quantity the allocation task actually needs. This script
closes that gap directly: label training (and validation) sequences with
their true paired marginal-benefit-minus-harm at delta=2 (the representative
middle value already used for validation-based tier selection), computed via
the same paired ("common random numbers") simulation design used to fix the
oracle's own noise floor, then train an XGBoost regressor on those labels
using the same sequence-composition features as the failure-risk model.

Usage:
    python models/train_benefit_model.py \\
        --config configs/experiment_config.yaml \\
        --key sub09_k3_simple --delta 2 \\
        --out models/saved_benefit/
"""

import argparse
import os
import pickle
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'allocation'))
sys.path.insert(0, os.path.dirname(__file__))


def _train_benefit_regressor(X_tr, y_tr, X_val, y_val, xgb_cfg, seed):
    """Grid search an XGBRegressor on the benefit label, selecting on val MSE.

    Mirrors allocation/train.py's _train_xgboost grid-search structure, but
    the target is a real-valued benefit score (can be negative), not a
    probability in [0, 1] -- uses a plain squared-error objective, no clip.
    """
    import xgboost as xgb

    best_model, best_mse = None, np.inf
    for depth in xgb_cfg['max_depth']:
        for lr in xgb_cfg['learning_rate']:
            for n_est in xgb_cfg['n_estimators']:
                reg = xgb.XGBRegressor(
                    objective='reg:squarederror',
                    max_depth=depth,
                    learning_rate=lr,
                    n_estimators=n_est,
                    subsample=xgb_cfg.get('subsample', 0.8),
                    colsample_bytree=xgb_cfg.get('colsample_bytree', 0.8),
                    eval_metric='rmse',
                    early_stopping_rounds=xgb_cfg.get('early_stopping_rounds', 20),
                    random_state=seed,
                    verbosity=0,
                )
                reg.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
                preds = reg.predict(X_val)
                mse = float(np.mean((y_val - preds) ** 2))
                if mse < best_mse:
                    best_mse, best_model = mse, reg
    return best_model, best_mse


def main():
    parser = argparse.ArgumentParser(
        description='Train a benefit-aware model (predicts marginal benefit '
                    'of added parity directly, instead of raw failure risk).'
    )
    parser.add_argument('--config', default='configs/experiment_config.yaml')
    parser.add_argument('--key',    required=True)
    parser.add_argument('--delta',  type=int, default=2,
                         help='Representative delta for the benefit label.')
    parser.add_argument('--n-runs', type=int, default=None)
    parser.add_argument('--out',    default='models/saved_benefit/')
    parser.add_argument('--labels-out', default='results/benefit_labels/')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    from dataset_assembler import load_dataset
    from channel_model import build_channel_from_config
    from experiment import estimate_soft_marginal_effects, _ORACLE_SEED_OFFSET

    n_runs    = args.n_runs or cfg['allocation']['n_monte_carlo_runs']
    base_seed = cfg['random_seed']

    parts    = args.key.split('_')
    sub_rate = int(parts[0].replace('sub', '')) / 100
    coverage = int(parts[1].replace('k', ''))

    print(f"[train_benefit_model] Loading {args.key} | delta={args.delta} | runs={n_runs}")
    X_tr, X_val, X_te, y_tr, y_val, y_te, feat_names = load_dataset(args.key, cfg)

    import pandas as pd
    data_path = os.path.join(cfg['paths']['datasets_dir'], f'{args.key}.parquet')
    df        = pd.read_parquet(data_path)
    splits    = pd.read_parquet(
        os.path.join(cfg['paths']['splits_dir'], f'{args.key}_splits.parquet')
    )
    train_idx = splits[splits['split'] == 'train']['index'].values
    val_idx   = splits[splits['split'] == 'val']['index'].values
    sequences_train = df['dna_sequence'].values[train_idx].tolist()
    sequences_val   = df['dna_sequence'].values[val_idx].tolist()

    l_rs_default = cfg['sequence']['l_rs_default']
    l_rs_min     = cfg['rs']['l_rs_min']
    l_rs_max     = cfg['rs']['l_rs_max']

    channel = build_channel_from_config(cfg, sub_rate, seed=base_seed)

    t0 = time.time()
    print(f"[train_benefit_model] Labeling {len(sequences_train)} training sequences "
          f"with soft (kernel-smoothed) marginal benefit-minus-harm ...")
    benefit_tr, harm_tr = estimate_soft_marginal_effects(
        sequences_train, channel, coverage, l_rs_default, args.delta, n_runs,
        base_seed=base_seed, seed_offset=_ORACLE_SEED_OFFSET + 20000,
        l_rs_min=l_rs_min, l_rs_max=l_rs_max,
    )
    y_benefit_tr = benefit_tr - harm_tr
    print(f"  done in {(time.time()-t0)/60:.1f} min")

    t0 = time.time()
    print(f"[train_benefit_model] Labeling {len(sequences_val)} validation sequences ...")
    benefit_val, harm_val = estimate_soft_marginal_effects(
        sequences_val, channel, coverage, l_rs_default, args.delta, n_runs,
        base_seed=base_seed, seed_offset=_ORACLE_SEED_OFFSET + 21000,
        l_rs_min=l_rs_min, l_rs_max=l_rs_max,
    )
    y_benefit_val = benefit_val - harm_val
    print(f"  done in {(time.time()-t0)/60:.1f} min")

    os.makedirs(args.labels_out, exist_ok=True)
    np.savez_compressed(
        os.path.join(args.labels_out, f'{args.key}_delta{args.delta}_benefit_labels.npz'),
        y_benefit_train=y_benefit_tr, y_benefit_val=y_benefit_val,
        train_idx=train_idx, val_idx=val_idx,
    )

    print(f"[train_benefit_model] Training benefit regressor ...")
    xgb_cfg = cfg['models']['xgboost']
    model, val_mse = _train_benefit_regressor(X_tr, y_benefit_tr, X_val, y_benefit_val, xgb_cfg, base_seed)
    print(f"  val MSE={val_mse:.6f}")

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f'{args.key}_delta{args.delta}_benefit_model.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"[train_benefit_model] Saved -> {out_path}")


if __name__ == '__main__':
    main()
