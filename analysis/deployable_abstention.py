"""
analysis/deployable_abstention.py
=================================
Follow-up to the deployable allocation experiments: does giving the
validation-based budget selection an *abstain* option (tier_fraction = 0,
i.e. plain uniform allocation) remove the losses that forced reallocation
causes in configs with little or no exploitable signal?

Background
----------
allocation/experiment.py chooses the deployable reallocation budget from
tier_grid = (0.05, 0.10, 0.20, 0.30) by minimising validation OFR. That grid
has no zero, so the policy can never decline to reallocate, even when every
candidate hurts on validation. This script re-scores the validation OFR for
the same candidates *plus* 0.0, using exactly the same seeds, scores and
channel as the original selection (validation byte errors depend only on the
config, not on delta, so they are simulated once per key and reused for all
deltas and both models).

The test-set outcome of any candidate already exists in results/allocation/
(ofr_*_deployable arrays were computed for the originally selected tier
fraction; candidates are unchanged, so an abstention-enabled policy that picks
a non-zero tier fraction picks the same one as before, and one that picks 0
reproduces uniform exactly). Aggregation therefore needs no new test-set
simulation: see manuscript/tools/abstention_summary.py.

Output: <out>/<key>_abstention.csv  with one row per (delta, model, tier_fraction):
    key, delta, model ('risk' | 'benefit'), tier_fraction, val_ofr, n_val, n_runs

Usage:
    python analysis/deployable_abstention.py --key sub09_k3_simple --out results/deployable_abstention/
"""
import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, 'src')
sys.path.insert(0, 'models')
sys.path.insert(0, 'allocation')

TIER_GRID = (0.0, 0.05, 0.10, 0.20, 0.30)
VAL_SEED_OFFSET = 9000  # must match select_tier_fraction_on_validation()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/experiment_config.yaml')
    parser.add_argument('--key', required=True)
    parser.add_argument('--models-dir', default='models/saved/')
    parser.add_argument('--benefit-models-dir', default='models/saved_benefit/')
    parser.add_argument('--out', default='results/deployable_abstention/')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    from channel_model import build_channel_from_config
    from dataset_assembler import load_dataset
    from experiment import _precompute_byte_errors
    from mechanism import AllocationMechanism
    from train import load_models

    n_runs = cfg['allocation']['n_monte_carlo_runs']
    base_seed = cfg['random_seed']
    parts = args.key.split('_')
    sub_rate = int(parts[0].replace('sub', '')) / 100
    coverage = int(parts[1].replace('k', ''))

    _, X_val, _, _, _, _, _ = load_dataset(args.key, cfg)
    models = load_models(args.models_dir, args.key)
    risk_val = models['xgboost'].predict_proba(X_val)

    data_path = os.path.join(cfg['paths']['datasets_dir'], f'{args.key}.parquet')
    df = pd.read_parquet(data_path)
    splits = pd.read_parquet(os.path.join(cfg['paths']['splits_dir'], f'{args.key}_splits.parquet'))
    val_idx = splits[splits['split'] == 'val']['index'].values
    sequences_val = df['dna_sequence'].values[val_idx].tolist()

    l_rs_default = cfg['sequence']['l_rs_default']
    l_rs_min = cfg['rs']['l_rs_min']
    l_rs_max = cfg['rs']['l_rs_max']

    channel = build_channel_from_config(cfg, sub_rate, seed=base_seed)
    print(f'[abstention] {args.key}: simulating validation byte errors '
          f'({n_runs} runs x {len(sequences_val)} sequences) ...', flush=True)
    byte_errs = []
    for run_idx in range(n_runs):
        run_channel = channel.clone(seed=base_seed + VAL_SEED_OFFSET + run_idx)
        byte_errs.append(_precompute_byte_errors(sequences_val, run_channel, coverage))

    rows = []
    for delta in cfg['allocation']['delta_values']:
        mech = AllocationMechanism(l_rs_default, delta, l_rs_min, l_rs_max)
        scorers = {'risk': risk_val}
        bpath = os.path.join(args.benefit_models_dir, f'{args.key}_delta{delta}_benefit_model.pkl')
        if os.path.exists(bpath):
            with open(bpath, 'rb') as f:
                scorers['benefit'] = pickle.load(f).predict(X_val)
        for name, scores in scorers.items():
            for tf in TIER_GRID:
                alloc = mech.allocate(scores, tier_fraction=tf)
                caps = alloc // 2
                ofr = float(np.mean([np.mean(be > caps) for be in byte_errs]))
                rows.append(dict(key=args.key, delta=delta, model=name, tier_fraction=tf,
                                 val_ofr=ofr, n_val=len(sequences_val), n_runs=n_runs))

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f'{args.key}_abstention.csv')
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f'[abstention] wrote {out_path}', flush=True)


if __name__ == '__main__':
    main()
