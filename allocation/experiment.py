"""
allocation/experiment.py
========================
Runs the main allocation experiments (Layer 2 evaluation) across all 16
dataset configurations.

Experiment surface per the implementation plan:
  - 16 configs × 4 conditions × 30 runs = 1,920 main runs
  - 16 configs × 2 models × 3 delta values × 30 runs = 2,880 delta-sensitivity runs

Four comparison conditions per config:
  1. UNIFORM  — baseline: equal L_RS for all sequences
  2. ORACLE   — theoretical ceiling: uses true failure_freq as risk scores
  3. XGB_CAL  — calibrated XGBoost predictions
  4. XGB_RAW  — uncalibrated XGBoost outputs (shows why calibration matters)

For each run, the experiment:
  1. Samples a test set from the channel simulator
  2. Applies the allocation (pre-computed from the held-out model predictions)
  3. Re-runs the channel with per-sequence L_RS
  4. Counts failures under each allocation
  5. Computes FRR = fraction of sequences that failed

Usage (CLI):
    python allocation/experiment.py --config configs/experiment_config.yaml \
        --key sub05_k30_simple --delta 2 --out results/allocation/
"""

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))
sys.path.insert(0, os.path.dirname(__file__))


def run_allocation_experiment(
    sequences:             List[str],
    true_failure_freq:     np.ndarray,
    calibrated_risk:       np.ndarray,
    raw_risk:              np.ndarray,
    channel,
    coverage:              int,
    l_rs_default:          int,
    delta:                 int,
    n_runs:                int = 30,
    base_seed:             int = 0,
    l_rs_min:              int = 4,
    l_rs_max:              int = 16,
    verbose:               bool = False,
) -> Dict[str, np.ndarray]:
    """Run all four allocation conditions across M independent simulator runs.

    Returns
    -------
    dict with keys:
      'frr_uniform'  : array (M,) FRR per run for uniform allocation
      'frr_oracle'   : array (M,) FRR per run for oracle allocation
      'frr_xgb_cal'  : array (M,) FRR per run for calibrated XGBoost
      'frr_xgb_raw'  : array (M,) FRR per run for uncalibrated XGBoost
    """
    from mechanism import AllocationMechanism, uniform_allocation, oracle_allocation

    # Pre-compute allocation vectors (allocation is based on held-out model predictions,
    # so it doesn't change across runs—only the channel is re-sampled)
    N = len(sequences)
    l_rs_uniform = uniform_allocation(N, l_rs_default)

    alloc = AllocationMechanism(l_rs_default, delta, l_rs_min, l_rs_max)
    l_rs_oracle  = oracle_allocation(true_failure_freq, l_rs_default, delta, l_rs_min, l_rs_max)
    l_rs_xgb_cal = alloc.allocate(calibrated_risk)
    l_rs_xgb_raw = alloc.allocate(raw_risk)

    frr_uniform  = np.zeros(n_runs)
    frr_oracle   = np.zeros(n_runs)
    frr_xgb_cal  = np.zeros(n_runs)
    frr_xgb_raw  = np.zeros(n_runs)

    for run_idx in range(n_runs):
        run_channel = channel.clone(seed=base_seed + 1000 + run_idx)

        for condition, l_rs_vec, frr_arr in [
            ('uniform',  l_rs_uniform,  frr_uniform),
            ('oracle',   l_rs_oracle,   frr_oracle),
            ('xgb_cal',  l_rs_xgb_cal,  frr_xgb_cal),
            ('xgb_raw',  l_rs_xgb_raw,  frr_xgb_raw),
        ]:
            failures = _count_failures_with_allocation(
                sequences, run_channel, coverage, l_rs_vec
            )
            frr_arr[run_idx] = failures / N

        if verbose and (run_idx + 1) % 5 == 0:
            print(f"  [allocation] Run {run_idx+1}/{n_runs} | "
                  f"FRR uniform={frr_uniform[:run_idx+1].mean():.4f} | "
                  f"oracle={frr_oracle[:run_idx+1].mean():.4f} | "
                  f"xgb_cal={frr_xgb_cal[:run_idx+1].mean():.4f}")

    return {
        'frr_uniform' : frr_uniform,
        'frr_oracle'  : frr_oracle,
        'frr_xgb_cal' : frr_xgb_cal,
        'frr_xgb_raw' : frr_xgb_raw,
    }


def _count_failures_with_allocation(
    sequences: List[str],
    channel,
    coverage: int,
    l_rs_vec: np.ndarray,
) -> int:
    """Count sequence failures given per-sequence L_RS allocation."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
    from consensus_voter import majority_vote_consensus, count_errors, is_decoding_failure

    failures = 0
    for seq, l_rs in zip(sequences, l_rs_vec):
        reads     = channel.simulate(seq, coverage)
        consensus = majority_vote_consensus(reads, len(seq))
        stats     = count_errors(seq, consensus)
        if is_decoding_failure(stats['byte_errors'], int(l_rs)):
            failures += 1
    return failures


def run_delta_sensitivity(
    sequences:         List[str],
    true_failure_freq: np.ndarray,
    calibrated_risk:   np.ndarray,
    channel,
    coverage:          int,
    l_rs_default:      int,
    delta_values:      List[int],
    n_runs:            int = 30,
    base_seed:         int = 0,
    l_rs_min:          int = 4,
    l_rs_max:          int = 16,
) -> Dict[int, Dict[str, np.ndarray]]:
    """Run FRR vs. delta analysis for calibrated XGBoost and Oracle conditions.

    Returns
    -------
    dict keyed by delta value, each with 'frr_xgb_cal', 'frr_oracle', 'frr_uniform' arrays.
    """
    from mechanism import AllocationMechanism, uniform_allocation, oracle_allocation

    N = len(sequences)
    results = {}

    for delta in delta_values:
        print(f"  [delta_sensitivity] delta={delta}")
        alloc_mech = AllocationMechanism(l_rs_default, delta, l_rs_min, l_rs_max)
        l_rs_xgb  = alloc_mech.allocate(calibrated_risk)
        l_rs_ora  = oracle_allocation(true_failure_freq, l_rs_default, delta, l_rs_min, l_rs_max)
        l_rs_uni  = uniform_allocation(N, l_rs_default)

        frr_xgb = np.zeros(n_runs)
        frr_ora = np.zeros(n_runs)
        frr_uni = np.zeros(n_runs)

        for run_idx in range(n_runs):
            run_channel = channel.clone(seed=base_seed + 2000 + run_idx)
            frr_xgb[run_idx] = _count_failures_with_allocation(
                sequences, run_channel, coverage, l_rs_xgb) / N
            frr_ora[run_idx] = _count_failures_with_allocation(
                sequences, run_channel, coverage, l_rs_ora) / N
            frr_uni[run_idx] = _count_failures_with_allocation(
                sequences, run_channel, coverage, l_rs_uni) / N

        results[delta] = {
            'frr_xgb_cal': frr_xgb,
            'frr_oracle' : frr_ora,
            'frr_uniform': frr_uni,
        }

    return results


def save_results(results: dict, out_path: str):
    """Save allocation experiment results as a compressed numpy archive."""
    np.savez_compressed(out_path, **{k: np.array(v) for k, v in results.items()
                                      if isinstance(v, (np.ndarray, list))})
    print(f"  [allocation] Saved to {out_path}")


# ── CLI entry point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Run allocation experiments for one dataset configuration.'
    )
    parser.add_argument('--config',    default='configs/experiment_config.yaml')
    parser.add_argument('--key',       required=True)
    parser.add_argument('--delta',     type=int, default=2)
    parser.add_argument('--n-runs',    type=int, default=None)
    parser.add_argument('--models-dir', default='models/saved/')
    parser.add_argument('--out',       default='results/allocation/')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    sys.path.insert(0, 'src')
    sys.path.insert(0, 'models')
    from dataset_assembler import load_dataset
    from channel_model import build_channel_from_config
    from train import load_models

    n_runs = args.n_runs or cfg['allocation']['n_monte_carlo_runs']

    # Parse config key to get sub_rate, coverage
    # Format: sub{XX}_k{K}_{encoding}
    parts     = args.key.split('_')
    sub_rate  = int(parts[0].replace('sub', '')) / 100
    coverage  = int(parts[1].replace('k', ''))
    encoding  = parts[2]

    print(f"[allocation] Loading {args.key} | delta={args.delta} | runs={n_runs}")
    X_tr, X_val, X_te, y_tr, y_val, y_te, feat_names = load_dataset(args.key, cfg)
    models = load_models(args.models_dir, args.key)

    # Get risk scores from calibrated and raw XGBoost
    xgb_cal   = models.get('xgboost')
    calibrated_risk = xgb_cal.predict_proba(X_te) if xgb_cal else np.zeros(len(y_te))
    if xgb_cal:
        _p = xgb_cal.base_model.predict_proba(X_te)
        raw_risk = _p[:, 1] if _p.shape[1] >= 2 else np.zeros(len(y_te))
    else:
        raw_risk = np.zeros(len(y_te))

    # Rebuild channel
    channel = build_channel_from_config(cfg, sub_rate, seed=cfg['random_seed'])

    # Load test sequences
    import pandas as pd
    data_path = os.path.join(cfg['paths']['datasets_dir'], f'{args.key}.parquet')
    df        = pd.read_parquet(data_path)
    splits    = pd.read_parquet(os.path.join(cfg['paths']['splits_dir'], f'{args.key}_splits.parquet'))
    test_idx  = splits[splits['split'] == 'test']['index'].values
    sequences = df['dna_sequence'].values[test_idx].tolist()

    os.makedirs(args.out, exist_ok=True)

    frr_results = run_allocation_experiment(
        sequences, y_te, calibrated_risk, raw_risk,
        channel, coverage, cfg['sequence']['l_rs_default'],
        args.delta, n_runs, verbose=True,
    )

    out_path = os.path.join(args.out, f'{args.key}_delta{args.delta}')
    save_results(frr_results, out_path)

    # Print summary
    sys.path.insert(0, 'models')
    from evaluate import allocation_statistics
    stats = allocation_statistics(
        frr_results['frr_uniform'],
        frr_results['frr_xgb_cal'],
        frr_results['frr_oracle'],
    )
    print(f"\nResults for {args.key} | delta={args.delta}")
    for k, v in stats.items():
        print(f"  {k:<25}: {v:.4f}" if isinstance(v, float) else f"  {k:<25}: {v}")


if __name__ == '__main__':
    main()
