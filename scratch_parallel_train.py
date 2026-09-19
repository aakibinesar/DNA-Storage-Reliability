"""
Parallel launcher for models/train.py across all 28 configs.

run_pipeline.py's stage_train runs these sequentially (one config at a
time) -- slow given XGBoost's 48-combo grid search per config. This runs
them with a worker pool instead.

Usage:
    python scratch_parallel_train.py [--workers N] [--force]

Safe to re-run: skips a key if all 4 of its expected model files already
exist, unless --force is passed (matches run_pipeline.py's
model_files_exist check).
"""
import argparse
import os
import subprocess
import sys
import time
from multiprocessing import Pool

import yaml


def get_all_keys(cfg):
    sub_rates = cfg['channel']['substitution_rates']
    coverages = cfg['coverage_depths']
    encodings = cfg['sequence']['encoding_schemes']
    return [
        f'sub{int(s * 100):02d}_k{k}_{e}'
        for s in sub_rates for k in coverages for e in encodings
    ]


def run_one(args_tuple):
    """Module-level (not a local function) so it can be pickled by
    Windows' spawn-based multiprocessing -- a local/nested function breaks
    Pool.imap_unordered silently at task-submission time."""
    key, config_path, out_dir = args_tuple
    t0 = time.time()
    os.makedirs('logs', exist_ok=True)
    log_path = f'logs/train_{key}.log'
    with open(log_path, 'w') as logf:
        rc = subprocess.call(
            [sys.executable, '-u', 'models/train.py',
             '--config', config_path, '--key', key, '--out', out_dir],
            stdout=logf, stderr=subprocess.STDOUT,
        )
    return key, rc, round(time.time() - t0, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--config', default='configs/experiment_config.yaml')
    parser.add_argument('--out', default='models/saved/')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    all_keys = get_all_keys(cfg)

    needed_suffixes = ['xgboost', 'random_forest', 'logistic_regression', 'logistic_scaler']

    def already_done(key):
        if args.force:
            return False
        return all(
            os.path.exists(os.path.join(args.out, f'{key}_{s}.pkl'))
            for s in needed_suffixes
        )

    keys = [k for k in all_keys if not already_done(k)]
    print(f"[train] {len(all_keys) - len(keys)} already done, {len(keys)} remaining")

    if not keys:
        print("[train] nothing to do")
        return

    print(f"Training {len(keys)} configs with {args.workers} workers ...")
    t_start = time.time()
    n_done = 0
    tasks = [(k, args.config, args.out) for k in keys]
    with Pool(processes=args.workers) as pool:
        for key, rc, elapsed in pool.imap_unordered(run_one, tasks):
            n_done += 1
            status = 'OK' if rc == 0 else f'FAIL(rc={rc})'
            print(f"  [{n_done}/{len(keys)}] {key}: {status}  [{elapsed}s]")
    print(f"[train] Done in {(time.time() - t_start) / 60:.1f} min")


if __name__ == '__main__':
    main()
