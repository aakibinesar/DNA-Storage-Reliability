"""
Parallel launcher for analysis/deployable_abstention.py across all 28 configs.

Each job simulates the validation byte errors once for a config (30 runs) and
scores the abstention-augmented budget grid for every delta and model. Same
structure and thread-pinning as scratch_parallel_allocation.py (module-level
worker for Windows spawn-based Pool; one internal thread per worker process).

Usage:
    python scratch_parallel_abstention.py [--workers N] [--force]

Safe to re-run: skips a key if its CSV already exists, unless --force.
"""
import argparse
import os
import subprocess
import sys
import time
from multiprocessing import Pool

import yaml


def get_all_keys(cfg):
    return [
        f'sub{int(s * 100):02d}_k{k}_{e}'
        for s in cfg['channel']['substitution_rates']
        for k in cfg['coverage_depths']
        for e in cfg['sequence']['encoding_schemes']
    ]


def run_one(job):
    key, config_path, out_dir = job
    t0 = time.time()
    os.makedirs('logs', exist_ok=True)
    env = os.environ.copy()
    for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        env[var] = '1'
    with open(f'logs/abstention_{key}.log', 'w') as logf:
        rc = subprocess.call(
            [sys.executable, '-u', 'analysis/deployable_abstention.py',
             '--config', config_path, '--key', key, '--out', out_dir],
            stdout=logf, stderr=subprocess.STDOUT, env=env)
    return key, rc, round(time.time() - t0, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--config', default='configs/experiment_config.yaml')
    parser.add_argument('--out', default='results/deployable_abstention/')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    keys = get_all_keys(cfg)
    todo = [k for k in keys
            if args.force or not os.path.exists(os.path.join(args.out, f'{k}_abstention.csv'))]
    print(f'[abstention] {len(keys) - len(todo)} already done, {len(todo)} remaining', flush=True)
    if not todo:
        return
    t0 = time.time()
    n = 0
    with Pool(processes=args.workers) as pool:
        for key, rc, el in pool.imap_unordered(run_one, [(k, args.config, args.out) for k in todo]):
            n += 1
            print(f"  [{n}/{len(todo)}] {key}: {'OK' if rc == 0 else f'FAIL(rc={rc})'} [{el}s]", flush=True)
    print(f'[abstention] Done in {(time.time() - t0) / 60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
