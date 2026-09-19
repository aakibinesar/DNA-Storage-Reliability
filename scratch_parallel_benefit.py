"""
Parallel launcher for models/train_benefit_model.py across all
28 configs x delta in {2,3,4} = 84 jobs.

Each job re-simulates the channel to estimate true marginal
benefit-minus-harm for every train+val sequence (the expensive part --
much heavier than models/train.py's feature-based grid search), then
fits an XGBoost regressor on those labels. Subprocess-per-job, matching
scratch_parallel_train.py's structure (and its fix: the per-job worker
function must live at module level, not nested inside main(), or Windows'
spawn-based multiprocessing can't pickle it to hand to worker processes).

Usage:
    python scratch_parallel_benefit.py [--workers N] [--deltas 2,3,4] [--force]

Safe to re-run: skips a (key, delta) pair if its model file already exists,
unless --force is passed.
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


def run_one(job_tuple):
    """Module-level so Windows' spawn-based Pool can pickle it (see
    scratch_parallel_train.py's fix for the same class of bug).

    Pins each subprocess to a single internal thread (numpy/BLAS/XGBoost
    all default to using every logical core otherwise) -- with N worker
    processes already dividing up the machine, letting each one ALSO try
    to use all 8 logical threads internally causes severe oversubscription
    (measured: 6 workers gave only ~1.8x real speedup, not 6x, before this
    fix). One thread per process lets N processes cleanly use N threads."""
    key, delta, config_path, out_dir, labels_out = job_tuple
    t0 = time.time()
    os.makedirs('logs', exist_ok=True)
    log_path = f'logs/benefit_{key}_delta{delta}.log'
    env = os.environ.copy()
    for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        env[var] = '1'
    with open(log_path, 'w') as logf:
        rc = subprocess.call(
            [sys.executable, '-u', 'models/train_benefit_model.py',
             '--config', config_path, '--key', key, '--delta', str(delta),
             '--out', out_dir, '--labels-out', labels_out],
            stdout=logf, stderr=subprocess.STDOUT, env=env,
        )
    return key, delta, rc, round(time.time() - t0, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--deltas', default='2,3,4')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--config', default='configs/experiment_config.yaml')
    parser.add_argument('--out', default='models/saved_benefit/')
    parser.add_argument('--labels-out', default='results/benefit_labels/')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    all_keys = get_all_keys(cfg)
    deltas = [int(d) for d in args.deltas.split(',')]

    def already_done(key, delta):
        if args.force:
            return False
        return os.path.exists(os.path.join(args.out, f'{key}_delta{delta}_benefit_model.pkl'))

    all_jobs = [(k, d) for d in deltas for k in all_keys]
    jobs = [(k, d) for k, d in all_jobs if not already_done(k, d)]
    print(f"[benefit] {len(all_jobs) - len(jobs)} already done, {len(jobs)} remaining "
          f"({len(all_keys)} keys x {len(deltas)} deltas)")

    if not jobs:
        print("[benefit] nothing to do")
        return

    print(f"Training {len(jobs)} (key, delta) jobs with {args.workers} workers ...")
    t_start = time.time()
    n_done = 0
    tasks = [(k, d, args.config, args.out, args.labels_out) for k, d in jobs]
    with Pool(processes=args.workers) as pool:
        for key, delta, rc, elapsed in pool.imap_unordered(run_one, tasks):
            n_done += 1
            status = 'OK' if rc == 0 else f'FAIL(rc={rc})'
            print(f"  [{n_done}/{len(jobs)}] {key} delta={delta}: {status}  [{elapsed}s]")
    print(f"[benefit] Done in {(time.time() - t_start) / 60:.1f} min")


if __name__ == '__main__':
    main()
