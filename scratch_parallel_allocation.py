"""
Parallel launcher for allocation/experiment.py across all
28 configs x delta in {1,2,3,4} = 112 jobs.

Each job: paired oracle marginal-effect estimation on the test set,
validation-based tier_fraction grid search (once for the risk model,
again for the benefit-aware model when one exists for that key/delta --
i.e. every delta except 1, since only delta=2,3,4 benefit models were
trained), then a 30-run Monte Carlo allocation comparison. Subprocess-per-
job, same structure as scratch_parallel_train.py / scratch_parallel_benefit.py
(module-level worker function -- Windows' spawn-based Pool can't pickle a
function nested inside main()) and the same thread-pinning fix validated
on the benefit-model stage (each worker process capped to 1 internal
thread, since numpy/XGBoost default to using every logical core and N
worker processes each doing that causes severe oversubscription on a
4-core machine).

Usage:
    python scratch_parallel_allocation.py [--workers N] [--deltas 1,2,3,4] [--force]

Safe to re-run: skips a (key, delta) pair if its .npz result already
exists, unless --force is passed.
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
    """Module-level so Windows' spawn-based Pool can pickle it. Pins each
    subprocess to a single internal thread (see scratch_parallel_benefit.py
    for the measured before/after on this machine)."""
    key, delta, config_path, models_dir, benefit_models_dir, out_dir = job_tuple
    t0 = time.time()
    os.makedirs('logs', exist_ok=True)
    log_path = f'logs/alloc_{key}_delta{delta}.log'
    env = os.environ.copy()
    for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        env[var] = '1'
    with open(log_path, 'w') as logf:
        rc = subprocess.call(
            [sys.executable, '-u', 'allocation/experiment.py',
             '--config', config_path, '--key', key, '--delta', str(delta),
             '--models-dir', models_dir, '--benefit-models-dir', benefit_models_dir,
             '--out', out_dir],
            stdout=logf, stderr=subprocess.STDOUT, env=env,
        )
    return key, delta, rc, round(time.time() - t0, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--deltas', default='1,2,3,4')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--config', default='configs/experiment_config.yaml')
    parser.add_argument('--models-dir', default='models/saved/')
    parser.add_argument('--benefit-models-dir', default='models/saved_benefit/')
    parser.add_argument('--out', default='results/allocation/')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    all_keys = get_all_keys(cfg)
    deltas = [int(d) for d in args.deltas.split(',')]

    def already_done(key, delta):
        if args.force:
            return False
        return os.path.exists(os.path.join(args.out, f'{key}_delta{delta}.npz'))

    all_jobs = [(k, d) for d in deltas for k in all_keys]
    jobs = [(k, d) for k, d in all_jobs if not already_done(k, d)]
    print(f"[allocation] {len(all_jobs) - len(jobs)} already done, {len(jobs)} remaining "
          f"({len(all_keys)} keys x {len(deltas)} deltas)")

    if not jobs:
        print("[allocation] nothing to do")
        return

    print(f"Running {len(jobs)} (key, delta) jobs with {args.workers} workers ...")
    t_start = time.time()
    n_done = 0
    tasks = [(k, d, args.config, args.models_dir, args.benefit_models_dir, args.out)
             for k, d in jobs]
    with Pool(processes=args.workers) as pool:
        for key, delta, rc, elapsed in pool.imap_unordered(run_one, tasks):
            n_done += 1
            status = 'OK' if rc == 0 else f'FAIL(rc={rc})'
            print(f"  [{n_done}/{len(jobs)}] {key} delta={delta}: {status}  [{elapsed}s]")
    print(f"[allocation] Done in {(time.time() - t_start) / 60:.1f} min")


if __name__ == '__main__':
    main()
