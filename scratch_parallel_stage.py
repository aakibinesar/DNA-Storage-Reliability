"""
Generic parallel launcher for any of the per-key analysis stages that
run_pipeline.py otherwise runs sequentially (slow at 28 configs):
  ablation, threshold_sensitivity, calibration_regimes, shap_stability,
  channel_ablation.

Usage:
    python scratch_parallel_stage.py <stage> [--workers N] [--force]

Where <stage> is one of: ablation, threshold_sensitivity,
calibration_regimes, shap_stability, channel_ablation.

Prints one line per completed config with pass/fail and elapsed time, and a
final summary. Safe to re-run: skips a key if its expected output file
already exists, unless --force is passed.
"""
import argparse
import os
import subprocess
import sys
import time
from multiprocessing import Pool

import yaml

STAGES = {
    'ablation': {
        'script': 'analysis/ablation.py',
        'out': 'results/ablation/',
        'needs_models_dir': False,
        'out_file': lambda key: f'{key}_ablation.csv',
    },
    'threshold_sensitivity': {
        'script': 'analysis/threshold_sensitivity.py',
        'out': 'results/threshold_sensitivity/',
        'needs_models_dir': True,
        'out_file': lambda key: f'{key}_threshold_sensitivity.csv',
    },
    'calibration_regimes': {
        'script': 'analysis/calibration_regimes.py',
        'out': 'results/calibration_regimes/',
        'needs_models_dir': True,
        'out_file': lambda key: f'{key}_calibration_regimes.csv',
    },
    'shap_stability': {
        'script': 'analysis/shap_stability.py',
        'out': 'results/shap_stability/',
        'needs_models_dir': True,
        'out_file': lambda key: f'shap_stability_{key}.csv',
    },
    'channel_ablation': {
        'script': 'analysis/channel_ablation.py',
        'out': 'results/channel_ablation/',
        'needs_models_dir': True,
        'out_file': lambda key: f'{key}_channel_ablation.csv',
    },
}


def get_all_keys(cfg):
    sub_rates = cfg['channel']['substitution_rates']
    coverages = cfg['coverage_depths']
    encodings = cfg['sequence']['encoding_schemes']
    return [
        f'sub{int(s * 100):02d}_k{k}_{e}'
        for s in sub_rates for k in coverages for e in encodings
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=list(STAGES.keys()))
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--config', default='configs/experiment_config.yaml')
    parser.add_argument('--models-dir', default='models/saved/')
    args = parser.parse_args()

    spec = STAGES[args.stage]
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    all_keys = get_all_keys(cfg)

    def already_done(key):
        if args.force:
            return False
        return os.path.exists(os.path.join(spec['out'], spec['out_file'](key)))

    keys = [k for k in all_keys if not already_done(k)]
    print(f"[{args.stage}] {len(all_keys) - len(keys)} already done, {len(keys)} remaining")

    def run_one(key):
        t0 = time.time()
        os.makedirs('logs', exist_ok=True)
        log_path = f'logs/{args.stage}_{key}.log'
        cmd = [sys.executable, '-u', spec['script'],
               '--config', args.config, '--key', key, '--out', spec['out']]
        if spec['needs_models_dir']:
            cmd += ['--models-dir', args.models_dir]
        with open(log_path, 'w') as logf:
            rc = subprocess.call(cmd, stdout=logf, stderr=subprocess.STDOUT)
        return key, rc, round(time.time() - t0, 1)

    if not keys:
        print(f"[{args.stage}] nothing to do")
        return

    t_start = time.time()
    n_done = 0
    with Pool(processes=args.workers) as pool:
        for key, rc, elapsed in pool.imap_unordered(run_one, keys):
            n_done += 1
            status = 'OK' if rc == 0 else f'FAIL(rc={rc})'
            print(f"  [{n_done}/{len(keys)}] {key}: {status}  [{elapsed}s]")
    print(f"[{args.stage}] Done in {(time.time() - t_start) / 60:.1f} min")


if __name__ == '__main__':
    main()
