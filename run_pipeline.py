#!/usr/bin/env python3
"""
run_pipeline.py
================
Orchestrates the full DNA-storage adaptive-redundancy pipeline end to end,
designed to be kicked off as a background process and checked in on
periodically (e.g. from Claude Code, tmux, or the mobile app) rather than
watched live.

Pipeline order (matches the dependency chain in the implementation plan):
    1. datasets                -- build all 28 dataset configs (sequence gen,
                                  channel sim, labels, features, splits)
    2. train                   -- train+calibrate 3 models for each of 28 keys
    3. gate_check              -- Week 4 go/no-go gate (informational by default)
    4. threshold_sensitivity   -- R8: sweep decision thresholds 0.10–0.90,
                                  report near-threshold label uncertainty
    5. calibration_regimes     -- R9: ECE/Brier stratified by under_failure /
                                  informative / saturated regimes, with bootstrap
                                  95% CIs and calibration deltas
    6. regime_evaluation       -- R11: full Layer 1 suite (AUROC, F1, ECE, Brier)
                                  per regime across all 28 configs; warns when
                                  the informative regime is < 10% of test set
    7. allocation              -- run allocation experiments per key x delta
                                  NOTE: allocation results produced before the
                                  R2+R3 fixes are stale.  Move them out of
                                  results/allocation/ or use --force to regenerate.
    7. ablation + dist_shift   -- run concurrently (both only depend on `train`)
    8. channel_ablation        -- R7: model quality under ablated channel variants
    9. figures                 -- generate all paper figures

Checkpointing
-------------
Every stage/item writes its state into pipeline_status.json. Re-running this
script skips anything already marked DONE (or whose output files already
exist on disk), so it is always safe to Ctrl-C and restart, or restart after
a crash. Use --force to ignore checkpoints for a given stage.

Logging
-------
Every subprocess's stdout/stderr streams live into logs/<stage>.log so you
can `tail -f logs/train_sub05_k5_simple.log` while it runs, or hand the file
to Claude to summarize.

Usage
-----
    # Kick off the whole thing in the background:
    nohup python run_pipeline.py > logs/_top_level.log 2>&1 &

    # Check in any time (does not require the background process to be Claude):
    python run_pipeline.py --status

    # Resume from a specific stage (e.g. after fixing a bug mid-run):
    python run_pipeline.py --from-stage allocation

    # Run just one stage:
    python run_pipeline.py --only train

    # See what would run without running it:
    python run_pipeline.py --dry-run
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

import yaml

ROOT       = os.path.dirname(os.path.abspath(__file__))
CONFIG_DEF = os.path.join(ROOT, 'configs', 'experiment_config.yaml')
STATUS_PATH = os.path.join(ROOT, 'pipeline_status.json')
LOG_DIR    = os.path.join(ROOT, 'logs')

STAGE_ORDER = ['datasets', 'train', 'gate_check', 'threshold_sensitivity',
               'calibration_regimes', 'regime_evaluation', 'allocation',
               'ablation', 'distribution_shift', 'transfer_radius',
               'shap_stability', 'encoding_confound',
               'channel_ablation', 'figures']

_status_lock = threading.Lock()


# ── status persistence ───────────────────────────────────────────────────────

def _now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def load_status():
    if os.path.exists(STATUS_PATH):
        with open(STATUS_PATH) as f:
            return json.load(f)
    return {}


def save_status(status):
    with _status_lock:
        tmp = STATUS_PATH + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(status, f, indent=2, sort_keys=True)
        os.replace(tmp, STATUS_PATH)


def set_item_status(name, **fields):
    status = load_status()
    entry = status.get(name, {})
    entry.update(fields)
    status[name] = entry
    save_status(status)
    return entry


# ── subprocess execution with live logging + heartbeat ─────────────────────

def run_cmd(name, cmd, log_path, dry_run=False):
    """Run `cmd` (list of args), streaming output to log_path, updating
    pipeline_status.json as it goes. Returns the process return code."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    if dry_run:
        print(f"[DRY-RUN] {name}: {' '.join(cmd)}  (log -> {log_path})")
        return 0

    set_item_status(name, state='RUNNING', started_at=_now(),
                     log=os.path.relpath(log_path, ROOT), cmd=' '.join(cmd))
    print(f"[{_now()}] START {name}: {' '.join(cmd)}")

    with open(log_path, 'a') as logf:
        logf.write(f"\n=== run started {_now()} ===\n")
        logf.write(f"cmd: {' '.join(cmd)}\n\n")
        logf.flush()

        proc = subprocess.Popen(
            cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        last_line = ''
        for line in proc.stdout:
            logf.write(line)
            logf.flush()
            last_line = line.rstrip()
            set_item_status(name, last_log_line=last_line, heartbeat=_now())
        proc.wait()

    rc = proc.returncode
    state = 'DONE' if rc == 0 else 'FAILED'
    set_item_status(name, state=state, ended_at=_now(), returncode=rc)
    print(f"[{_now()}] {state} {name} (rc={rc})")

    if rc != 0:
        print(f"  --- last 20 lines of {log_path} ---")
        try:
            with open(log_path) as f:
                for l in f.readlines()[-20:]:
                    print('  ' + l.rstrip())
        except OSError:
            pass

    return rc


# ── checkpoint predicates (skip work that's already done on disk) ──────────

def all_dataset_files_exist(cfg, keys):
    ddir = os.path.join(ROOT, cfg['paths']['datasets_dir'])
    sdir = os.path.join(ROOT, cfg['paths']['splits_dir'])
    return all(
        os.path.exists(os.path.join(ddir, f'{k}.parquet')) and
        os.path.exists(os.path.join(sdir, f'{k}_splits.parquet'))
        for k in keys
    )


def model_files_exist(models_dir, key):
    needed = [f'{key}_xgboost.pkl', f'{key}_random_forest.pkl',
              f'{key}_logistic_regression.pkl', f'{key}_logistic_scaler.pkl']
    return all(os.path.exists(os.path.join(ROOT, models_dir, n)) for n in needed)


def allocation_file_exists(out_dir, key, delta):
    return os.path.exists(os.path.join(ROOT, out_dir, f'{key}_delta{delta}.npz'))


def ablation_file_exists(out_dir, key):
    return os.path.exists(os.path.join(ROOT, out_dir, f'{key}_ablation.csv'))


def dist_shift_file_exists(out_dir, coverage, encoding):
    return os.path.exists(os.path.join(ROOT, out_dir, f'dist_shift_k{coverage}_{encoding}.csv'))


def transfer_radius_file_exists(out_dir):
    return os.path.exists(os.path.join(ROOT, out_dir, 'transfer_radius_summary.csv'))


def shap_stability_file_exists(out_dir, key):
    return os.path.exists(os.path.join(ROOT, out_dir, f'shap_stability_{key}.csv'))


def encoding_confound_file_exists(out_dir):
    return os.path.exists(os.path.join(ROOT, out_dir, 'encoding_confound_summary.csv'))


def figures_exist(out_dir):
    p = os.path.join(ROOT, out_dir)
    return os.path.isdir(p) and len(os.listdir(p)) > 0


# ── config helpers ──────────────────────────────────────────────────────────

def get_all_keys(cfg):
    sub_rates = cfg['channel']['substitution_rates']
    coverages = cfg['coverage_depths']
    encodings = cfg['sequence']['encoding_schemes']
    return [
        f'sub{int(s*100):02d}_k{k}_{e}'
        for s in sub_rates for k in coverages for e in encodings
    ]


# ── stage implementations ───────────────────────────────────────────────────

def stage_datasets(cfg, args):
    keys = get_all_keys(cfg)
    if not args.force and all_dataset_files_exist(cfg, keys):
        print("[datasets] All 16 dataset/split files already exist — skipping.")
        set_item_status('datasets', state='SKIPPED')
        return 0
    cmd = [sys.executable, '-u', 'src/dataset_assembler.py', '--config', args.config]
    return run_cmd('datasets', cmd, os.path.join(LOG_DIR, 'datasets.log'), args.dry_run)


def stage_train(cfg, args):
    keys = get_all_keys(cfg)
    rc_all = 0
    for key in keys:
        name = f'train:{key}'
        if not args.force and model_files_exist(cfg['paths']['models_dir'], key):
            print(f"[train] {key}: models already exist — skipping.")
            set_item_status(name, state='SKIPPED')
            continue
        cmd = [sys.executable, '-u', 'models/train.py', '--config', args.config,
               '--key', key, '--out', cfg['paths']['models_dir']]
        rc = run_cmd(name, cmd, os.path.join(LOG_DIR, f'train_{key}.log'), args.dry_run)
        if rc != 0:
            rc_all = rc
            if not args.keep_going:
                return rc_all
    return rc_all


def stage_gate_check(cfg, args):
    cmd = [sys.executable, '-u', 'gate_check.py', '--config', args.config,
           '--models-dir', cfg['paths']['models_dir']]
    rc = run_cmd('gate_check', cmd, os.path.join(LOG_DIR, 'gate_check.log'), args.dry_run)
    if rc != 0 and args.enforce_gate and not args.keep_going:
        print("[gate_check] FAILED and --enforce-gate set — halting before allocation/ablation.")
        return rc
    return 0  # informational by default; doesn't block the rest of the run


def stage_allocation(cfg, args):
    keys   = get_all_keys(cfg)
    deltas = cfg['allocation']['delta_values']
    out_dir = os.path.join(cfg['paths']['results_dir'], 'allocation')
    rc_all = 0
    for key in keys:
        for delta in deltas:
            name = f'allocation:{key}:delta{delta}'
            if not args.force and allocation_file_exists(out_dir, key, delta):
                print(f"[allocation] {key} delta={delta}: result already exists — skipping.")
                set_item_status(name, state='SKIPPED')
                continue
            cmd = [sys.executable, '-u', 'allocation/experiment.py', '--config', args.config,
                   '--key', key, '--delta', str(delta),
                   '--models-dir', cfg['paths']['models_dir'], '--out', out_dir]
            rc = run_cmd(name, cmd,
                         os.path.join(LOG_DIR, f'allocation_{key}_delta{delta}.log'),
                         args.dry_run)
            if rc != 0:
                rc_all = rc
                if not args.keep_going:
                    return rc_all
    return rc_all


def stage_ablation(cfg, args):
    keys = get_all_keys(cfg)
    out_dir = os.path.join(cfg['paths']['results_dir'], 'ablation')
    rc_all = 0
    for key in keys:
        name = f'ablation:{key}'
        if not args.force and ablation_file_exists(out_dir, key):
            print(f"[ablation] {key}: result already exists — skipping.")
            set_item_status(name, state='SKIPPED')
            continue
        cmd = [sys.executable, '-u', 'analysis/ablation.py', '--config', args.config,
               '--key', key, '--out', out_dir]
        rc = run_cmd(name, cmd, os.path.join(LOG_DIR, f'ablation_{key}.log'), args.dry_run)
        if rc != 0:
            rc_all = rc
            if not args.keep_going:
                return rc_all
    return rc_all


def stage_distribution_shift(cfg, args):
    coverages = cfg['coverage_depths']
    encodings = cfg['sequence']['encoding_schemes']
    out_dir = os.path.join(cfg['paths']['results_dir'], 'distribution_shift')
    rc_all = 0
    for cov in coverages:
        for enc in encodings:
            name = f'distribution_shift:k{cov}_{enc}'
            if not args.force and dist_shift_file_exists(out_dir, cov, enc):
                print(f"[distribution_shift] k={cov} {enc}: result already exists — skipping.")
                set_item_status(name, state='SKIPPED')
                continue
            cmd = [sys.executable, '-u', 'analysis/distribution_shift.py', '--config', args.config,
                   '--coverage', str(cov), '--encoding', enc, '--out', out_dir]
            rc = run_cmd(name, cmd,
                         os.path.join(LOG_DIR, f'dist_shift_k{cov}_{enc}.log'),
                         args.dry_run)
            if rc != 0:
                rc_all = rc
                if not args.keep_going:
                    return rc_all
    return rc_all


def stage_regime_evaluation(cfg, args):
    """R11: full Layer 1 metric suite stratified by failure-freq regime."""
    out_dir  = os.path.join(cfg['paths']['results_dir'], 'regime_evaluation')
    out_path = os.path.join(out_dir, 'regime_evaluation_all.csv')
    if not args.force and os.path.exists(os.path.join(ROOT, out_path)):
        print("[regime_evaluation] Combined CSV already exists — skipping.")
        set_item_status('regime_evaluation', state='SKIPPED')
        return 0
    cmd = [sys.executable, '-u', 'analysis/regime_evaluation.py',
           '--config', args.config,
           '--models-dir', cfg['paths']['models_dir'],
           '--out', out_dir]
    return run_cmd('regime_evaluation', cmd,
                   os.path.join(LOG_DIR, 'regime_evaluation.log'), args.dry_run)


def stage_calibration_regimes(cfg, args):
    """R9: regime-stratified calibration with bootstrap CIs."""
    keys    = get_all_keys(cfg)
    out_dir = os.path.join(cfg['paths']['results_dir'], 'calibration_regimes')
    rc_all  = 0
    for key in keys:
        name     = f'calibration_regimes:{key}'
        out_path = os.path.join(out_dir, f'{key}_calibration_regimes.csv')
        if not args.force and os.path.exists(os.path.join(ROOT, out_path)):
            print(f"[calibration_regimes] {key}: result already exists — skipping.")
            set_item_status(name, state='SKIPPED')
            continue
        cmd = [sys.executable, '-u', 'analysis/calibration_regimes.py',
               '--config', args.config,
               '--key', key,
               '--models-dir', cfg['paths']['models_dir'],
               '--out', out_dir]
        rc = run_cmd(name, cmd,
                     os.path.join(LOG_DIR, f'calibration_regimes_{key}.log'),
                     args.dry_run)
        if rc != 0:
            rc_all = rc
            if not args.keep_going:
                return rc_all
    return rc_all


def stage_threshold_sensitivity(cfg, args):
    """R8: sweep classification thresholds and quantify near-threshold label noise."""
    keys    = get_all_keys(cfg)
    out_dir = os.path.join(cfg['paths']['results_dir'], 'threshold_sensitivity')
    rc_all  = 0
    for key in keys:
        name     = f'threshold_sensitivity:{key}'
        out_path = os.path.join(out_dir, f'{key}_threshold_sensitivity.csv')
        if not args.force and os.path.exists(os.path.join(ROOT, out_path)):
            print(f"[threshold_sensitivity] {key}: result already exists — skipping.")
            set_item_status(name, state='SKIPPED')
            continue
        cmd = [sys.executable, '-u', 'analysis/threshold_sensitivity.py',
               '--config', args.config,
               '--key', key,
               '--models-dir', cfg['paths']['models_dir'],
               '--out', out_dir]
        rc = run_cmd(name, cmd,
                     os.path.join(LOG_DIR, f'threshold_sensitivity_{key}.log'),
                     args.dry_run)
        if rc != 0:
            rc_all = rc
            if not args.keep_going:
                return rc_all
    return rc_all


def stage_transfer_radius(cfg, args):
    """R12: systematic all-pairs transfer radius using saved models + regime-aware AUROC."""
    out_dir  = os.path.join(cfg['paths']['results_dir'], 'transfer_radius')
    if not args.force and transfer_radius_file_exists(out_dir):
        print("[transfer_radius] Summary CSV already exists — skipping.")
        set_item_status('transfer_radius', state='SKIPPED')
        return 0
    cmd = [sys.executable, '-u', 'analysis/transfer_radius.py',
           '--config', args.config,
           '--models-dir', cfg['paths']['models_dir'],
           '--out', out_dir]
    return run_cmd('transfer_radius', cmd,
                   os.path.join(LOG_DIR, 'transfer_radius.log'), args.dry_run)


def stage_shap_stability(cfg, args):
    """R13: bootstrap SHAP stability analysis — per-key, informative regime only."""
    keys    = get_all_keys(cfg)
    out_dir = os.path.join(cfg['paths']['results_dir'], 'shap_stability')
    rc_all  = 0
    for key in keys:
        name     = f'shap_stability:{key}'
        if not args.force and shap_stability_file_exists(out_dir, key):
            print(f"[shap_stability] {key}: result already exists — skipping.")
            set_item_status(name, state='SKIPPED')
            continue
        cmd = [sys.executable, '-u', 'analysis/shap_stability.py',
               '--config', args.config,
               '--key', key,
               '--models-dir', cfg['paths']['models_dir'],
               '--out', out_dir]
        rc = run_cmd(name, cmd,
                     os.path.join(LOG_DIR, f'shap_stability_{key}.log'),
                     args.dry_run)
        if rc != 0:
            rc_all = rc
            if not args.keep_going:
                return rc_all
    return rc_all


def stage_encoding_confound(cfg, args):
    """R14: deconfound simple vs constrained encoding comparison for all strata."""
    out_dir = os.path.join(cfg['paths']['results_dir'], 'encoding_confound')
    if not args.force and encoding_confound_file_exists(out_dir):
        print("[encoding_confound] Summary CSV already exists — skipping.")
        set_item_status('encoding_confound', state='SKIPPED')
        return 0
    cmd = [sys.executable, '-u', 'analysis/encoding_confound.py',
           '--config', args.config,
           '--out', out_dir]
    return run_cmd('encoding_confound', cmd,
                   os.path.join(LOG_DIR, 'encoding_confound.log'), args.dry_run)


def stage_channel_ablation(cfg, args):
    """R7: test model under ablated channel variants (no GC skew / no HP stutter)."""
    keys    = get_all_keys(cfg)
    out_dir = os.path.join(cfg['paths']['results_dir'], 'channel_ablation')
    rc_all  = 0
    for key in keys:
        name     = f'channel_ablation:{key}'
        out_path = os.path.join(out_dir, f'{key}_channel_ablation.csv')
        if not args.force and os.path.exists(os.path.join(ROOT, out_path)):
            print(f"[channel_ablation] {key}: result already exists — skipping.")
            set_item_status(name, state='SKIPPED')
            continue
        cmd = [sys.executable, '-u', 'analysis/channel_ablation.py',
               '--config', args.config,
               '--key', key,
               '--models-dir', cfg['paths']['models_dir'],
               '--out', out_dir]
        rc = run_cmd(name, cmd,
                     os.path.join(LOG_DIR, f'channel_ablation_{key}.log'),
                     args.dry_run)
        if rc != 0:
            rc_all = rc
            if not args.keep_going:
                return rc_all
    return rc_all


def stage_figures(cfg, args):
    out_dir = cfg['paths']['figures_dir']
    if not args.force and figures_exist(out_dir):
        print("[figures] Figures directory already populated — skipping.")
        set_item_status('figures', state='SKIPPED')
        return 0
    cmd = [sys.executable, '-u', 'analysis/figures.py', '--config', args.config, '--out', out_dir]
    return run_cmd('figures', cmd, os.path.join(LOG_DIR, 'figures.log'), args.dry_run)


def stage_ablation_and_shift_parallel(cfg, args):
    """Ablation and distribution_shift only depend on `train`, so run them
    in separate threads to use the time-overlap the implementation plan
    explicitly calls out."""
    results = {}

    def _run(name, fn):
        results[name] = fn(cfg, args)

    t1 = threading.Thread(target=_run, args=('ablation', stage_ablation))
    t2 = threading.Thread(target=_run, args=('distribution_shift', stage_distribution_shift))
    t1.start(); t2.start()
    t1.join(); t2.join()

    rc = max(results.get('ablation', 0), results.get('distribution_shift', 0))
    return rc


STAGE_FUNCS = {
    'datasets'              : stage_datasets,
    'train'                 : stage_train,
    'gate_check'            : stage_gate_check,
    'threshold_sensitivity' : stage_threshold_sensitivity,
    'calibration_regimes'   : stage_calibration_regimes,
    'regime_evaluation'     : stage_regime_evaluation,
    'allocation'            : stage_allocation,
    'ablation'              : stage_ablation,
    'distribution_shift'    : stage_distribution_shift,
    'transfer_radius'       : stage_transfer_radius,
    'shap_stability'        : stage_shap_stability,
    'encoding_confound'     : stage_encoding_confound,
    'channel_ablation'      : stage_channel_ablation,
    'figures'               : stage_figures,
}


# ── status report ────────────────────────────────────────────────────────────

def print_status_report():
    status = load_status()
    if not status:
        print("No pipeline_status.json yet — nothing has been run.")
        return

    def sort_key(item):
        name = item[0]
        stage = name.split(':')[0]
        try:
            return (STAGE_ORDER.index(stage), name)
        except ValueError:
            return (len(STAGE_ORDER), name)

    print(f"{'ITEM':<45} {'STATE':<10} {'STARTED':<20} {'LAST UPDATE':<20} RC")
    print('-' * 110)
    for name, entry in sorted(status.items(), key=sort_key):
        state = entry.get('state', '?')
        started = entry.get('started_at', '')
        heartbeat = entry.get('heartbeat', entry.get('ended_at', ''))
        rc = entry.get('returncode', '')
        print(f"{name:<45} {state:<10} {started:<20} {heartbeat:<20} {rc}")
        if state == 'RUNNING' and entry.get('last_log_line'):
            print(f"   ↳ {entry['last_log_line'][:100]}")
        if state == 'FAILED':
            print(f"   ↳ see {entry.get('log')}")

    states = [e.get('state') for e in status.values()]
    print('-' * 110)
    print(f"Summary: {states.count('DONE')} done, {states.count('SKIPPED')} skipped, "
          f"{states.count('RUNNING')} running, {states.count('FAILED')} failed, "
          f"{len(states)} total tracked items.")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Run the full DNA-storage pipeline.')
    parser.add_argument('--config', default=os.path.relpath(CONFIG_DEF, ROOT))
    parser.add_argument('--from-stage', choices=STAGE_ORDER, default=None,
                         help='Start at this stage, run everything after it too.')
    parser.add_argument('--only', choices=list(STAGE_FUNCS.keys()), default=None,
                         help='Run only this one stage.')
    parser.add_argument('--force', action='store_true',
                         help='Ignore on-disk checkpoints and re-run regardless.')
    parser.add_argument('--keep-going', action='store_true',
                         help='Continue to the next item/stage even if one fails.')
    parser.add_argument('--enforce-gate', action='store_true',
                         help='Halt the pipeline if the Week 4 go/no-go gate fails.')
    parser.add_argument('--dry-run', action='store_true',
                         help='Print planned commands without executing anything.')
    parser.add_argument('--status', action='store_true',
                         help='Print the current status table and exit.')
    args = parser.parse_args()

    if args.status:
        print_status_report()
        return

    os.makedirs(LOG_DIR, exist_ok=True)

    with open(os.path.join(ROOT, args.config)) as f:
        cfg = yaml.safe_load(f)

    if args.only:
        stages_to_run = [args.only]
    elif args.from_stage:
        idx = STAGE_ORDER.index(args.from_stage)
        stages_to_run = STAGE_ORDER[idx:]
    else:
        stages_to_run = list(STAGE_ORDER)

    print(f"[{_now()}] Pipeline starting. Stages: {stages_to_run}")

    i = 0
    while i < len(stages_to_run):
        stage = stages_to_run[i]
        if stage in ('ablation', 'distribution_shift') and \
           'ablation' in stages_to_run and 'distribution_shift' in stages_to_run and \
           stages_to_run.index('ablation') + 1 == stages_to_run.index('distribution_shift'):
            # Run the pair concurrently exactly once, then skip the second entry.
            rc = stage_ablation_and_shift_parallel(cfg, args)
            i += 2
        else:
            rc = STAGE_FUNCS[stage](cfg, args)
            i += 1

        if rc != 0 and not args.keep_going:
            print(f"[{_now()}] Stage '{stage}' failed (rc={rc}). Halting. "
                  f"Fix the issue and re-run with --from-stage {stage} to resume.")
            sys.exit(rc)

    print(f"[{_now()}] Pipeline complete.")


if __name__ == '__main__':
    main()
