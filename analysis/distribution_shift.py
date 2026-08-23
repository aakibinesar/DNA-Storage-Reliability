"""
analysis/distribution_shift.py
================================
Distribution shift experiments: train on one substitution regime, evaluate
on another without retraining.

R12 extension
-------------
The original four conditions only tested upward shifts starting from 9% or
12%.  R12 adds:
  • symmetric downward transfers (higher sub_rate → lower)
  • additional upward pairs to cover lower-rate sources (1%, 5%)
  • `delta` (|tgt_rate − src_rate|) and `direction` ('up'/'down') per result
  • `auroc_drop` and `robust_transfer_auroc` alongside the ECE criterion
  • `compute_transfer_radius()` — formal radius = max δ where every tested
    pair at |Δ| ≤ δ satisfies BOTH robustness thresholds

This replaces the single "~3% transfer radius" claim with per-stratum,
per-direction radius estimates that can differ across coverage depths and
encoding schemes.  For a full systematic sweep of ALL sub_rate pairs using
the saved calibrated models see analysis/transfer_radius.py.

Robustness thresholds (same as ROBUST_ECE / ROBUST_AUROC in transfer_radius.py):
  ECE degradation ratio  < 2.0   (< 2× ECE worsening)
  AUROC drop             ≤ 0.05  (≤ 5 pp AUROC decrease)

Usage:
    python analysis/distribution_shift.py \\
        --config configs/experiment_config.yaml \\
        --coverage 5 --encoding simple \\
        --out results/distribution_shift/
"""

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'allocation'))


TRANSFER_CONDITIONS = [
    # ── Upward transfers (train low, test high) ─────────────────────────────
    (0.01, 0.05, 'up_very_low'),
    (0.05, 0.09, 'up_low_to_mid'),
    (0.09, 0.12, 'up_mild'),
    (0.09, 0.15, 'up_moderate'),
    (0.09, 0.20, 'up_severe'),
    (0.12, 0.18, 'up_moderate_high'),
    # ── Downward transfers (train high, test low) ───────────────────────────
    (0.05, 0.01, 'down_very_low'),
    (0.09, 0.05, 'down_low_to_mid'),
    (0.12, 0.09, 'down_mild'),
    (0.15, 0.09, 'down_moderate'),
    (0.18, 0.12, 'down_moderate_high'),
    (0.20, 0.09, 'down_severe'),
]

# Robustness thresholds used by compute_transfer_radius()
_ROBUST_ECE_RATIO   = 2.0   # ECE must not more than double
_ROBUST_AUROC_DROP  = 0.05  # AUROC may not fall more than 5 pp


def run_distribution_shift(
    datasets:     Dict[str, tuple],   # key → (X_tr, X_val, X_te, y_tr, y_val, y_te, feat_names)
    cfg:          dict,
    coverage:     int,
    encoding:     str,
    seed:         int = 42,
    verbose:      bool = True,
) -> Dict[str, dict]:
    """Run all transfer conditions.

    Parameters
    ----------
    datasets : pre-loaded datasets keyed by config key

    Returns
    -------
    results dict keyed by condition label (e.g. 'mild_1pct->5pct')
    """
    from evaluate import full_evaluation, allocation_statistics
    from mechanism import AllocationMechanism

    results = {}

    for src_rate, tgt_rate, severity in TRANSFER_CONDITIONS:
        src_key = _config_key(src_rate, coverage, encoding)
        tgt_key = _config_key(tgt_rate, coverage, encoding)
        label   = f"{severity}_{int(src_rate*100)}pct->{int(tgt_rate*100)}pct"

        if src_key not in datasets or tgt_key not in datasets:
            if verbose:
                print(f"  [dist_shift] SKIP {label} — dataset not found")
            continue

        if verbose:
            print(f"  [dist_shift] {label} | train={src_key} -> test={tgt_key}")

        X_tr, X_val, _, y_tr, y_val, _, _ = datasets[src_key]
        _,    _,    X_te_tgt, _, _, y_te_tgt, _ = datasets[tgt_key]

        # Train calibrated model on SOURCE regime
        model = _train_and_calibrate(X_tr, y_tr, X_val, y_val, cfg, seed)

        # Evaluate on TARGET regime (no retraining)
        proba_transfer = model.predict_proba(X_te_tgt)
        y_te_bin       = (y_te_tgt >= 0.5).astype(int)
        metrics_transfer = full_evaluation(y_te_bin, proba_transfer)

        # In-distribution evaluation for comparison
        _, _, X_te_src, _, _, y_te_src, _ = datasets[src_key]
        proba_indist   = model.predict_proba(X_te_src)
        y_src_bin      = (y_te_src >= 0.5).astype(int)
        metrics_indist = full_evaluation(y_src_bin, proba_indist)

        # Degradation ratios
        deg_ece   = metrics_transfer['ece']   / max(metrics_indist['ece'],   1e-6)
        deg_brier = metrics_transfer['brier'] / max(metrics_indist['brier'], 1e-6)
        auroc_drop = metrics_indist['auroc'] - metrics_transfer['auroc']

        robust_ece   = deg_ece < _ROBUST_ECE_RATIO
        robust_auroc = auroc_drop <= _ROBUST_AUROC_DROP
        robust_both  = robust_ece and robust_auroc

        results[label] = {
            'severity'              : severity,
            'src_rate'              : src_rate,
            'tgt_rate'              : tgt_rate,
            'delta'                 : abs(tgt_rate - src_rate),
            'direction'             : 'up' if tgt_rate > src_rate else 'down',
            'transfer_ece'          : metrics_transfer['ece'],
            'transfer_brier'        : metrics_transfer['brier'],
            'transfer_pr_auc'       : metrics_transfer['pr_auc'],
            'transfer_auroc'        : metrics_transfer['auroc'],
            'indist_ece'            : metrics_indist['ece'],
            'indist_brier'          : metrics_indist['brier'],
            'indist_pr_auc'         : metrics_indist['pr_auc'],
            'indist_auroc'          : metrics_indist['auroc'],
            'deg_ratio_ece'         : deg_ece,
            'deg_ratio_brier'       : deg_brier,
            'auroc_drop'            : auroc_drop,
            'robust_transfer_ece'   : robust_ece,
            'robust_transfer_auroc' : robust_auroc,
            'robust_transfer'       : robust_both,  # BOTH thresholds must pass
        }

        if verbose:
            print(f"    ECE: in-dist={metrics_indist['ece']:.4f} -> transfer={metrics_transfer['ece']:.4f} "
                  f"(ratio={deg_ece:.2f})")
            print(f"    PR-AUC: in-dist={metrics_indist['pr_auc']:.4f} -> "
                  f"transfer={metrics_transfer['pr_auc']:.4f}")
            print(f"    Robust: {'YES' if results[label]['robust_transfer'] else 'NO'}")

    return results


def _train_and_calibrate(X_tr, y_tr, X_val, y_val, cfg, seed):
    """Train + calibrate XGBoost for distribution shift experiments."""
    from calibrate import CalibratedModel
    try:
        import xgboost as xgb  # type: ignore
    except ImportError:
        raise ImportError("xgboost required")

    y_tr_bin  = (y_tr  >= 0.5).astype(int)
    y_val_bin = (y_val >= 0.5).astype(int)

    clf = xgb.XGBClassifier(
        max_depth=4, learning_rate=0.1, n_estimators=300,
        eval_metric='logloss', early_stopping_rounds=20,
        random_state=seed, verbosity=0,
    )
    clf.fit(X_tr, y_tr_bin, eval_set=[(X_val, y_val_bin)], verbose=False)

    cal = CalibratedModel(clf, method='platt', seed=seed)
    cal.fit(X_val, y_val_bin)
    return cal


def _config_key(sub_rate: float, coverage: int, encoding: str) -> str:
    return f'sub{int(sub_rate*100):02d}_k{coverage}_{encoding}'


def compute_transfer_radius(
    results: Dict[str, dict],
    auroc_threshold: float = _ROBUST_AUROC_DROP,
    ece_threshold:   float = _ROBUST_ECE_RATIO,
) -> Dict[str, dict]:
    """Compute formal per-direction transfer radius from a results dict.

    Transfer radius (per direction) = max δ such that *every* tested pair
    with |Δ| ≤ δ satisfies both robustness thresholds.  Pairs are evaluated
    in ascending order of δ; the radius stops at the first delta where any
    pair fails.

    Parameters
    ----------
    results : output of run_distribution_shift()
    auroc_threshold : max acceptable AUROC drop (default 0.05)
    ece_threshold   : max acceptable ECE degradation ratio (default 2.0)

    Returns
    -------
    dict with keys 'up' and 'down', each containing:
        transfer_radius      : max passing δ (0 if no pair passes)
        first_failing_delta  : smallest δ where any pair fails (inf if none fail)
        n_pairs_tested       : number of pairs in that direction
        n_pairs_robust       : number passing both thresholds
    """
    from collections import defaultdict

    by_dir: Dict[str, List[dict]] = defaultdict(list)
    for row in results.values():
        d = row.get('direction', 'up')
        by_dir[d].append(row)

    summary: Dict[str, dict] = {}
    for direction, rows in by_dir.items():
        # Group by unique delta values, ascending
        delta_map: Dict[float, List[dict]] = defaultdict(list)
        for r in rows:
            delta_map[round(r['delta'], 6)].append(r)

        radius           = 0.0
        first_failing    = float('inf')
        found_failure    = False

        for delta in sorted(delta_map):
            pairs_at_delta = delta_map[delta]
            robust_here = all(
                (r['deg_ratio_ece']   < ece_threshold) and
                (r['auroc_drop']      <= auroc_threshold)
                for r in pairs_at_delta
            )
            if robust_here and not found_failure:
                radius = delta
            else:
                if not found_failure:
                    first_failing = delta
                found_failure = True

        summary[direction] = {
            'transfer_radius'     : radius,
            'first_failing_delta' : first_failing,
            'n_pairs_tested'      : len(rows),
            'n_pairs_robust'      : sum(
                1 for r in rows
                if r['deg_ratio_ece'] < ece_threshold and r['auroc_drop'] <= auroc_threshold
            ),
        }

    return summary


def print_shift_table(results: Dict[str, dict]):
    """Print distribution shift summary table."""
    cols = ['dir', 'delta', 'indist_auroc', 'transfer_auroc', 'auroc_drop',
            'deg_ratio_ece', 'robust_transfer']
    print(f"\n{'Condition':<38}" + ''.join(f"{c:>16}" for c in cols))
    print('-' * (38 + 16 * len(cols)))
    for label, row in results.items():
        line = f"{label:<38}"
        for c in cols:
            v = row.get(c, float('nan'))
            if isinstance(v, bool):
                line += f"{'YES' if v else 'NO':>16}"
            elif isinstance(v, str):
                line += f"{v:>16}"
            else:
                line += f"{v:>16.4f}"
        print(line)


def save_shift_results(results: dict, out_path: str):
    """Save distribution shift results as CSV."""
    import pandas as pd
    rows = []
    for label, row in results.items():
        r = {'condition': label}
        r.update({k: v for k, v in row.items()})
        rows.append(r)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"  [dist_shift] Saved to {out_path}")
    return df


# ── CLI entry point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Distribution shift analysis.')
    parser.add_argument('--config',   default='configs/experiment_config.yaml')
    parser.add_argument('--coverage', type=int, default=30)
    parser.add_argument('--encoding', default='simple')
    parser.add_argument('--out',      default='results/distribution_shift/')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    sys.path.insert(0, 'src')
    from dataset_assembler import load_dataset, get_all_config_keys

    # Load relevant datasets
    all_keys = get_all_config_keys(cfg)
    relevant = [k for k in all_keys if f'k{args.coverage}' in k and args.encoding in k]

    datasets = {}
    for key in relevant:
        try:
            datasets[key] = load_dataset(key, cfg)
            print(f"  Loaded {key}")
        except Exception as e:
            print(f"  Could not load {key}: {e}")

    results = run_distribution_shift(
        datasets, cfg, args.coverage, args.encoding,
        seed=cfg['random_seed'], verbose=True
    )
    print_shift_table(results)

    # Formal transfer radius per direction
    radius_summary = compute_transfer_radius(results)
    print('\n  Transfer radius summary:')
    for direction, s in radius_summary.items():
        print(f"    {direction}: radius={s['transfer_radius']:.3f}  "
              f"first_fail={s['first_failing_delta']:.3f}  "
              f"robust={s['n_pairs_robust']}/{s['n_pairs_tested']}")

    out_path = os.path.join(args.out, f'dist_shift_k{args.coverage}_{args.encoding}.csv')
    save_shift_results(results, out_path)

    # Save radius summary
    import pandas as pd
    radius_rows = [{'direction': d, **v} for d, v in radius_summary.items()]
    radius_path = os.path.join(args.out, f'transfer_radius_k{args.coverage}_{args.encoding}.csv')
    os.makedirs(os.path.dirname(radius_path) or '.', exist_ok=True)
    pd.DataFrame(radius_rows).to_csv(radius_path, index=False, float_format='%.6f')
    print(f"  [dist_shift] Radius summary saved to {radius_path}")


if __name__ == '__main__':
    main()
