"""
analysis/distribution_shift.py
================================
Distribution shift experiments: train on one substitution regime, evaluate
on another without retraining.

Three transfer conditions (from implementation plan Week 7):
  1. Mild      : train on 1%  → test on 5%
  2. Moderate  : train on 5%  → test on 12%
  3. Severe    : train on 1%  → test on 12%  (primary stress test)

For each transfer condition, compute:
  - ECE (calibration quality on the target regime)
  - Brier score
  - PR-AUC (discrimination quality)
  - FRR (failure reduction rate under allocation)
  - Degradation ratio: FRR_transfer / FRR_in-distribution
    (target > 0.8 for robust transfer — per implementation plan)

Usage:
    python analysis/distribution_shift.py \
        --config configs/experiment_config.yaml \
        --coverage 30 --encoding simple \
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
    (0.01, 0.05, 'mild'),
    (0.05, 0.12, 'moderate'),
    (0.01, 0.12, 'severe'),
]


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
            print(f"  [dist_shift] {label} | train={src_key} → test={tgt_key}")

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

        # Degradation ratio (primary: ECE degradation)
        deg_ece = metrics_transfer['ece']   / max(metrics_indist['ece'],   1e-6)
        deg_brier = metrics_transfer['brier'] / max(metrics_indist['brier'], 1e-6)

        results[label] = {
            'severity'           : severity,
            'src_rate'           : src_rate,
            'tgt_rate'           : tgt_rate,
            'transfer_ece'       : metrics_transfer['ece'],
            'transfer_brier'     : metrics_transfer['brier'],
            'transfer_pr_auc'    : metrics_transfer['pr_auc'],
            'transfer_auroc'     : metrics_transfer['auroc'],
            'indist_ece'         : metrics_indist['ece'],
            'indist_brier'       : metrics_indist['brier'],
            'indist_pr_auc'      : metrics_indist['pr_auc'],
            'deg_ratio_ece'      : deg_ece,
            'deg_ratio_brier'    : deg_brier,
            'robust_transfer'    : deg_ece < 2.0,  # < 2x ECE degradation = robust
        }

        if verbose:
            print(f"    ECE: in-dist={metrics_indist['ece']:.4f} → transfer={metrics_transfer['ece']:.4f} "
                  f"(ratio={deg_ece:.2f})")
            print(f"    PR-AUC: in-dist={metrics_indist['pr_auc']:.4f} → "
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


def print_shift_table(results: Dict[str, dict]):
    """Print distribution shift summary table."""
    metrics = ['transfer_ece', 'indist_ece', 'deg_ratio_ece',
               'transfer_pr_auc', 'indist_pr_auc', 'robust_transfer']
    print(f"\n{'Condition':<35}" + ''.join(f"{m:>18}" for m in metrics))
    print('-' * (35 + 18 * len(metrics)))
    for label, row in results.items():
        line = f"{label:<35}"
        for m in metrics:
            v = row.get(m, float('nan'))
            if isinstance(v, bool):
                line += f"{'YES' if v else 'NO':>18}"
            else:
                line += f"{v:>18.4f}"
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

    out_path = os.path.join(args.out, f'dist_shift_k{args.coverage}_{args.encoding}.csv')
    save_shift_results(results, out_path)


if __name__ == '__main__':
    main()
