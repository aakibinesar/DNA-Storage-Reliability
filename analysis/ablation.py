"""
analysis/ablation.py
====================
Feature ablation analysis: systematically remove feature groups and measure
the resulting drop in ECE, Brier score, and PR-AUC.

Feature groups (5 ablation conditions):
  1. k-mer frequencies (dinucleotide + trinucleotide)
  2. Homopolymer features
  3. GC-based features (global + windowed)
  4. Structural proxies (hairpin, free energy, palindromes)
  5. Positional features (dist_to_start/end, GC first/last quarter)

The ablation is run across all 16 dataset configurations to assess which
features are universally important vs. configuration-specific.

Usage:
    python analysis/ablation.py --config configs/experiment_config.yaml \
        --out results/ablation/
"""

import argparse
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))


# ── Feature group definitions ────────────────────────────────────────────────

FEATURE_GROUPS = {
    'kmer': [
        'dinuc_', 'trinuc_',
    ],
    'homopolymer': [
        'homopolymer_', 'hp_max_',
    ],
    'gc_composition': [
        'gc_global', 'at_global', 'gc_win', 'gc_skew', 'at_skew',
        'freq_A', 'freq_C', 'freq_G', 'freq_T',
    ],
    'structural': [
        'palindrome', 'hairpin', 'approx_free_energy', 'repeat_fraction',
        'inverted_repeat',
    ],
    'positional': [
        'dist_to_start', 'dist_to_end', 'gc_first_quarter', 'gc_last_quarter',
        'gc_middle_half', 'gc_constraint_violation',
    ],
}


def get_group_mask(feature_names: List[str], group: str) -> np.ndarray:
    """Return boolean mask selecting features in the given group."""
    prefixes = FEATURE_GROUPS[group]
    mask = np.zeros(len(feature_names), dtype=bool)
    for i, name in enumerate(feature_names):
        for prefix in prefixes:
            if prefix in name:
                mask[i] = True
                break
    return mask


def run_ablation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    X_test:  np.ndarray,
    y_test:  np.ndarray,
    feature_names: List[str],
    cfg:     dict,
    seed:    int = 42,
    verbose: bool = True,
) -> Dict[str, dict]:
    """Run all 5 group ablations + full-feature baseline.

    Returns
    -------
    dict keyed by condition name, each containing evaluation metric dicts.
    """
    from train import train_all_models
    from evaluate import full_evaluation

    results = {}

    # Full-feature baseline
    if verbose:
        print("  [ablation] Full features (baseline)...")
    baseline_models = _train_quick_xgb(X_train, y_train, X_val, y_val, cfg, seed)
    baseline_proba  = baseline_models.predict_proba(X_test)
    results['full'] = full_evaluation((y_test >= 0.5).astype(int), baseline_proba)

    # Group ablations
    for group in FEATURE_GROUPS:
        mask   = get_group_mask(feature_names, group)
        n_feat = mask.sum()
        if n_feat == 0:
            if verbose:
                print(f"  [ablation] Group '{group}' — no features found, skipping")
            continue

        if verbose:
            print(f"  [ablation] Remove group '{group}' ({n_feat} features)...")

        # Remove the group by setting to zero (permutation-style ablation)
        X_tr_abl  = X_train.copy()
        X_val_abl = X_val.copy()
        X_te_abl  = X_test.copy()
        X_tr_abl[:, mask]  = 0.0
        X_val_abl[:, mask] = 0.0
        X_te_abl[:, mask]  = 0.0

        abl_model = _train_quick_xgb(X_tr_abl, y_train, X_val_abl, y_val, cfg, seed)
        abl_proba = abl_model.predict_proba(X_te_abl)
        results[f'no_{group}'] = full_evaluation((y_test >= 0.5).astype(int), abl_proba)

    # Compute delta vs. baseline
    for key in list(results.keys()):
        if key == 'full':
            continue
        results[key]['delta_ece']    = results[key]['ece']    - results['full']['ece']
        results[key]['delta_pr_auc'] = results[key]['pr_auc'] - results['full']['pr_auc']
        results[key]['delta_brier']  = results[key]['brier']  - results['full']['brier']

    return results


def _train_quick_xgb(X_tr, y_tr, X_val, y_val, cfg, seed):
    """Quick single-param XGBoost fit for ablation (no grid search)."""
    from calibrate import CalibratedModel
    try:
        import xgboost as xgb  # type: ignore
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier as xgb  # type: ignore

    y_tr_bin  = (y_tr  >= 0.5).astype(int)
    y_val_bin = (y_val >= 0.5).astype(int)

    clf = xgb.XGBClassifier(
        max_depth=4, learning_rate=0.1, n_estimators=200,
        use_label_encoder=False, eval_metric='logloss',
        random_state=seed, verbosity=0,
    )
    clf.fit(X_tr, y_tr_bin, eval_set=[(X_val, y_val_bin)],
            early_stopping_rounds=20, verbose=False)

    cal = CalibratedModel(clf, method='platt', seed=seed)
    cal.fit(X_val, y_val_bin)
    return cal


def print_ablation_table(results: Dict[str, dict]):
    """Print formatted ablation table."""
    conditions = ['full'] + [k for k in results if k.startswith('no_')]
    metrics    = ['ece', 'pr_auc', 'brier', 'auroc']

    header = f"{'Condition':<25}" + ''.join(f"{m:>12}" for m in metrics)
    print(header)
    print('-' * len(header))

    for cond in conditions:
        row = f"{cond:<25}"
        for m in metrics:
            row += f"{results[cond].get(m, float('nan')):>12.4f}"
        print(row)


def save_ablation_results(results: dict, out_path: str):
    """Save ablation results as CSV."""
    import pandas as pd
    rows = []
    for condition, metrics in results.items():
        row = {'condition': condition}
        row.update(metrics)
        rows.append(row)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"  [ablation] Saved to {out_path}")
    return df


# ── CLI entry point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Feature ablation analysis.')
    parser.add_argument('--config', default='configs/experiment_config.yaml')
    parser.add_argument('--key',    default='sub05_k30_simple')
    parser.add_argument('--out',    default='results/ablation/')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    from dataset_assembler import load_dataset

    print(f"[ablation] Loading {args.key}")
    X_tr, X_val, X_te, y_tr, y_val, y_te, feat_names = load_dataset(args.key, cfg)
    print(f"  Features: {len(feat_names)}, train={len(y_tr)}, test={len(y_te)}")

    results = run_ablation(
        X_tr, y_tr, X_val, y_val, X_te, y_te, feat_names, cfg,
        seed=cfg['random_seed'], verbose=True
    )
    print_ablation_table(results)

    out_path = os.path.join(args.out, f'{args.key}_ablation.csv')
    save_ablation_results(results, out_path)


if __name__ == '__main__':
    main()
