"""
analysis/transfer_radius.py
============================
R12 mitigation: systematic transfer-radius analysis across all sub_rate pairs.

Motivation
----------
The vague "~3% transfer radius" claim in the paper is derived from only 4
upward transfer conditions (§ distribution_shift.py).  A single-number radius
is not defensible because:
  • It ignores downward transfers (high → low sub_rate).
  • It was measured only from sub09 and sub12 as source rates.
  • It does not vary by coverage depth or encoding scheme.
  • It uses aggregate AUROC, which R11 shows is inflated by degenerate regimes.

This script replaces the single number with a per-stratum radius table that
covers ALL ordered (src, tgt) pairs within a configurable max_delta, using:
  1. The saved calibrated XGBoost models from models/saved/ (consistent with R1
     soft-label regression pipeline; no retraining).
  2. Regime-aware AUROC (informative regime only, regime_lo/hi from config),
     consistent with R11's stratified evaluation.

Robustness criterion (both must hold):
  ECE degradation ratio  = ECE_transfer / ECE_in-dist  < ECE_THRESHOLD (2.0)
  AUROC drop             = AUROC_in-dist − AUROC_transfer ≤ AUROC_THRESHOLD (0.05)

Outputs
-------
  <out>/transfer_radius_pairs.csv   — one row per (stratum, src_key, tgt_key)
  <out>/transfer_radius_summary.csv — one row per (stratum, direction) with
                                       formal radius and first-failing delta

Usage:
    python analysis/transfer_radius.py \\
        --config configs/experiment_config.yaml \\
        --models-dir models/saved/ \\
        --out results/transfer_radius/ \\
        --max-delta 0.12
"""

import argparse
import os
import sys
from collections import defaultdict
from itertools import permutations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))

ECE_THRESHOLD  = 2.0   # ECE must not more than double
AUROC_THRESHOLD = 0.05  # AUROC may not fall more than 5 pp


# ── Model and dataset helpers ─────────────────────────────────────────────────

def _load_model_for_key(models_dir: str, key: str):
    """Return calibrated XGBoost for key, or None if not found."""
    from train import load_models
    m = load_models(models_dir, key)
    return m.get('xgboost')


def _informative_metrics(failure_freq: np.ndarray, y_prob: np.ndarray,
                         lo: float, hi: float, n_bins: int = 10) -> dict:
    """Return informative-regime metrics from stratified_evaluation, or NaN dict."""
    from evaluate import stratified_evaluation
    rows = stratified_evaluation(failure_freq, y_prob,
                                 regime_thresholds=(lo, hi), n_bins=n_bins)
    info = next((r for r in rows if r['regime'] == 'informative'), None)
    if info is None or info['n'] == 0 or info['is_degenerate']:
        return {'auroc': float('nan'), 'ece': float('nan'),
                'brier': float('nan'), 'n': 0, 'is_degenerate': True}
    return info


# ── Core transfer evaluation ─────────────────────────────────────────────────

def evaluate_transfer_pair(
    src_key: str,
    tgt_key: str,
    model,
    datasets: dict,
    lo: float,
    hi: float,
    n_bins: int = 10,
) -> Optional[dict]:
    """Evaluate a single (src, tgt) transfer.

    Parameters
    ----------
    src_key : key whose model is used (train domain)
    tgt_key : key whose test set is used (deployment domain)
    model   : calibrated model loaded from src_key
    datasets: dict of key → (X_tr, X_val, X_te, y_tr, y_val, y_te, feat_names)
    lo, hi  : regime thresholds from config

    Returns
    -------
    dict with metrics, or None if either dataset is missing.
    """
    if src_key not in datasets or tgt_key not in datasets:
        return None

    # In-distribution: model on its own test set
    _, _, X_te_src, _, _, y_te_src, _ = datasets[src_key]
    y_prob_src = model.predict_proba(np.asarray(X_te_src))
    ff_src     = np.asarray(y_te_src, dtype=float)
    m_src = _informative_metrics(ff_src, y_prob_src, lo, hi, n_bins)

    # Transfer: same model on target test set
    _, _, X_te_tgt, _, _, y_te_tgt, _ = datasets[tgt_key]
    y_prob_tgt = model.predict_proba(np.asarray(X_te_tgt))
    ff_tgt     = np.asarray(y_te_tgt, dtype=float)
    m_tgt = _informative_metrics(ff_tgt, y_prob_tgt, lo, hi, n_bins)

    auroc_indist   = m_src['auroc']
    auroc_transfer = m_tgt['auroc']
    ece_indist     = m_src['ece']
    ece_transfer   = m_tgt['ece']

    auroc_drop = auroc_indist - auroc_transfer
    ece_ratio  = ece_transfer / max(ece_indist, 1e-6)

    # Robustness: both criteria must be finite and pass
    is_robust = (
        np.isfinite(auroc_drop)  and auroc_drop <= AUROC_THRESHOLD and
        np.isfinite(ece_ratio)   and ece_ratio  <  ECE_THRESHOLD
    )

    return {
        'auroc_indist'    : auroc_indist,
        'auroc_transfer'  : auroc_transfer,
        'auroc_drop'      : auroc_drop,
        'ece_indist'      : ece_indist,
        'ece_transfer'    : ece_transfer,
        'ece_ratio'       : ece_ratio,
        'brier_indist'    : m_src['brier'],
        'brier_transfer'  : m_tgt['brier'],
        'n_info_src'      : m_src['n'],
        'n_info_tgt'      : m_tgt['n'],
        'degenerate_src'  : m_src['is_degenerate'],
        'degenerate_tgt'  : m_tgt['is_degenerate'],
        'is_robust'       : is_robust,
    }


# ── Radius computation ────────────────────────────────────────────────────────

def compute_radius_from_pairs(pair_rows: List[dict]) -> dict:
    """Compute formal transfer radius for one (stratum, direction) slice.

    Transfer radius = max δ such that every tested pair with |Δ| ≤ δ passes
    both robustness thresholds.  Pairs are grouped by delta and checked in
    ascending order; the radius stops at the first delta where any pair fails.

    Returns
    -------
    dict with transfer_radius, first_failing_delta, n_tested, n_robust.
    """
    if not pair_rows:
        return {
            'transfer_radius'    : float('nan'),
            'first_failing_delta': float('nan'),
            'n_tested'           : 0,
            'n_robust'           : 0,
        }

    # Group by delta
    delta_groups: Dict[float, List[dict]] = defaultdict(list)
    for r in pair_rows:
        delta_groups[round(r['delta'], 6)].append(r)

    radius        = 0.0
    first_failing = float('inf')
    found_failure = False

    for delta in sorted(delta_groups):
        all_robust_at_delta = all(r['is_robust'] for r in delta_groups[delta])
        if all_robust_at_delta and not found_failure:
            radius = delta
        else:
            if not found_failure:
                first_failing = delta
            found_failure = True

    return {
        'transfer_radius'    : radius,
        'first_failing_delta': first_failing,
        'n_tested'           : len(pair_rows),
        'n_robust'           : sum(1 for r in pair_rows if r['is_robust']),
    }


# ── Full sweep ────────────────────────────────────────────────────────────────

def run_transfer_radius(
    sub_rates : List[float],
    coverages : List[int],
    encodings : List[str],
    datasets  : dict,
    models_dir: str,
    cfg       : dict,
    max_delta : float = 0.12,
    verbose   : bool  = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Sweep all (src, tgt) pairs within max_delta; return pair and summary DFs.

    Parameters
    ----------
    sub_rates  : list of substitution rates in the experiment
    coverages  : coverage depth values
    encodings  : encoding scheme names
    datasets   : pre-loaded datasets dict
    models_dir : directory holding saved models
    cfg        : experiment config dict
    max_delta  : skip pairs where |src - tgt| > max_delta

    Returns
    -------
    pair_df    : one row per tested (stratum, src_key, tgt_key) pair
    summary_df : one row per (stratum, direction) with transfer radius
    """
    lo     = cfg.get('evaluation', {}).get('regime_lo', 0.15)
    hi     = cfg.get('evaluation', {}).get('regime_hi', 0.85)
    n_bins = cfg.get('evaluation', {}).get('ece_n_bins', 10)

    all_pairs: List[dict] = []

    for cov in coverages:
        for enc in encodings:
            stratum = f'k{cov}_{enc}'
            if verbose:
                print(f'\n[transfer_radius] stratum={stratum}')

            # Ordered pairs within max_delta
            rate_pairs = [
                (s, t) for s in sub_rates for t in sub_rates
                if s != t and abs(s - t) <= max_delta
            ]

            for src_rate, tgt_rate in rate_pairs:
                src_key = f'sub{int(src_rate*100):02d}_k{cov}_{enc}'
                tgt_key = f'sub{int(tgt_rate*100):02d}_k{cov}_{enc}'

                model = _load_model_for_key(models_dir, src_key)
                if model is None:
                    if verbose:
                        print(f'  [SKIP] no model for {src_key}')
                    continue

                result = evaluate_transfer_pair(
                    src_key, tgt_key, model, datasets, lo, hi, n_bins
                )
                if result is None:
                    if verbose:
                        print(f'  [SKIP] dataset missing for {src_key}->{tgt_key}')
                    continue

                direction = 'up' if tgt_rate > src_rate else 'down'
                delta     = abs(tgt_rate - src_rate)

                row = {
                    'stratum'     : stratum,
                    'coverage'    : cov,
                    'encoding'    : enc,
                    'src_rate'    : src_rate,
                    'tgt_rate'    : tgt_rate,
                    'src_key'     : src_key,
                    'tgt_key'     : tgt_key,
                    'delta'       : delta,
                    'direction'   : direction,
                    'ece_threshold'  : ECE_THRESHOLD,
                    'auroc_threshold': AUROC_THRESHOLD,
                    **result,
                }
                all_pairs.append(row)

                if verbose:
                    status = 'ROBUST' if result['is_robust'] else 'FAIL'
                    print(f'  {src_key}->{tgt_key}  Delta={delta:.3f} [{direction}]  '
                          f'AUROC_drop={result["auroc_drop"]:+.4f}  '
                          f'ECE_ratio={result["ece_ratio"]:.3f}  [{status}]')

    pair_df = pd.DataFrame(all_pairs)

    # ── Radius summary: one row per (stratum, direction) ─────────────────────
    summary_rows: List[dict] = []
    if not pair_df.empty:
        for (stratum, direction), grp in pair_df.groupby(['stratum', 'direction']):
            rad = compute_radius_from_pairs(grp.to_dict('records'))
            cov, enc = grp[['coverage', 'encoding']].iloc[0]
            summary_rows.append({
                'stratum'         : stratum,
                'coverage'        : cov,
                'encoding'        : enc,
                'direction'       : direction,
                'ece_threshold'   : ECE_THRESHOLD,
                'auroc_threshold' : AUROC_THRESHOLD,
                **rad,
            })

    summary_df = pd.DataFrame(summary_rows)
    return pair_df, summary_df


def _print_radius_table(summary_df: pd.DataFrame):
    """Print a compact per-stratum radius summary table."""
    if summary_df.empty:
        print('  (no results)')
        return

    hdr = (f"  {'Stratum':<15} {'Dir':<6} {'Radius':>8} {'1st fail':>10} "
           f"{'Robust/N':>10}")
    print(hdr)
    print(f"  {'-' * 55}")
    for _, r in summary_df.sort_values(['stratum', 'direction']).iterrows():
        rad = f"{r['transfer_radius']:.3f}" if np.isfinite(r['transfer_radius']) else '—'
        ff  = f"{r['first_failing_delta']:.3f}" if np.isfinite(r['first_failing_delta']) else '∞'
        print(f"  {r['stratum']:<15} {r['direction']:<6} {rad:>8} {ff:>10} "
              f"  {int(r['n_robust'])}/{int(r['n_tested'])}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='R12: systematic transfer-radius analysis across all sub_rate pairs.'
    )
    parser.add_argument('--config',     default='configs/experiment_config.yaml')
    parser.add_argument('--models-dir', default='models/saved/')
    parser.add_argument('--out',        default='results/transfer_radius/')
    parser.add_argument('--max-delta',  type=float, default=0.12,
                        help='Only test pairs with |src_rate − tgt_rate| ≤ this value.')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    from dataset_assembler import load_dataset

    sub_rates = cfg['channel']['substitution_rates']
    coverages = cfg['coverage_depths']
    encodings = cfg['sequence']['encoding_schemes']

    # Build key set for pairs within max_delta (to avoid loading all datasets)
    needed_keys = set()
    for cov in coverages:
        for enc in encodings:
            for s in sub_rates:
                for t in sub_rates:
                    if s != t and abs(s - t) <= args.max_delta:
                        needed_keys.add(f'sub{int(s*100):02d}_k{cov}_{enc}')
                        needed_keys.add(f'sub{int(t*100):02d}_k{cov}_{enc}')

    datasets = {}
    print(f'[transfer_radius] Loading {len(needed_keys)} datasets ...')
    for key in sorted(needed_keys):
        try:
            datasets[key] = load_dataset(key, cfg)
        except Exception as e:
            print(f'  [WARN] Could not load {key}: {e}')

    print(f'[transfer_radius] Loaded {len(datasets)} datasets.')

    pair_df, summary_df = run_transfer_radius(
        sub_rates=sub_rates,
        coverages=coverages,
        encodings=encodings,
        datasets=datasets,
        models_dir=args.models_dir,
        cfg=cfg,
        max_delta=args.max_delta,
        verbose=not args.quiet,
    )

    print('\n[transfer_radius] Radius summary (regime-aware AUROC, informative regime):')
    _print_radius_table(summary_df)

    os.makedirs(args.out, exist_ok=True)
    pair_path    = os.path.join(args.out, 'transfer_radius_pairs.csv')
    summary_path = os.path.join(args.out, 'transfer_radius_summary.csv')

    pair_df.to_csv(pair_path,    index=False, float_format='%.6f')
    summary_df.to_csv(summary_path, index=False, float_format='%.6f')

    print(f'\n[transfer_radius] Saved {len(pair_df)} pair rows -> {pair_path}')
    print(f'[transfer_radius] Saved {len(summary_df)} summary rows -> {summary_path}')

    # Print a concise "what the paper should say" conclusion
    if not summary_df.empty:
        print('\n  -- Defensible claim (replace "~3% transfer radius") --')
        for _, r in summary_df.sort_values(['stratum', 'direction']).iterrows():
            rad = f"{r['transfer_radius']*100:.1f}%" if np.isfinite(r['transfer_radius']) else 'N/A'
            ff  = (f"{r['first_failing_delta']*100:.1f}%"
                   if np.isfinite(r['first_failing_delta']) else 'none')
            print(f"  {r['stratum']} [{r['direction']}]: "
                  f"radius={rad}, first_fail={ff} "
                  f"({int(r['n_robust'])}/{int(r['n_tested'])} pairs robust)")


if __name__ == '__main__':
    main()
