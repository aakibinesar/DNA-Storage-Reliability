"""
allocation/significance.py
===========================
Paired significance testing for the allocation results in results/allocation/.

Background
----------
run_allocation_experiment() (see experiment.py) evaluates all eight allocation
conditions (uniform, oracle, xgb_cal, xgb_raw, and four rule-based baselines)
against the *same* channel noise realisation within each of the n_runs Monte
Carlo runs -- only the parity allocation differs between conditions at a given
run_idx. That makes ofr_<condition> and ofr_uniform paired samples, not
independent ones: a paired test (Wilcoxon signed-rank) removes the shared
run-to-run channel noise instead of treating it as extra variance, and is
the statistically correct choice here.

The headline oracle-vs-uniform breakdown reported in the README (47/27/10)
is a point-estimate comparison of means. This module adds the missing
significance layer: for every config x delta combination, is the observed
gap between a given condition and uniform allocation distinguishable from
noise, and after correcting for running 84 tests at once?

Four comparisons are tested per config x delta (the last two only exist for
delta in {2, 4}, since the benefit-aware model was not trained at delta=1):
  oracle_vs_uniform              -- the R4-fixed theoretical ceiling vs. uniform
                                     (diagnostic: oracle-sized budget)
  xgb_cal_vs_uniform             -- risk model vs. uniform, oracle-sized budget
                                     (diagnostic: ranking quality only)
  xgb_cal_deployable_vs_uniform  -- risk model vs. uniform, validation-chosen
                                     budget (the paper's actual deployable claim)
  benefit_model_deployable_vs_uniform -- benefit-aware model (Part B) vs.
                                     uniform, validation-chosen budget (the
                                     paper's actual deployable claim)

Multiple-comparisons correction: Benjamini-Hochberg FDR at alpha=0.05,
applied separately within each comparison's family of 84 tests.

Outputs
-------
  <out>/allocation_significance.csv -- one row per (config, delta, comparison)

Usage:
    python allocation/significance.py \\
        --results-dir results/allocation/ \\
        --out results/allocation_significance/
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

_FILENAME_RE = re.compile(r'^(?P<key>.+)_delta(?P<delta>\d+)\.npz$')

# condition -> comparison name
_COMPARISONS = {
    'ofr_oracle':                   'oracle_vs_uniform',
    'ofr_xgb_cal':                  'xgb_cal_vs_uniform',
    'ofr_xgb_cal_deployable':       'xgb_cal_deployable_vs_uniform',
    'ofr_benefit_model_deployable': 'benefit_model_deployable_vs_uniform',
}


def _benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """BH FDR-adjusted p-values. NaN inputs pass through as NaN."""
    n = len(pvals)
    adjusted = np.full(n, np.nan)
    valid_mask = ~np.isnan(pvals)
    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) == 0:
        return adjusted

    valid_p = pvals[valid_idx]
    order = np.argsort(valid_p)
    ranked_p = valid_p[order]
    m = len(ranked_p)

    bh = ranked_p * m / (np.arange(1, m + 1))
    # Enforce monotonicity from the largest p-value down (standard BH step-up)
    bh = np.minimum.accumulate(bh[::-1])[::-1]
    bh = np.clip(bh, 0, 1)

    out = np.empty(m)
    out[order] = bh
    adjusted[valid_idx] = out
    return adjusted


def _paired_test(condition: np.ndarray, uniform: np.ndarray) -> dict:
    """Wilcoxon signed-rank test between paired condition/uniform OFR arrays."""
    diff = condition - uniform  # negative = condition has fewer failures (better)
    n_better = int(np.sum(diff < 0))
    n_worse  = int(np.sum(diff > 0))
    n_tied   = int(np.sum(diff == 0))
    median_diff = float(np.median(diff))
    mean_diff = float(np.mean(diff))

    if n_tied == len(diff):
        # All paired differences are exactly zero -- wilcoxon is undefined here.
        return dict(median_diff=0.0, mean_diff=0.0, n_better=n_better, n_worse=n_worse,
                    n_tied=n_tied, statistic=np.nan, p_value=np.nan)

    try:
        stat, p = wilcoxon(condition, uniform, zero_method='wilcox')
    except ValueError:
        stat, p = np.nan, np.nan

    return dict(median_diff=median_diff, mean_diff=mean_diff, n_better=n_better, n_worse=n_worse,
                n_tied=n_tied, statistic=float(stat) if np.isfinite(stat) else np.nan,
                p_value=float(p) if np.isfinite(p) else np.nan)


def run_allocation_significance(results_dir: str, verbose: bool = True) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(results_dir, '*.npz')))
    if not files:
        raise FileNotFoundError(f"No .npz files found in {results_dir}")

    rows = []
    for path in files:
        fname = os.path.basename(path)
        m = _FILENAME_RE.match(fname)
        if not m:
            if verbose:
                print(f"  [SKIP] {fname}: does not match '<key>_delta<N>.npz'")
            continue
        key, delta = m.group('key'), int(m.group('delta'))

        data = np.load(path)
        if 'ofr_uniform' not in data.files:
            if verbose:
                print(f"  [SKIP] {fname}: missing 'ofr_uniform'")
            continue
        uniform = data['ofr_uniform']

        for condition_key, comparison_name in _COMPARISONS.items():
            if condition_key not in data.files:
                continue
            result = _paired_test(data[condition_key], uniform)
            rows.append({
                'key': key,
                'delta': delta,
                'comparison': comparison_name,
                'n_pairs': len(uniform),
                **result,
            })

    df = pd.DataFrame(rows)

    # BH-FDR correction, separately within each comparison's family of tests
    df['p_adjusted'] = np.nan
    for comparison_name in df['comparison'].unique():
        mask = df['comparison'] == comparison_name
        df.loc[mask, 'p_adjusted'] = _benjamini_hochberg(df.loc[mask, 'p_value'].to_numpy())

    df['significant_at_0.05'] = df['p_adjusted'] < 0.05
    df.loc[df['p_adjusted'].isna(), 'significant_at_0.05'] = False

    # Direction of a significant result is taken from the MEAN paired difference (negative =
    # fewer failures than uniform = better). The median can be exactly 0 for a significant
    # result, which made a median-based better/worse split ambiguous.
    df['direction'] = np.where(~df['significant_at_0.05'], 'ns',
                               np.where(df['mean_diff'] < 0, 'better', 'worse'))

    return df


def main():
    parser = argparse.ArgumentParser(
        description='Paired Wilcoxon significance tests for allocation results, '
                    'with BH-FDR correction across all config x delta combinations.'
    )
    parser.add_argument('--results-dir', default='results/allocation/')
    parser.add_argument('--out', default='results/allocation_significance/')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    verbose = not args.quiet
    df = run_allocation_significance(args.results_dir, verbose=verbose)

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, 'allocation_significance.csv')
    df.to_csv(out_path, index=False, float_format='%.6f')
    print(f"\n[allocation_significance] Saved {len(df)} rows -> {out_path}")

    for comparison_name in df['comparison'].unique():
        sub = df[df['comparison'] == comparison_name]
        n_sig = int(sub['significant_at_0.05'].sum())
        n_sig_better = int((sub['direction'] == 'better').sum())
        n_sig_worse  = int((sub['direction'] == 'worse').sum())
        print(f"\n  -- {comparison_name} ({len(sub)} tests, BH-FDR alpha=0.05) --")
        print(f"  Significant: {n_sig}/{len(sub)}  "
              f"({n_sig_better} significantly better than uniform, "
              f"{n_sig_worse} significantly worse)")


if __name__ == '__main__':
    main()
