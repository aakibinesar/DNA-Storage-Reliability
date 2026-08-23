"""
analysis/encoding_confound.py
==============================
R14 mitigation: deconfound the simple vs. constrained encoding comparison.

Background
----------
The paper compares oligo failure rates between two encoding schemes:
  • simple      — unconstrained 2-bit mapping; no restrictions on GC or HP
  • constrained — approximate R-infinity/P8 encoding; enforces
                  GC content in [0.40, 0.60] and max homopolymer run <= 3

The concern is that the raw OFR difference between encodings is confounded
by GC content and homopolymer run length: both factors independently increase
failure frequency, and simple sequences can have extreme GC and long HP runs
while constrained sequences cannot.  Attributing the full OFR difference to
the encoding scheme is therefore misleading.

Approach
--------
For each (sub_rate, coverage) stratum:

1. Extract gc_global and homopolymer_max_run from the test feature matrices
   for both encodings.

2. Define the "constrained-encoding support": GC in [gc_lo, gc_hi] AND
   max_hp <= hp_max (from experiment config).  Simple sequences outside
   this support are structurally incomparable to any constrained sequence.

3. Compute three encoding-effect estimates:
     raw_effect     = mean(ff_simple) - mean(ff_constrained)
     overlap_effect = mean(ff_simple | in_support) - mean(ff_constrained)
     adjusted_effect = post-stratification estimate controlling for
                       (GC_bin, HP_bin) cell membership

4. Confound fraction = (raw_effect - adjusted_effect) / raw_effect
   i.e. what fraction of the raw difference disappears once GC and HP
   are held constant.

GC bins (5): [0.0, 0.2), [0.2, 0.4), [0.4, 0.6), [0.6, 0.8), [0.8, 1.0]
HP bins (3): <=2, 3, >=4   (split at constrained bound of 3)

Outputs
-------
  <out>/encoding_confound_cells.csv    — per-cell × per-encoding mean failure_freq
  <out>/encoding_confound_summary.csv  — per-stratum effect estimates

Usage:
    python analysis/encoding_confound.py \\
        --config configs/experiment_config.yaml \\
        --out results/encoding_confound/
"""

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'models'))

# ── Binning parameters ────────────────────────────────────────────────────────

# GC bins — chosen so the constrained-encoding range [0.40, 0.60] occupies
# exactly one bin, making the confound comparison visually clean.
GC_EDGES  = [0.0, 0.20, 0.40, 0.60, 0.80, 1.001]
GC_LABELS = ['GC<20%', 'GC20-40%', 'GC40-60%', 'GC60-80%', 'GC>=80%']

# HP bins — np.digitize breakpoints [3, 4] so that:
#   HP < 3  (HP=1,2) -> bin 0  (HP<=2)
#   HP >= 3, < 4     -> bin 1  (HP=3)
#   HP >= 4          -> bin 2  (HP>=4)
HP_EDGES  = [0, 3, 4, 99]
HP_LABELS = ['HP<=2', 'HP=3', 'HP>=4']


# ── Feature extraction helpers ────────────────────────────────────────────────

def _feature_col(X: np.ndarray, feat_names: List[str], name: str) -> np.ndarray:
    """Return a single feature column by exact name."""
    if name not in feat_names:
        raise KeyError(f"Feature '{name}' not found in feat_names")
    return X[:, feat_names.index(name)]


def _gc_bin(gc: np.ndarray) -> np.ndarray:
    return np.digitize(gc, GC_EDGES[1:-1])   # 0-indexed: 0 = GC<20%


def _hp_bin(hp: np.ndarray) -> np.ndarray:
    return np.digitize(hp, HP_EDGES[1:-1])   # 0-indexed: 0 = HP<=2


def _in_constrained_support(gc: np.ndarray, hp: np.ndarray,
                             gc_lo: float = 0.40, gc_hi: float = 0.60,
                             hp_max: int = 3) -> np.ndarray:
    """Boolean mask: True for sequences within the constrained-encoding support."""
    return (gc >= gc_lo) & (gc <= gc_hi) & (hp <= hp_max)


# ── Core analysis ─────────────────────────────────────────────────────────────

def run_encoding_confound(
    src_sub_rate : float,
    coverage     : int,
    datasets     : dict,
    cfg          : dict,
    verbose      : bool = True,
) -> Tuple[Optional[pd.DataFrame], Optional[dict]]:
    """Compare simple vs. constrained encoding, controlling for GC and HP.

    Parameters
    ----------
    src_sub_rate : substitution rate to compare at
    coverage     : coverage depth
    datasets     : dict keyed by config key (key -> dataset tuple)
    cfg          : experiment config dict

    Returns
    -------
    cells_df : per-(GC_bin, HP_bin, encoding) cell statistics, or None
    summary  : dict with raw/overlap/adjusted encoding effects, or None
    """
    gc_lo  = cfg['sequence'].get('gc_min', 0.40)
    gc_hi  = cfg['sequence'].get('gc_max', 0.60)
    hp_max = cfg['sequence'].get('max_homopolymer', 3)

    key_simple = f'sub{int(src_sub_rate*100):02d}_k{coverage}_simple'
    key_const  = f'sub{int(src_sub_rate*100):02d}_k{coverage}_constrained'

    if key_simple not in datasets or key_const not in datasets:
        if verbose:
            print(f"  [SKIP] sub={src_sub_rate} k={coverage}: "
                  f"one or both datasets missing")
        return None, None

    def _extract(key):
        _, _, X_te, _, _, y_te, feat_names = datasets[key]
        feat_names = list(feat_names)
        gc   = _feature_col(X_te, feat_names, 'gc_global')
        hp   = _feature_col(X_te, feat_names, 'homopolymer_max_run')
        ff   = np.asarray(y_te, dtype=float)
        return gc, hp, ff

    gc_s, hp_s, ff_s = _extract(key_simple)
    gc_c, hp_c, ff_c = _extract(key_const)

    # Support mask: which simple sequences are comparable to constrained ones?
    support_mask = _in_constrained_support(gc_s, hp_s, gc_lo, gc_hi, hp_max)
    frac_out_of_support = float((~support_mask).mean())

    if verbose:
        print(f"  sub={src_sub_rate:.2f} k={coverage}: "
              f"simple N={len(ff_s)}, constrained N={len(ff_c)}")
        print(f"    {frac_out_of_support:.1%} of simple sequences are outside "
              f"the constrained-encoding support (GC not in "
              f"[{gc_lo:.2f},{gc_hi:.2f}] or HP>{hp_max})")

    # ── Encoding effects ──────────────────────────────────────────────────────
    raw_effect      = float(np.mean(ff_s) - np.mean(ff_c))
    overlap_effect  = (float(np.mean(ff_s[support_mask]) - np.mean(ff_c))
                       if support_mask.sum() > 0 else float('nan'))

    # Post-stratification: compare within (GC_bin, HP_bin) cells
    gc_bin_s = _gc_bin(gc_s)
    hp_bin_s = _hp_bin(hp_s)
    gc_bin_c = _gc_bin(gc_c)
    hp_bin_c = _hp_bin(hp_c)

    n_gc = len(GC_LABELS)
    n_hp = len(HP_LABELS)

    cell_rows = []
    weighted_diffs = []
    total_weight   = 0.0

    for gi in range(n_gc):
        for hi in range(n_hp):
            mask_s = (gc_bin_s == gi) & (hp_bin_s == hi)
            mask_c = (gc_bin_c == gi) & (hp_bin_c == hi)
            n_s    = int(mask_s.sum())
            n_c    = int(mask_c.sum())
            mean_s = float(np.mean(ff_s[mask_s])) if n_s > 0 else float('nan')
            mean_c = float(np.mean(ff_c[mask_c])) if n_c > 0 else float('nan')

            cell_rows.append({
                'sub_rate'  : src_sub_rate,
                'coverage'  : coverage,
                'gc_bin'    : GC_LABELS[gi],
                'hp_bin'    : HP_LABELS[hi],
                'n_simple'  : n_s,
                'n_const'   : n_c,
                'mean_ff_simple'     : mean_s,
                'mean_ff_constrained': mean_c,
                'within_cell_diff'   : (mean_s - mean_c
                                        if (n_s > 0 and n_c > 0) else float('nan')),
            })

            # Only cells with both encodings contribute to the adjusted estimate
            if n_s > 0 and n_c > 0:
                w = float(n_c)   # weight by constrained-cell size
                weighted_diffs.append(w * (mean_s - mean_c))
                total_weight += w

    adjusted_effect = (sum(weighted_diffs) / total_weight
                       if total_weight > 0 else float('nan'))

    confound_frac = (
        float((raw_effect - adjusted_effect) / abs(raw_effect))
        if abs(raw_effect) > 1e-10 and np.isfinite(adjusted_effect)
        else float('nan')
    )

    if verbose:
        print(f"    raw_effect     = {raw_effect:+.6f}  (all simple - all constrained)")
        print(f"    overlap_effect = {overlap_effect:+.6f}  (in-support simple only)")
        print(f"    adjusted_eff   = {adjusted_effect:+.6f}  (within GC×HP cells)")
        print(f"    confound_frac  = {confound_frac:.1%}  of raw effect is GC/HP confound")

    cells_df = pd.DataFrame(cell_rows)
    summary  = {
        'sub_rate'            : src_sub_rate,
        'coverage'            : coverage,
        'n_simple'            : len(ff_s),
        'n_constrained'       : len(ff_c),
        'gc_lo'               : gc_lo,
        'gc_hi'               : gc_hi,
        'hp_max_constrained'  : hp_max,
        'frac_simple_out_of_support': frac_out_of_support,
        'mean_ff_simple'      : float(np.mean(ff_s)),
        'mean_ff_constrained' : float(np.mean(ff_c)),
        'raw_effect'          : raw_effect,
        'overlap_effect'      : overlap_effect,
        'adjusted_effect'     : adjusted_effect,
        'confound_fraction'   : confound_frac,
        'n_cells_with_both'   : sum(
            1 for r in cell_rows
            if r['n_simple'] > 0 and r['n_const'] > 0
        ),
    }
    return cells_df, summary


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='R14: deconfound simple vs constrained encoding comparison.'
    )
    parser.add_argument('--config', default='configs/experiment_config.yaml')
    parser.add_argument('--out',    default='results/encoding_confound/')
    parser.add_argument('--quiet',  action='store_true')
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    from dataset_assembler import load_dataset

    sub_rates = cfg['channel']['substitution_rates']
    coverages = cfg['coverage_depths']
    encodings = cfg['sequence']['encoding_schemes']

    if 'simple' not in encodings or 'constrained' not in encodings:
        print("[encoding_confound] Both 'simple' and 'constrained' "
              "encoding schemes must be present in config — aborting.")
        sys.exit(1)

    # Load all needed datasets
    needed_keys = [
        f'sub{int(s*100):02d}_k{k}_{e}'
        for s in sub_rates for k in coverages for e in encodings
    ]
    datasets = {}
    print(f"[encoding_confound] Loading {len(needed_keys)} datasets ...")
    for key in needed_keys:
        try:
            datasets[key] = load_dataset(key, cfg)
        except Exception as e:
            print(f"  [WARN] Could not load {key}: {e}")

    all_cells    = []
    all_summaries = []

    for sub_rate in sub_rates:
        for cov in coverages:
            cells_df, summary = run_encoding_confound(
                src_sub_rate=sub_rate,
                coverage=cov,
                datasets=datasets,
                cfg=cfg,
                verbose=not args.quiet,
            )
            if cells_df is not None:
                all_cells.append(cells_df)
            if summary is not None:
                all_summaries.append(summary)

    if not all_summaries:
        print("[encoding_confound] No results — are datasets generated?")
        sys.exit(1)

    os.makedirs(args.out, exist_ok=True)

    cells_path   = os.path.join(args.out, 'encoding_confound_cells.csv')
    summary_path = os.path.join(args.out, 'encoding_confound_summary.csv')

    pd.concat(all_cells, ignore_index=True).to_csv(
        cells_path, index=False, float_format='%.6f'
    )
    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv(summary_path, index=False, float_format='%.6f')

    print(f"\n[encoding_confound] Saved {len(all_cells)} cell tables -> {cells_path}")
    print(f"[encoding_confound] Saved {len(all_summaries)} summary rows -> {summary_path}")

    # Defensible claim summary
    print('\n  -- Defensible claim (replace raw encoding comparison) --')
    print(f"  {'Sub':<6} {'Cov':<5} {'raw_eff':>10} "
          f"{'adj_eff':>10} {'confound%':>10} {'out_of_supp%':>14}")
    print(f"  {'-' * 60}")
    for s in all_summaries:
        raw = s['raw_effect']
        adj = s['adjusted_effect']
        cf  = s['confound_fraction']
        oos = s['frac_simple_out_of_support']
        raw_s = f"{raw:+.4f}" if np.isfinite(raw) else '—'
        adj_s = f"{adj:+.4f}" if np.isfinite(adj) else '—'
        cf_s  = f"{cf:.0%}"   if np.isfinite(cf)  else '—'
        oos_s = f"{oos:.0%}"
        print(f"  {s['sub_rate']:.2f}  k={s['coverage']:<3} "
              f"{raw_s:>10} {adj_s:>10} {cf_s:>10} {oos_s:>14}")


if __name__ == '__main__':
    main()
