"""
allocation/baselines.py
========================
R10 mitigation: rule-based allocation baselines that compete with the ML model
in the Layer 2 allocation experiment.

Background
----------
The paper claims the calibrated XGBoost model reduces OFR relative to uniform
allocation.  A reviewer could ask: does a simple deterministic rule (allocate
more parity to sequences with extreme GC or long HP runs) achieve the same
effect?  If it does, the ML contribution is minimal.

This module defines four baseline risk scores that are passed to the same
AllocationMechanism used by the ML model, so budget-neutral constraints are
enforced identically for all conditions.

Baselines
---------
  gc_dev    — |gc_content - 0.5|; extreme GC content correlates with higher
              failure probability through the pcr_bias reliability-skew mechanism.
  hp        — max homopolymer run length; long HP runs correlate with higher
              failure probability through the hp_stutter mechanism.
  composite — weighted average of min-max-normalised gc_dev and hp scores
              (equal weights); the best purely rule-based competitor.
  random    — uniform random scores; allocation sanity check — should perform
              no better than uniform on average.

If XGBoost outperforms composite, the ML model is contributing beyond what can
be obtained by a two-feature heuristic, providing genuine value.
"""

from typing import Dict, List
import numpy as np


# -- Sequence-level feature extractors ----------------------------------------

def _gc_content(seq: str) -> float:
    return sum(1 for c in seq if c in 'GC') / len(seq) if seq else 0.5


def _max_hp_run(seq: str) -> int:
    """Length of the longest homopolymer run in seq."""
    if not seq:
        return 0
    max_run = cur_run = 1
    for i in range(1, len(seq)):
        cur_run = cur_run + 1 if seq[i] == seq[i - 1] else 1
        if cur_run > max_run:
            max_run = cur_run
    return max_run


# -- Risk score vectors --------------------------------------------------------

def gc_deviation_scores(sequences: List[str]) -> np.ndarray:
    """Risk score = |gc_content - 0.5|; range [0, 0.5].

    Higher -> sequence has extreme GC content -> more parity allocated.
    The AllocationMechanism only uses rankings, not absolute values.
    """
    return np.array([abs(_gc_content(s) - 0.5) for s in sequences])


def hp_scores(sequences: List[str]) -> np.ndarray:
    """Risk score = max homopolymer run length; range [1, seq_len].

    Higher -> sequence has longer HP runs -> more parity allocated.
    """
    return np.array([float(_max_hp_run(s)) for s in sequences])


def composite_scores(sequences: List[str], gc_weight: float = 0.5) -> np.ndarray:
    """Weighted average of min-max-normalised GC-deviation and HP scores.

    Both components are normalised to [0, 1] before combining so that neither
    dominates due to scale differences.  Default gc_weight=0.5 gives equal
    contribution.
    """
    gc = gc_deviation_scores(sequences)
    hp = hp_scores(sequences)
    gc_norm = gc / (gc.max() + 1e-10)
    hp_norm = hp / (hp.max() + 1e-10)
    return gc_weight * gc_norm + (1.0 - gc_weight) * hp_norm


def random_scores(n: int, seed: int = 8000) -> np.ndarray:
    """Uniformly random risk scores — sanity check / allocation control.

    Expected to perform no better than uniform allocation on average.
    """
    return np.random.default_rng(seed).uniform(0.0, 1.0, size=n)


# -- Combined allocation helper ------------------------------------------------

def get_baseline_allocations(
    sequences:     List[str],
    l_rs_default:  int,
    delta:         int,
    l_rs_min:      int,
    l_rs_max:      int,
    random_seed:   int = 8000,
    tier_fraction: float = None,
) -> Dict[str, np.ndarray]:
    """Compute l_rs allocation vectors for all four rule-based baselines.

    All baselines use the same AllocationMechanism as the ML model so that
    the budget-neutral constraint is enforced identically for all conditions.

    Parameters
    ----------
    random_seed   : seed for the random baseline; keep well away from channel
                     and model seeds (default 8000 is outside all used ranges)
    tier_fraction  : R4 fix -- fraction of sequences to promote/demote. Pass the
                     oracle's n_star / N (from oracle_allocation_greedy_swap) so
                     every baseline is compared against the model and oracle using
                     the same budget size, differing only in ranking quality.
                     None falls back to AllocationMechanism's old max-tier default.

    Returns
    -------
    dict with keys 'gc_dev', 'hp', 'composite', 'random' — each an ndarray
    (N,) of per-sequence l_rs values.
    """
    from mechanism import AllocationMechanism  # local import to avoid circular deps
    alloc = AllocationMechanism(l_rs_default, delta, l_rs_min, l_rs_max)
    N = len(sequences)
    return {
        'gc_dev'   : alloc.allocate(gc_deviation_scores(sequences), tier_fraction),
        'hp'       : alloc.allocate(hp_scores(sequences), tier_fraction),
        'composite': alloc.allocate(composite_scores(sequences), tier_fraction),
        'random'   : alloc.allocate(random_scores(N, seed=random_seed), tier_fraction),
    }
