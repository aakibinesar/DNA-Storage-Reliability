"""
allocation/mechanism.py
=======================
Budget-neutral two-threshold greedy redundancy redistribution.

Given calibrated risk scores r_i ∈ [0, 1] for N sequences and a fixed total
redundancy budget B = N × L_RS_default:

  1. Sort sequences by risk score.
  2. Label the top-p fraction as HIGH-RISK   → allocate L_RS + delta parity bytes.
  3. Label the bottom-p fraction as LOW-RISK  → allocate L_RS - delta parity bytes.
  4. Middle fraction keeps L_RS (neutral tier).

Budget neutrality: |HIGH| × delta = |LOW| × delta (by construction: |HIGH| = |LOW| = p × N).

The efficiency of the allocation depends on how well the risk scores predict
actual failures (ECE matters here: miscalibrated models may assign high risk to
low-failure sequences, wasting the extra redundancy on "safe" sequences).

Oracle condition: use empirical failure frequency (true labels) as risk scores.
Miscalibrated baseline: use uncalibrated raw XGBoost outputs.
Uniform baseline: keep L_RS identical for all sequences (no reallocation).

Usage:
    from allocation.mechanism import AllocationMechanism
    alloc = AllocationMechanism(l_rs_default=8, delta=2, budget_tolerance=1e-6)
    l_rs_alloc = alloc.allocate(risk_scores)
"""

import numpy as np
from typing import Optional, Tuple


class AllocationMechanism:
    """Two-threshold greedy redundancy allocation under a fixed byte budget.

    Parameters
    ----------
    l_rs_default    : default RS parity bytes per sequence (baseline)
    delta           : parity byte increment/decrement (sensitivity parameter)
    l_rs_min        : minimum allowed parity bytes (hard floor)
    l_rs_max        : maximum allowed parity bytes (hard ceiling)
    budget_tolerance: floating-point tolerance for budget neutrality assertion
    """

    def __init__(
        self,
        l_rs_default: int = 8,
        delta: int = 2,
        l_rs_min: int = 4,
        l_rs_max: int = 16,
        budget_tolerance: float = 1e-6,
    ):
        self.l_rs_default     = l_rs_default
        self.delta            = delta
        self.l_rs_min         = l_rs_min
        self.l_rs_max         = l_rs_max
        self.budget_tolerance = budget_tolerance

    def allocate(
        self,
        risk_scores: np.ndarray,
        tier_fraction: Optional[float] = None,
    ) -> np.ndarray:
        """Allocate RS parity bytes based on per-sequence risk scores.

        Parameters
        ----------
        risk_scores    : array of shape (N,) with values in [0, 1]
        tier_fraction  : fraction of sequences to place in HIGH/LOW tiers.
                         If None, uses the maximum balanced fraction that
                         satisfies l_rs_min and l_rs_max constraints.

        Returns
        -------
        l_rs_alloc : integer array of shape (N,) — parity bytes per sequence.
                     Budget neutrality: sum(l_rs_alloc) == N × l_rs_default.
        """
        N = len(risk_scores)
        l_rs_alloc = np.full(N, self.l_rs_default, dtype=int)

        if N == 0 or self.delta == 0:
            return l_rs_alloc

        # Determine how many sequences can actually change.
        # NOTE (fix): max_high/max_low as originally written scale with N and are
        # not bounded by N//2, which lets n_tier exceed N. When that happens the
        # high-risk and low-risk index slices both collapse to the *entire* array,
        # so every sequence ends up assigned l_rs_default - delta and budget
        # neutrality is silently violated. The high/low tiers can never overlap,
        # so n_tier must additionally be capped at N // 2.
        max_high = N * (self.l_rs_max - self.l_rs_default) // self.delta
        max_low  = N * (self.l_rs_default - self.l_rs_min) // self.delta
        max_tier = min(max_high, max_low, N // 2)
        if max_tier == 0:
            return l_rs_alloc

        if tier_fraction is None:
            n_tier = max_tier
        else:
            n_tier = min(int(N * tier_fraction), max_tier)

        n_tier = max(1, n_tier)

        # Rank by risk score
        sorted_idx = np.argsort(risk_scores)
        low_risk_idx  = sorted_idx[:n_tier]    # lowest-risk sequences
        high_risk_idx = sorted_idx[-n_tier:]   # highest-risk sequences

        l_rs_alloc[high_risk_idx] = np.clip(
            self.l_rs_default + self.delta, self.l_rs_min, self.l_rs_max
        )
        l_rs_alloc[low_risk_idx] = np.clip(
            self.l_rs_default - self.delta, self.l_rs_min, self.l_rs_max
        )

        # Budget neutrality check
        total_budget  = N * self.l_rs_default
        actual_budget = int(l_rs_alloc.sum())
        if abs(actual_budget - total_budget) > self.budget_tolerance * total_budget:
            # Correct residual rounding: adjust the middle-tier sequences
            diff = total_budget - actual_budget
            neutral_idx = sorted_idx[n_tier:-n_tier] if n_tier < N // 2 else []
            if len(neutral_idx) > 0 and diff != 0:
                # Distribute correction across neutral sequences (fractional bytes → round)
                per_seq = diff // len(neutral_idx)
                remainder = diff - per_seq * len(neutral_idx)
                l_rs_alloc[neutral_idx] = np.clip(
                    l_rs_alloc[neutral_idx] + per_seq, self.l_rs_min, self.l_rs_max
                )
                if remainder > 0 and len(neutral_idx) > 0:
                    l_rs_alloc[neutral_idx[0]] = np.clip(
                        l_rs_alloc[neutral_idx[0]] + remainder, self.l_rs_min, self.l_rs_max
                    )

        self._assert_budget_neutral(l_rs_alloc, N)
        return l_rs_alloc

    def _assert_budget_neutral(self, l_rs_alloc: np.ndarray, N: int):
        """Raise if budget neutrality is violated beyond tolerance."""
        total   = int(l_rs_alloc.sum())
        budget  = N * self.l_rs_default
        deficit = abs(total - budget)
        if deficit > max(1, self.budget_tolerance * budget):
            raise RuntimeError(
                f"Budget neutrality violation: allocated={total}, budget={budget}, "
                f"deficit={deficit}"
            )

    def tier_summary(
        self,
        risk_scores: np.ndarray,
        l_rs_alloc: np.ndarray,
    ) -> dict:
        """Return tier assignment statistics for diagnostic purposes."""
        N = len(risk_scores)
        high_mask    = l_rs_alloc > self.l_rs_default
        low_mask     = l_rs_alloc < self.l_rs_default
        neutral_mask = l_rs_alloc == self.l_rs_default

        return {
            'n_high'           : int(high_mask.sum()),
            'n_low'            : int(low_mask.sum()),
            'n_neutral'        : int(neutral_mask.sum()),
            'frac_high'        : float(high_mask.mean()),
            'mean_risk_high'   : float(risk_scores[high_mask].mean()) if high_mask.any() else 0.0,
            'mean_risk_low'    : float(risk_scores[low_mask].mean()) if low_mask.any() else 0.0,
            'mean_l_rs'        : float(l_rs_alloc.mean()),
            'total_budget'     : int(l_rs_alloc.sum()),
            'expected_budget'  : N * self.l_rs_default,
        }


# ── Oracle and uniform baselines ────────────────────────────────────────────

def uniform_allocation(N: int, l_rs_default: int) -> np.ndarray:
    """Baseline: all sequences get identical parity allocation."""
    return np.full(N, l_rs_default, dtype=int)


def oracle_allocation_baseline(
    empirical_failure_freq: np.ndarray,
    l_rs_default: int,
    delta: int,
    l_rs_min: int = 4,
    l_rs_max: int = 16,
) -> np.ndarray:
    """Baseline risk-ranking oracle: ranks by failure frequency at default parity.

    NOTE: this is NOT a true allocation oracle. A sequence with high baseline
    failure_freq may gain nothing from extra parity (e.g., if it is already in
    the over-failure regime). Use oracle_allocation_marginal for a valid oracle.
    Retained here as a diagnostic reference baseline.
    """
    alloc = AllocationMechanism(l_rs_default, delta, l_rs_min, l_rs_max)
    return alloc.allocate(empirical_failure_freq)


def oracle_allocation_marginal(
    marginal_benefits: np.ndarray,
    l_rs_default: int,
    delta: int,
    l_rs_min: int = 4,
    l_rs_max: int = 16,
) -> np.ndarray:
    """True oracle: rank sequences by marginal failure reduction from extra parity.

    Parameters
    ----------
    marginal_benefits : array (N,) of p_i(l_rs_default) - p_i(l_rs_default + delta).
                        Positive = sequence benefits from promotion; zero/negative =
                        extra parity does not help.

    Returns
    -------
    Budget-neutral parity allocation ranked by marginal benefit.
    Sequences with the highest benefit are promoted to l_rs_default + delta;
    sequences with the lowest benefit are demoted to l_rs_default - delta.
    """
    alloc = AllocationMechanism(l_rs_default, delta, l_rs_min, l_rs_max)
    return alloc.allocate(marginal_benefits)


def miscalibrated_allocation(
    raw_scores: np.ndarray,
    l_rs_default: int,
    delta: int,
    l_rs_min: int = 4,
    l_rs_max: int = 16,
) -> np.ndarray:
    """Miscalibrated baseline: raw model outputs without Platt scaling.

    Used to demonstrate why calibration (ECE) specifically matters for this task.
    """
    alloc = AllocationMechanism(l_rs_default, delta, l_rs_min, l_rs_max)
    return alloc.allocate(raw_scores)
