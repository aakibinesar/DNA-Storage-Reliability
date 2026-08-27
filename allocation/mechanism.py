"""
allocation/mechanism.py
=======================
Budget-neutral two-threshold greedy redundancy redistribution.

Given calibrated risk scores r_i in [0, 1] for N sequences and a fixed total
redundancy budget B = N × L_RS_default:

  1. Sort sequences by risk score.
  2. Label the top-p fraction as HIGH-RISK   -> allocate L_RS + delta parity bytes.
  3. Label the bottom-p fraction as LOW-RISK  -> allocate L_RS - delta parity bytes.
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
            n_tier = max(1, max_tier)
        else:
            # R4 fix: an explicit tier_fraction (e.g. the oracle's n_star / N) is
            # honoured exactly, including 0 -- if the oracle found no beneficial
            # swaps for this config, the model/baselines should also make none,
            # rather than being forced to move at least one sequence anyway.
            n_tier = min(int(round(N * tier_fraction)), max_tier)
            if n_tier == 0:
                return l_rs_alloc

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
                # Distribute correction across neutral sequences (fractional bytes -> round)
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


# -- Oracle and uniform baselines --------------------------------------------

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


def oracle_allocation_greedy_swap(
    marginal_benefits: np.ndarray,
    marginal_harm:     np.ndarray,
    l_rs_default: int,
    delta: int,
    l_rs_min: int = 4,
    l_rs_max: int = 16,
    n_runs: int = 30,
    z_threshold: float = 1.0,
) -> Tuple[np.ndarray, int]:
    """True oracle (R4 fix): budget-neutral swaps, made only while net-beneficial
    AND the benefit clears a noise floor derived from the estimation sample size.

    oracle_allocation_marginal (above) had a real bug: it ranked ALL sequences by
    a single metric (benefit from adding delta) and forced exactly N//2 promotions
    and N//2 demotions, regardless of how many sequences actually had a positive
    marginal benefit. A sequence with marginal_benefit near zero could mean either
    "already safe -- extra parity doesn't matter" (fine to demote) or "already
    deep in the saturated/over-failure regime -- extra parity can't save it either"
    (actively harmful to demote further) -- the old code could not tell these
    apart, and empirically this caused the oracle to lose to uniform allocation
    in 66/84 production configs, since it was forced to demote already-failing
    sequences just to hit the fixed 50% tier size.

    This version uses two independent per-sequence signals -- benefit from
    promotion and harm from demotion (see estimate_marginal_harm in
    experiment.py) -- and greedily pairs the best promotion candidate with the
    least-harmed demotion candidate, one swap at a time. Requiring only
    benefit > harm is NOT enough in practice: marginal_benefits/marginal_harm
    are each estimated from only n_runs simulator runs, so individual
    per-sequence estimates carry substantial sampling noise (empirically,
    std(marginal_benefit) ~ 0.10 against a true population mean near 0 --
    i.e. most of the apparent per-sequence variation is noise, not signal).
    Greedily selecting the noisiest-looking "best" and "safest" sequences is a
    textbook winner's-curse: it looks great evaluated on the same noisy
    estimates used to select it, then regresses on a fresh evaluation sample
    -- confirmed empirically (oracle still lost to uniform on real production
    data even after the benefit>harm fix alone). So a swap is now only made
    if its margin (benefit - harm) clears an analytically-derived noise floor:
    for two independent binomial-proportion estimates from n_runs trials each,
    Var(p) <= 0.25/n_runs, so Var(benefit_i - harm_j) <= 4 * 0.25/n_runs (four
    independent proportion terms: p_default_i, p_high_i, p_low_j, p_default_j),
    giving SE <= 1/sqrt(n_runs). Empirically swept z in {0..4} across several
    production configs (see allocation experiment notes): z=2.0 fully
    eliminates residual losses in no-signal configs but also suppresses real,
    fresh-sample-confirmed wins in signal-rich configs (e.g. a genuine 3.4%
    OFR reduction collapsed to a wash). z=1.0 was chosen as the default
    instead -- it recovers most of the real wins while bounding the residual
    loss in no-signal configs to a small, explicitly reportable magnitude
    (observed: <=2.7% relative, versus 3-6% under the original unfixed
    oracle). This is a deliberate trade-off, not a guarantee: at z=1.0 the
    oracle can still show a small loss to uniform in configs with
    insufficient true signal at n_runs=30 -- that should be reported
    transparently as a limitation, not hidden by pushing z higher.

    Because a swap is only made once it clears both the net-benefit AND the
    noise-floor bar, large systematic losses (the original bug: 66/84 configs,
    often 3-6%) are structurally impossible; only small, bounded ones can
    remain, and only in genuinely low-signal configs.

    Parameters
    ----------
    marginal_benefits : array (N,) of p_i(l_rs_default) - p_i(l_rs_default + delta).
                         Positive = sequence benefits from promotion.
    marginal_harm      : array (N,) of p_i(l_rs_default - delta) - p_i(l_rs_default).
                         Positive = sequence is hurt by demotion (the common case,
                         by RS decode-failure monotonicity in parity bytes).
    n_runs             : number of simulator runs each per-sequence probability
                         estimate is based on -- used to size the noise floor.
    z_threshold        : how many standard errors of margin to require before
                         trusting a swap. Higher = more conservative (fewer,
                         more confident swaps); 0.0 recovers the benefit>harm-
                         only behaviour (not recommended -- see above).

    Returns
    -------
    l_rs_alloc : budget-neutral parity allocation, shape (N,).
    n_star     : number of promote/demote swaps actually made. Pass
                 tier_fraction = n_star / N to AllocationMechanism.allocate()
                 for the model and rule-based baselines, so every condition is
                 compared using the same oracle-determined budget size and
                 differs only in which sequences it picks to fill it.
    """
    N = len(marginal_benefits)
    l_rs_alloc = np.full(N, l_rs_default, dtype=int)
    if N == 0 or delta == 0:
        return l_rs_alloc, 0
    if (l_rs_default + delta) > l_rs_max or (l_rs_default - delta) < l_rs_min:
        return l_rs_alloc, 0

    margin_threshold = z_threshold / np.sqrt(max(n_runs, 1))

    promote_order = np.argsort(-marginal_benefits)  # best promotion candidates first
    demote_order  = np.argsort(marginal_harm)        # least-harmed demotion candidates first

    used      = np.zeros(N, dtype=bool)
    max_swaps = N // 2
    n_star    = 0
    pi = di = 0

    while n_star < max_swaps:
        while pi < N and used[promote_order[pi]]:
            pi += 1
        while di < N and used[demote_order[di]]:
            di += 1
        if pi >= N or di >= N:
            break
        p_idx, d_idx = promote_order[pi], demote_order[di]
        if p_idx == d_idx:
            pi += 1  # same sequence can't fill both roles -- try the next candidate
            continue
        if marginal_benefits[p_idx] - marginal_harm[d_idx] <= margin_threshold:
            break  # best remaining swap doesn't clear the noise floor -- stop
        l_rs_alloc[p_idx] = l_rs_default + delta
        l_rs_alloc[d_idx] = l_rs_default - delta
        used[p_idx] = used[d_idx] = True
        n_star += 1
        pi += 1
        di += 1

    return l_rs_alloc, n_star


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
