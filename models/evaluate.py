"""
models/evaluate.py
==================
All evaluation metrics for Layer 1 (predictive quality) and Layer 2 (allocation
quality) evaluation described in the implementation plan.

Layer 1 metrics (Table 1):
  - Accuracy, Precision, Recall, F1
  - AUROC, PR-AUC (AUPRC)
  - Brier score (with reliability/refinement decomposition)
  - Expected Calibration Error (ECE)

Layer 2 metrics (Table 2):
  - Failure Reduction Rate (FRR) = post-allocation failure rate
  - Efficiency ratio FRR_model / FRR_oracle
  - Paired t-test p-value and Cohen's d

Calibration analysis:
  - Reliability diagram data
  - Brier score decomposition into reliability + resolution + uncertainty
"""

import sys
import os
import numpy as np
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


# ── Calibration metrics ──────────────────────────────────────────────────────

def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (ECE).

    ECE = sum_b (|B_b| / N) * |acc(B_b) - conf(B_b)|

    The key evaluation metric per the implementation plan — required to be < 0.05
    at the Week 4 go/no-go gate.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    N      = len(y_true)
    if N == 0:
        return 0.0

    bins    = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)
    ece     = 0.0

    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        acc  = y_true[mask].mean()    # empirical fraction positive
        conf = y_prob[mask].mean()    # mean predicted confidence
        ece += (mask.sum() / N) * abs(acc - conf)

    return float(ece)


def brier_score_decomposed(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, float]:
    """Brier score with Murphy (1973) reliability-resolution-uncertainty decomposition.

    Brier = REL - RES + UNC

    REL (reliability)  : penalises miscalibration; lower is better
    RES (resolution)   : reward for deviating from climatology; higher is better
    UNC (uncertainty)  : irreducible noise in the binary outcome

    Returns dict with keys: 'brier', 'reliability', 'resolution', 'uncertainty'
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    N      = len(y_true)

    brier       = float(np.mean((y_true - y_prob) ** 2))
    base_rate   = float(y_true.mean())
    uncertainty = float(base_rate * (1.0 - base_rate))

    bins    = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)
    reliability = 0.0
    resolution  = 0.0

    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() == 0:
            continue
        n_b   = mask.sum()
        o_b   = y_true[mask].mean()
        f_b   = y_prob[mask].mean()
        reliability += (n_b / N) * (f_b - o_b) ** 2
        resolution  += (n_b / N) * (o_b - base_rate) ** 2

    return {
        'brier'      : brier,
        'reliability': float(reliability),
        'resolution' : float(resolution),
        'uncertainty': uncertainty,
    }


# ── Classification metrics ───────────────────────────────────────────────────

def classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute accuracy, precision, recall, F1, AUROC, PR-AUC."""
    from sklearn.metrics import (  # type: ignore
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score,
    )

    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    y_pred = (y_prob >= threshold).astype(int)

    # Handle degenerate case (only one class)
    if len(np.unique(y_true)) == 1:
        auroc  = 0.5
        prauc  = float(y_true.mean())
    else:
        auroc = float(roc_auc_score(y_true, y_prob))
        prauc = float(average_precision_score(y_true, y_prob))

    return {
        'accuracy'  : float(accuracy_score(y_true, y_pred)),
        'precision' : float(precision_score(y_true, y_pred, zero_division=0)),
        'recall'    : float(recall_score(y_true, y_pred, zero_division=0)),
        'f1'        : float(f1_score(y_true, y_pred, zero_division=0)),
        'auroc'     : auroc,
        'pr_auc'    : prauc,
    }


def full_evaluation(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    threshold: float = 0.5,
    label: str = '',
) -> Dict[str, float]:
    """Run all Layer 1 metrics and return a flat results dict."""
    clf_metrics = classification_metrics(y_true, y_prob, threshold)
    brier_d     = brier_score_decomposed(y_true, y_prob, n_bins)
    ece         = expected_calibration_error(y_true, y_prob, n_bins)

    results = {**clf_metrics, **brier_d, 'ece': ece}
    if label:
        results = {f'{label}/{k}': v for k, v in results.items()}
    return results


# ── Layer 2 allocation metrics ───────────────────────────────────────────────

def failure_reduction_rate(
    failure_flags: np.ndarray,
) -> float:
    """FRR = mean post-allocation failure indicator.

    Parameters
    ----------
    failure_flags : boolean/int array (N,) — 1 if sequence failed after allocation
    """
    return float(np.mean(failure_flags))


def efficiency_ratio(frr_model: float, frr_oracle: float, frr_uniform: float) -> float:
    """FRR efficiency ratio = fraction of theoretically available gain captured.

    efficiency = (FRR_uniform - FRR_model) / (FRR_uniform - FRR_oracle)

    1.0 = model achieves oracle performance
    0.0 = model offers no improvement over uniform allocation
    """
    denom = frr_uniform - frr_oracle
    if abs(denom) < 1e-10:
        return 1.0
    return float((frr_uniform - frr_model) / denom)


def allocation_statistics(
    frr_uniform_runs:    np.ndarray,
    frr_model_runs:      np.ndarray,
    frr_oracle_runs:     np.ndarray,
) -> Dict[str, float]:
    """Aggregate FRR statistics across M independent runs (paired t-test).

    Parameters
    ----------
    frr_*_runs : arrays of shape (M,) — one value per simulator run

    Returns
    -------
    Dict with means, std, t-statistic, p-value, Cohen's d, efficiency ratio
    """
    from scipy import stats  # type: ignore

    t_stat, p_val = stats.ttest_rel(frr_uniform_runs, frr_model_runs)
    d_val = float(np.mean(frr_uniform_runs - frr_model_runs) /
                  (np.std(frr_uniform_runs - frr_model_runs) + 1e-12))

    frr_u = float(frr_uniform_runs.mean())
    frr_m = float(frr_model_runs.mean())
    frr_o = float(frr_oracle_runs.mean())

    return {
        'frr_uniform_mean'  : frr_u,
        'frr_uniform_std'   : float(frr_uniform_runs.std()),
        'frr_model_mean'    : frr_m,
        'frr_model_std'     : float(frr_model_runs.std()),
        'frr_oracle_mean'   : frr_o,
        'frr_oracle_std'    : float(frr_oracle_runs.std()),
        't_statistic'       : float(t_stat),
        'p_value'           : float(p_val),
        'cohens_d'          : d_val,
        'efficiency_ratio'  : efficiency_ratio(frr_m, frr_o, frr_u),
        'significant_p005'  : bool(p_val < 0.05),
    }


# ── Print helpers ────────────────────────────────────────────────────────────

def print_evaluation_table(results: Dict[str, Dict[str, float]]):
    """Print a formatted evaluation table for multiple models."""
    models = list(results.keys())
    metrics = list(next(iter(results.values())).keys())

    header = f"{'Metric':<25}" + ''.join(f"{m:>15}" for m in models)
    print(header)
    print('-' * len(header))
    for metric in metrics:
        row = f"{metric:<25}" + ''.join(
            f"{results[m].get(metric, float('nan')):>15.4f}" for m in models
        )
        print(row)


def go_no_go_check(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Tuple[bool, str]:
    """Week 4 go/no-go gate: ECE < 0.05 AND Brier below no-skill baseline.

    No-skill baseline Brier = base_rate * (1 - base_rate).
    """
    ece    = expected_calibration_error(y_true, y_prob, n_bins)
    brier  = brier_score_decomposed(y_true, y_prob)['brier']
    base   = float(np.asarray(y_true).mean())
    no_skill_brier = base * (1 - base)

    ece_pass   = ece   < 0.05
    brier_pass = brier < no_skill_brier

    msg = (
        f"  ECE={ece:.4f} ({'PASS' if ece_pass else 'FAIL'} threshold=0.05)\n"
        f"  Brier={brier:.4f} ({'PASS' if brier_pass else 'FAIL'} no-skill={no_skill_brier:.4f})"
    )
    return ece_pass and brier_pass, msg
