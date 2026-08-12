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
  - Oligo Failure Rate (OFR) = fraction of oligos that fail RS decoding after allocation
  - OFR reduction = (OFR_uniform − OFR_method) / OFR_uniform
  - Efficiency ratio = (OFR_uniform − OFR_model) / (OFR_uniform − OFR_oracle)
  - Paired t-test p-value and Cohen's d

Terminology note (R5):
  The quantity failures/N is an oligo failure *rate*, not a failure *reduction* rate.
  FRR was a misnomer in earlier versions; all public names now use OFR.

Calibration analysis:
  - Reliability diagram data
  - Brier score decomposition into reliability + resolution + uncertainty
  - R9: bootstrap CIs for ECE/Brier and regime-stratified calibration
    (aggregate ECE is deflated by under-failure and saturated regimes where
    calibration is trivially easy; the informative regime shows the real picture)
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


# ── R9: bootstrap CIs + regime-stratified calibration ───────────────────────

def bootstrap_calibration_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn,
    n_bootstrap: int = 500,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float, float]:
    """Bootstrap (1−alpha) confidence interval for a scalar calibration metric.

    Parameters
    ----------
    y_true, y_prob : arrays (N,) — targets and predictions
    metric_fn      : callable(y_true, y_prob) → float
    n_bootstrap    : bootstrap replicates; set 0 to skip (returns NaN CI)
    alpha          : significance level (0.05 → 95% CI)

    Returns
    -------
    (point_estimate, ci_lo, ci_hi)  — NaN bounds when n < 5 or n_bootstrap == 0
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    n = len(y_true)

    point = float(metric_fn(y_true, y_prob)) if n > 0 else float('nan')
    if n < 5 or n_bootstrap == 0:
        return point, float('nan'), float('nan')

    rng  = np.random.default_rng(seed)
    boot = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot[i] = float(metric_fn(y_true[idx], y_prob[idx]))

    ci_lo = float(np.percentile(boot, 100.0 * alpha / 2.0))
    ci_hi = float(np.percentile(boot, 100.0 * (1.0 - alpha / 2.0)))
    return point, ci_lo, ci_hi


def calibration_by_regime(
    failure_freq: np.ndarray,
    y_prob: np.ndarray,
    regime_thresholds: Tuple[float, float] = (0.15, 0.85),
    n_bootstrap: int = 500,
    alpha: float = 0.05,
    n_bins: int = 10,
    seed: int = 0,
) -> List[Dict]:
    """ECE, Brier, and class prevalence stratified by failure_freq regime.

    R9 mitigation: aggregate ECE is deflated by the under-failure and saturated
    regimes where calibration is trivially good (predictions ≈ 0 vs labels ≈ 0,
    or ≈ 1 vs ≈ 1).  Stratifying exposes calibration quality in the informative
    regime where discrimination actually matters.

    Regime definitions (default thresholds 0.15 / 0.85):
      under_failure — failure_freq < lo   : no real failures; any near-zero pred is "calibrated"
      informative   — lo ≤ ff ≤ hi       : model must genuinely discriminate
      saturated     — failure_freq > hi  : near-certain failure; symmetric to under_failure
      aggregate     — all sequences       : reported for direct comparison with per-regime values

    Parameters
    ----------
    failure_freq       : continuous empirical failure probability (N,) — used as
                         both regime assignment key and calibration target
    y_prob             : model predicted probabilities (N,)
    regime_thresholds  : (lo, hi) boundaries; sequences at lo/hi go to informative
    n_bootstrap        : bootstrap replicates for CI (0 → skip, returns NaN CIs)
    alpha              : CI significance level
    seed               : base RNG seed for bootstrap

    Returns
    -------
    List of 4 dicts (under_failure, informative, saturated, aggregate), each with:
        regime, n, frac_total, class_prevalence, mean_ff,
        ece, ece_ci_lo, ece_ci_hi,
        brier, brier_ci_lo, brier_ci_hi
    """
    lo, hi = regime_thresholds
    ff  = np.clip(np.asarray(failure_freq, dtype=float), 0.0, 1.0)
    yp  = np.clip(np.asarray(y_prob,       dtype=float), 0.0, 1.0)
    N   = len(ff)

    masks = {
        'under_failure': ff < lo,
        'informative'  : (ff >= lo) & (ff <= hi),
        'saturated'    : ff > hi,
        'aggregate'    : np.ones(N, dtype=bool),
    }

    def _ece(yt, yp_):
        return expected_calibration_error(yt, yp_, n_bins)

    def _brier(yt, yp_):
        return float(np.mean((np.asarray(yt, float) - np.asarray(yp_, float)) ** 2))

    rows = []
    boot_seed = seed
    for regime_name, mask in masks.items():
        n = int(mask.sum())
        if n == 0:
            rows.append({
                'regime': regime_name, 'n': 0, 'frac_total': 0.0,
                'class_prevalence': float('nan'), 'mean_ff': float('nan'),
                'ece': float('nan'), 'ece_ci_lo': float('nan'), 'ece_ci_hi': float('nan'),
                'brier': float('nan'), 'brier_ci_lo': float('nan'), 'brier_ci_hi': float('nan'),
            })
            continue

        ff_r = ff[mask]
        yp_r = yp[mask]

        ece_pt, ece_lo, ece_hi = bootstrap_calibration_ci(
            ff_r, yp_r, _ece, n_bootstrap=n_bootstrap, alpha=alpha, seed=boot_seed
        )
        brier_pt, brier_lo, brier_hi = bootstrap_calibration_ci(
            ff_r, yp_r, _brier, n_bootstrap=n_bootstrap, alpha=alpha, seed=boot_seed + 1
        )
        boot_seed += 10

        rows.append({
            'regime'          : regime_name,
            'n'               : n,
            'frac_total'      : float(n / N),
            'class_prevalence': float((ff_r >= 0.5).mean()),
            'mean_ff'         : float(ff_r.mean()),
            'ece'             : ece_pt,
            'ece_ci_lo'       : ece_lo,
            'ece_ci_hi'       : ece_hi,
            'brier'           : brier_pt,
            'brier_ci_lo'     : brier_lo,
            'brier_ci_hi'     : brier_hi,
        })

    return rows


def stratified_evaluation(
    failure_freq: np.ndarray,
    y_prob: np.ndarray,
    regime_thresholds: Tuple[float, float] = (0.15, 0.85),
    n_bins: int = 10,
) -> List[Dict]:
    """All Layer 1 metrics stratified by failure_freq regime.

    R11 mitigation: aggregate AUROC, F1, ECE and Brier are inflated by the
    under-failure and saturated regimes where prediction is trivially easy.
    Stratifying exposes genuine model quality in the informative regime.

    Regime definitions (default 0.15 / 0.85 — stored in cfg['evaluation']):
      under_failure — failure_freq < lo   : few/no failures; near-0 predictions correct by default
      informative   — lo ≤ ff ≤ hi       : genuine discrimination required
      saturated     — failure_freq > hi   : near-certain failure; symmetric to under_failure
      aggregate     — all sequences       : for direct comparison with per-regime values

    Binary labels within each regime are derived from failure_freq >= 0.5.
    AUROC/PR-AUC are set to NaN for degenerate regimes (all-positive or all-negative),
    flagged via the is_degenerate column.

    Parameters
    ----------
    failure_freq       : continuous empirical failure probability (N,) — used
                         for regime assignment and as the calibration target
    y_prob             : model predicted probabilities (N,)
    regime_thresholds  : (lo, hi) boundaries; sequences at boundary go to informative
    n_bins             : ECE bin count

    Returns
    -------
    List of 4 dicts (under_failure, informative, saturated, aggregate), each with:
        regime, n, frac_total, class_prevalence, is_degenerate,
        auroc, pr_auc, f1, precision, recall, accuracy,
        ece, brier
    """
    try:
        from sklearn.metrics import (  # type: ignore
            roc_auc_score, average_precision_score,
            f1_score, precision_score, recall_score, accuracy_score,
        )
    except ImportError:
        raise ImportError("scikit-learn required for stratified_evaluation")

    lo, hi = regime_thresholds
    ff  = np.clip(np.asarray(failure_freq, dtype=float), 0.0, 1.0)
    yp  = np.clip(np.asarray(y_prob,       dtype=float), 0.0, 1.0)
    N   = len(ff)

    masks = {
        'under_failure': ff < lo,
        'informative'  : (ff >= lo) & (ff <= hi),
        'saturated'    : ff > hi,
        'aggregate'    : np.ones(N, dtype=bool),
    }

    rows = []
    for regime_name, mask in masks.items():
        n = int(mask.sum())
        if n == 0:
            rows.append({
                'regime': regime_name, 'n': 0, 'frac_total': 0.0,
                'class_prevalence': float('nan'), 'is_degenerate': True,
                'auroc': float('nan'), 'pr_auc': float('nan'),
                'f1': float('nan'), 'precision': float('nan'),
                'recall': float('nan'), 'accuracy': float('nan'),
                'ece': float('nan'), 'brier': float('nan'),
            })
            continue

        ff_r   = ff[mask]
        yp_r   = yp[mask]
        y_bin  = (ff_r >= 0.5).astype(int)
        y_pred = (yp_r >= 0.5).astype(int)
        n_pos  = int(y_bin.sum())
        n_neg  = n - n_pos
        is_deg = n_pos == 0 or n_neg == 0

        if is_deg:
            auroc = float('nan')
            prauc = float('nan')
        else:
            auroc = float(roc_auc_score(y_bin, yp_r))
            prauc = float(average_precision_score(y_bin, yp_r))

        rows.append({
            'regime'          : regime_name,
            'n'               : n,
            'frac_total'      : float(n / N),
            'class_prevalence': float(y_bin.mean()),
            'is_degenerate'   : is_deg,
            'auroc'           : auroc,
            'pr_auc'          : prauc,
            'f1'              : float(f1_score(y_bin, y_pred, zero_division=0)),
            'precision'       : float(precision_score(y_bin, y_pred, zero_division=0)),
            'recall'          : float(recall_score(y_bin, y_pred, zero_division=0)),
            'accuracy'        : float(accuracy_score(y_bin, y_pred)),
            'ece'             : float(expected_calibration_error(ff_r, yp_r, n_bins)),
            'brier'           : float(np.mean((ff_r - yp_r) ** 2)),
        })

    return rows


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

    # Binarise at 0.5 for classification metrics (AUROC, PR-AUC, F1).
    # y_true may be continuous failure_freq — dtype=int would truncate incorrectly.
    y_true = (np.asarray(y_true, dtype=float) >= 0.5).astype(int)
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


def threshold_sensitivity(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Optional[np.ndarray] = None,
) -> List[Dict[str, float]]:
    """Sweep the decision threshold and report classification metrics at each value.

    R8 mitigation: demonstrates that AUROC (threshold-free) is stable across the
    sweep, and identifies what threshold maximises F1 if 0.5 is suboptimal for the
    class balance of a given config.

    Parameters
    ----------
    y_true     : continuous failure_freq or binary labels (N,); thresholded at 0.5
    y_prob     : predicted probabilities (N,)
    thresholds : thresholds to evaluate; default np.arange(0.10, 0.91, 0.05)

    Returns
    -------
    List of dicts, each with keys: threshold, precision, recall, f1, accuracy,
    auroc, pr_auc, brier.  auroc and pr_auc are threshold-free — they repeat
    across rows but are included for convenient CSV export.
    """
    from sklearn.metrics import (  # type: ignore
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score,
    )

    if thresholds is None:
        thresholds = np.arange(0.10, 0.91, 0.05)

    y_true_bin = (np.asarray(y_true, dtype=float) >= 0.5).astype(int)
    y_prob_arr = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)

    if len(np.unique(y_true_bin)) == 1:
        auroc = 0.5
        prauc = float(y_true_bin.mean())
    else:
        auroc = float(roc_auc_score(y_true_bin, y_prob_arr))
        prauc = float(average_precision_score(y_true_bin, y_prob_arr))

    brier = float(np.mean((y_true_bin.astype(float) - y_prob_arr) ** 2))

    rows = []
    for t in thresholds:
        y_pred = (y_prob_arr >= t).astype(int)
        rows.append({
            'threshold' : float(t),
            'precision' : float(precision_score(y_true_bin, y_pred, zero_division=0)),
            'recall'    : float(recall_score(y_true_bin, y_pred, zero_division=0)),
            'f1'        : float(f1_score(y_true_bin, y_pred, zero_division=0)),
            'accuracy'  : float(accuracy_score(y_true_bin, y_pred)),
            'auroc'     : auroc,
            'pr_auc'    : prauc,
            'brier'     : brier,
        })
    return rows


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

def oligo_failure_rate(
    failure_flags: np.ndarray,
) -> float:
    """OFR = fraction of oligos that fail RS decoding after allocation.

    Parameters
    ----------
    failure_flags : boolean/int array (N,) — 1 if oligo failed after allocation
    """
    return float(np.mean(failure_flags))


# Backward-compatibility alias (was misnamed "failure_reduction_rate" in v1)
failure_reduction_rate = oligo_failure_rate


def ofr_reduction(ofr_uniform: float, ofr_method: float) -> float:
    """Relative OFR reduction = (OFR_uniform − OFR_method) / OFR_uniform.

    1.0  = method eliminated every failure that uniform would have had.
    0.0  = no improvement over uniform.
    <0   = method performs worse than uniform.
    """
    if abs(ofr_uniform) < 1e-10:
        return 0.0
    return float((ofr_uniform - ofr_method) / ofr_uniform)


def efficiency_ratio(ofr_model: float, ofr_oracle: float, ofr_uniform: float) -> float:
    """Efficiency ratio = fraction of theoretically available OFR gain captured.

    efficiency = (OFR_uniform − OFR_model) / (OFR_uniform − OFR_oracle)

    1.0 = model achieves oracle performance.
    0.0 = model offers no improvement over uniform allocation.
    """
    denom = ofr_uniform - ofr_oracle
    if abs(denom) < 1e-10:
        return 1.0
    return float((ofr_uniform - ofr_model) / denom)


def allocation_statistics(
    ofr_uniform_runs: np.ndarray,
    ofr_model_runs:   np.ndarray,
    ofr_oracle_runs:  np.ndarray,
) -> Dict[str, float]:
    """Aggregate OFR statistics across M independent runs (paired t-test).

    Parameters
    ----------
    ofr_*_runs : arrays of shape (M,) — OFR per simulator run

    Returns
    -------
    Dict with means, std, t-statistic, p-value, Cohen's d, OFR reduction,
    efficiency ratio.
    """
    from scipy import stats  # type: ignore

    t_stat, p_val = stats.ttest_rel(ofr_uniform_runs, ofr_model_runs)
    d_val = float(np.mean(ofr_uniform_runs - ofr_model_runs) /
                  (np.std(ofr_uniform_runs - ofr_model_runs) + 1e-12))

    ofr_u = float(ofr_uniform_runs.mean())
    ofr_m = float(ofr_model_runs.mean())
    ofr_o = float(ofr_oracle_runs.mean())

    return {
        'ofr_uniform_mean'  : ofr_u,
        'ofr_uniform_std'   : float(ofr_uniform_runs.std()),
        'ofr_model_mean'    : ofr_m,
        'ofr_model_std'     : float(ofr_model_runs.std()),
        'ofr_oracle_mean'   : ofr_o,
        'ofr_oracle_std'    : float(ofr_oracle_runs.std()),
        't_statistic'       : float(t_stat),
        'p_value'           : float(p_val),
        'cohens_d'          : d_val,
        'ofr_reduction'     : ofr_reduction(ofr_u, ofr_m),
        'efficiency_ratio'  : efficiency_ratio(ofr_m, ofr_o, ofr_u),
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
