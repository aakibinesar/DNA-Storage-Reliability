"""
models/calibrate.py
===================
Post-hoc calibration wrappers for trained models.

Implements:
  - Platt scaling    : logistic regression on model outputs (primary for XGBoost)
  - Isotonic regression : non-parametric monotone calibration (primary for RF)
  - Temperature scaling : single-parameter logit rescaling (diagnostic)

The CalibratedModel class wraps a base estimator so that:
  1. calibrated.predict_proba(X) returns calibrated probabilities
  2. calibrated.base_model exposes the raw uncalibrated model
  3. calibrated.temperature (for temperature scaling) is accessible

Based on Guo et al. (2017) "On Calibration of Modern Neural Networks"
and Platt (1999) "Probabilistic Outputs for Support Vector Machines."
"""

import numpy as np
from typing import Literal, Optional


class CalibratedModel:
    """Wraps an sklearn-compatible classifier with a post-hoc calibration layer.

    Parameters
    ----------
    base_model : fitted sklearn-style classifier (has predict_proba method)
    method     : 'platt', 'isotonic', or 'temperature'
    seed       : random seed for reproducibility
    """

    def __init__(
        self,
        base_model,
        method: Literal['platt', 'isotonic', 'temperature', 'none'] = 'platt',
        seed: int = 42,
    ):
        self.base_model   = base_model
        self.method       = method
        self.seed         = seed
        self._calibrator  = None
        self.temperature  = 1.0        # only used for temperature scaling
        self._fitted      = False

    def fit(self, X_cal: np.ndarray, y_cal: np.ndarray) -> 'CalibratedModel':
        """Fit the calibration layer on a held-out calibration set.

        Parameters
        ----------
        X_cal : calibration features (N, d)
        y_cal : binary calibration labels (N,)
        """
        if self.method == 'none':
            self._fitted = True
            return self

        raw_proba = self._raw_proba(X_cal)

        if self.method == 'platt':
            self._calibrator = _PlattScaler()
            self._calibrator.fit(raw_proba, y_cal)

        elif self.method == 'isotonic':
            from sklearn.isotonic import IsotonicRegression  # type: ignore
            self._calibrator = IsotonicRegression(out_of_bounds='clip')
            self._calibrator.fit(raw_proba, y_cal)

        elif self.method == 'temperature':
            self.temperature = _fit_temperature(raw_proba, y_cal)

        elif self.method == 'none':
            pass   # no calibration needed

        self._fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return calibrated probability estimates for the positive class.

        Returns
        -------
        proba : 1D array of shape (N,) — P(failure)
        """
        raw = self._raw_proba(X)

        if not self._fitted or self.method == 'none':
            return raw

        if self.method in ('platt', 'isotonic'):
            return np.clip(self._calibrator.predict(raw), 0.0, 1.0)

        if self.method == 'temperature':
            logits = np.log(np.clip(raw, 1e-10, 1 - 1e-10)) - \
                     np.log(np.clip(1 - raw, 1e-10, 1 - 1e-10))
            scaled = logits / self.temperature
            return 1.0 / (1.0 + np.exp(-scaled))

        return raw

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return binary predictions using calibrated probabilities."""
        return (self.predict_proba(X) >= threshold).astype(int)

    def _raw_proba(self, X: np.ndarray) -> np.ndarray:
        """Get raw (uncalibrated) probabilities from the base model."""
        proba = self.base_model.predict_proba(X)
        if proba.ndim == 2:
            if proba.shape[1] >= 2:
                return proba[:, 1]
            return np.zeros(proba.shape[0])  # single-class model; failure prob = 0
        return proba


# ── Internal calibration components ─────────────────────────────────────────

class _PlattScaler:
    """Logistic regression on sigmoid-transformed model scores."""

    def __init__(self):
        self._lr = None

    def fit(self, proba: np.ndarray, y: np.ndarray):
        from sklearn.linear_model import LogisticRegression  # type: ignore
        if len(np.unique(y)) < 2:
            self._lr = None  # single-class cal set; skip calibration
            return
        logits = np.log(np.clip(proba, 1e-10, 1 - 1e-10)) - \
                 np.log(np.clip(1 - proba, 1e-10, 1 - 1e-10))
        self._lr = LogisticRegression(max_iter=1000, C=1e10)
        self._lr.fit(logits.reshape(-1, 1), y)

    def predict(self, proba: np.ndarray) -> np.ndarray:
        if self._lr is None:
            return proba  # no calibration applied; return raw proba
        logits = np.log(np.clip(proba, 1e-10, 1 - 1e-10)) - \
                 np.log(np.clip(1 - proba, 1e-10, 1 - 1e-10))
        return self._lr.predict_proba(logits.reshape(-1, 1))[:, 1]


def _fit_temperature(proba: np.ndarray, y: np.ndarray) -> float:
    """Find the temperature T that minimises NLL on the calibration set."""
    from scipy.optimize import minimize_scalar  # type: ignore

    def nll(T):
        T = max(T, 0.01)
        logits = np.log(np.clip(proba, 1e-10, 1 - 1e-10)) - \
                 np.log(np.clip(1 - proba, 1e-10, 1 - 1e-10))
        scaled = logits / T
        p = 1.0 / (1.0 + np.exp(-scaled))
        p = np.clip(p, 1e-10, 1 - 1e-10)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    result = minimize_scalar(nll, bounds=(0.1, 10.0), method='bounded')
    return float(result.x)


# ── Standalone calibration utilities ─────────────────────────────────────────

def calibration_curve_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """Compute reliability diagram data points.

    Returns
    -------
    dict with 'bin_means', 'bin_fracs', 'bin_counts' arrays
    """
    bins       = np.linspace(0, 1, n_bins + 1)
    bin_idx    = np.digitize(y_prob, bins) - 1
    bin_idx    = np.clip(bin_idx, 0, n_bins - 1)

    bin_means  = np.zeros(n_bins)
    bin_fracs  = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)

    for b in range(n_bins):
        mask = bin_idx == b
        if mask.sum() > 0:
            bin_means[b]  = y_prob[mask].mean()
            bin_fracs[b]  = y_true[mask].mean()
            bin_counts[b] = mask.sum()

    return {
        'bin_means'  : bin_means,
        'bin_fracs'  : bin_fracs,
        'bin_counts' : bin_counts,
        'n_bins'     : n_bins,
    }


def compare_calibration(
    y_true:   np.ndarray,
    raw_proba: np.ndarray,
    cal_proba: np.ndarray,
    n_bins:    int = 10,
) -> dict:
    """Compare raw vs. calibrated ECE and Brier scores."""
    from evaluate import expected_calibration_error, brier_score_decomposed

    return {
        'ece_raw'       : expected_calibration_error(y_true, raw_proba, n_bins),
        'ece_calibrated': expected_calibration_error(y_true, cal_proba, n_bins),
        'brier_raw'     : brier_score_decomposed(y_true, raw_proba),
        'brier_cal'     : brier_score_decomposed(y_true, cal_proba),
    }
