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
        """Get raw (uncalibrated) probability estimates from the base model.

        Supports classifiers (predict_proba) and regressors (predict).
        """
        if hasattr(self.base_model, 'predict_proba'):
            proba = self.base_model.predict_proba(X)
            if proba.ndim == 2:
                if proba.shape[1] >= 2:
                    return proba[:, 1]
                return np.zeros(proba.shape[0])
            return proba
        # Regressor path: XGBRegressor / RandomForestRegressor output [0, 1] directly
        return np.clip(self.base_model.predict(X), 0.0, 1.0)


# -- Internal calibration components -----------------------------------------

class _PlattScaler:
    """Platt scaling via direct NLL minimisation — supports continuous targets in [0, 1].

    Fits A, B in sigma(A * logit(raw) + B) to minimise cross-entropy against y,
    where y may be a continuous empirical failure frequency (R1 compatible).
    Replaces the previous LogisticRegression approach, which rejected continuous y.
    """

    def __init__(self):
        self._A: float = 1.0
        self._B: float = 0.0
        self._fitted: bool = False

    def fit(self, proba: np.ndarray, y: np.ndarray):
        from scipy.optimize import minimize  # type: ignore

        y = np.asarray(y, dtype=float)
        if float(y.max() - y.min()) < 1e-8:
            self._fitted = False  # degenerate cal set; skip
            return

        logits = np.log(np.clip(proba, 1e-10, 1 - 1e-10)) - \
                 np.log(np.clip(1 - proba, 1e-10, 1 - 1e-10))

        def neg_ll(params):
            A, B = params
            p = 1.0 / (1.0 + np.exp(-(A * logits + B)))
            p = np.clip(p, 1e-10, 1 - 1e-10)
            return -float(np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

        result = minimize(neg_ll, [1.0, 0.0], method='L-BFGS-B',
                          options={'maxiter': 1000, 'ftol': 1e-12})
        self._A, self._B = float(result.x[0]), float(result.x[1])
        self._fitted = True

    def predict(self, proba: np.ndarray) -> np.ndarray:
        if not self._fitted:
            return proba
        logits = np.log(np.clip(proba, 1e-10, 1 - 1e-10)) - \
                 np.log(np.clip(1 - proba, 1e-10, 1 - 1e-10))
        return 1.0 / (1.0 + np.exp(-(self._A * logits + self._B)))


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


# -- Standalone calibration utilities -----------------------------------------

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
