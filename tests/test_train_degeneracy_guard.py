"""
Regression test for the model-generation bug fixed in models/train.py:

Before the fix, the single-class training guard for XGBoost/Random Forest
checked a crude binarised (>=0.5) cutoff on the target instead of the
continuous failure_freq's actual variation. Any config where every sequence
happened to land on one side of that cutoff got a trivial constant
DummyRegressor in place of a real model -- discarding genuine, learnable
continuous signal. This silently hit 15 of 28 real configs.

These tests reconstruct both halves of that bug directly:
  1. A target that is entirely below the 0.5 binarisation cutoff (so the old
     buggy guard would have fired) but has real continuous variance must
     still produce a REAL XGBoost/RF model, not a dummy.
  2. A target with no real variation at all must still correctly fall back
     to a dummy (that part of the guard was never wrong; it's the negative
     control).
"""
import numpy as np
from sklearn.dummy import DummyRegressor

from train import train_all_models


def _synthetic_features(n, d, seed):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, d))


def test_single_sided_but_continuous_target_trains_real_models(fast_cfg):
    """All labels land below the 0.5 binarisation cutoff, but the continuous
    target still has real, learnable variance -- must NOT fall back to a
    dummy regressor for XGBoost/RF (the exact bug that hit 15/28 configs)."""
    n, d = 120, 6
    X = _synthetic_features(n, d, seed=1)
    # y correlates with a feature and stays entirely in [0.05, 0.45] --
    # every sequence binarises to label 0, but there's real variance to fit.
    rng = np.random.default_rng(2)
    y = np.clip(0.25 + 0.15 * X[:, 0] + rng.normal(scale=0.02, size=n), 0.0, 0.45)
    assert (y < 0.5).all(), "test setup: target must be entirely below the binarisation cutoff"
    assert (np.max(y) - np.min(y)) > 1e-3, "test setup: target must have real continuous variance"

    X_tr, X_val, X_cal = X[:60], X[60:90], X[90:]
    y_tr,  y_val, y_cal = y[:60], y[60:90], y[90:]

    results = train_all_models(X_tr, y_tr, X_val, y_val, X_cal, y_cal,
                                fast_cfg, seed=42, verbose=False)

    assert not isinstance(results['xgboost'].base_model, DummyRegressor), (
        "XGBoost fell back to a dummy model despite real continuous target "
        "variance -- the degeneracy guard regressed to the old binary-cutoff bug."
    )
    assert not isinstance(results['random_forest'].base_model, DummyRegressor), (
        "Random Forest fell back to a dummy model despite real continuous target "
        "variance -- the degeneracy guard regressed to the old binary-cutoff bug."
    )
    # Real models should discriminate somewhat rather than collapse to one value.
    preds = results['xgboost'].predict_proba(X_val)
    assert np.std(preds) > 1e-4, "XGBoost predictions are (near-)constant on real data"


def test_truly_constant_target_still_uses_dummy(fast_cfg):
    """Negative control: a target with essentially no variation at all must
    still correctly fall back to a constant DummyRegressor."""
    n, d = 80, 6
    X = _synthetic_features(n, d, seed=3)
    y = np.full(n, 0.42)

    X_tr, X_val, X_cal = X[:40], X[40:60], X[60:]
    y_tr,  y_val, y_cal = y[:40], y[40:60], y[60:]

    results = train_all_models(X_tr, y_tr, X_val, y_val, X_cal, y_cal,
                                fast_cfg, seed=42, verbose=False)

    assert isinstance(results['xgboost'].base_model, DummyRegressor)
    assert isinstance(results['random_forest'].base_model, DummyRegressor)
    # The dummy must predict the actual constant, not an arbitrary default.
    preds = results['xgboost'].predict_proba(X_val)
    assert np.allclose(preds, 0.42, atol=1e-6)
