"""
Regression test for the calibration bug fixed in models/calibrate.py:

Before the fix, CalibratedModel._raw_proba always returned 0.0 for any
single-column predict_proba output (the case a degenerate DummyClassifier
produces), regardless of which class that column actually represented. For
the 4 near-100%-failure configs whose Logistic Regression component
degenerated to an all-positive DummyClassifier, this silently reported 0%
failure probability for sequences that fail almost every time -- the
opposite of reality.

The fix inspects base_model.classes_ to return the correct constant (1.0 if
the sole class is 1, 0.0 if it is 0) instead of hard-coding zero.
"""
import numpy as np
from sklearn.dummy import DummyClassifier

from calibrate import CalibratedModel


def test_degenerate_all_positive_dummy_returns_one_not_zero():
    """The exact bug: an all-positive DummyClassifier must report P(failure)=1,
    not the old hard-coded 0."""
    X = np.random.default_rng(0).normal(size=(20, 3))
    y_bin = np.ones(20, dtype=int)  # every calibration example is class 1

    dummy = DummyClassifier(strategy='most_frequent').fit(X, y_bin)
    assert list(dummy.classes_) == [1], "test setup: dummy must be single-class positive"

    model = CalibratedModel(dummy, method='none')
    model.fit(X, y_bin.astype(float))
    proba = model.predict_proba(X)

    assert np.allclose(proba, 1.0), (
        f"Expected P(failure)=1.0 for an all-positive degenerate classifier, "
        f"got {proba[:3]} -- regression of the calibration-inversion bug."
    )


def test_degenerate_all_negative_dummy_returns_zero():
    """Symmetric case: an all-negative DummyClassifier must report P(failure)=0."""
    X = np.random.default_rng(1).normal(size=(20, 3))
    y_bin = np.zeros(20, dtype=int)

    dummy = DummyClassifier(strategy='most_frequent').fit(X, y_bin)
    assert list(dummy.classes_) == [0], "test setup: dummy must be single-class negative"

    model = CalibratedModel(dummy, method='none')
    model.fit(X, y_bin.astype(float))
    proba = model.predict_proba(X)

    assert np.allclose(proba, 0.0)


def test_non_degenerate_classifier_still_uses_both_columns():
    """Sanity check that the fix didn't break the normal two-class path."""
    rng = np.random.default_rng(2)
    X = rng.normal(size=(60, 3))
    y_bin = (X[:, 0] > 0).astype(int)
    assert len(np.unique(y_bin)) == 2

    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression().fit(X, y_bin)

    model = CalibratedModel(clf, method='none')
    model.fit(X, y_bin.astype(float))
    proba = model.predict_proba(X)

    assert proba.min() >= 0.0 and proba.max() <= 1.0
    assert np.std(proba) > 1e-3, "expected real discrimination on separable synthetic data"
