"""Tests for bootstrap-refit explanation stability."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from xai_framework.xai.stability import run_stability_analysis


def _data():
    """Create a deterministic synthetic binary train/test partition.

    Returns:
        Training matrix, test matrix, training labels, and test labels.
    """
    rng = np.random.default_rng(7)
    X = pd.DataFrame(rng.normal(size=(80, 4)), columns=list("abcd"))
    y = pd.Series((1.8 * X["a"] - 0.8 * X["b"] + rng.normal(size=80) > 0).astype(int))
    return X.iloc[:60], X.iloc[60:], y.iloc[:60], y.iloc[60:]


def _fit(X_bootstrap, y_bootstrap):
    """Fit the deterministic logistic specification used in stability tests.

    Args:
        X_bootstrap: Bootstrapped training matrix.
        y_bootstrap: Bootstrapped binary outcomes.

    Returns:
        Fitted logistic-regression estimator.
    """
    return LogisticRegression(max_iter=1000, random_state=42).fit(
        X_bootstrap, y_bootstrap
    )


def test_bootstrap_stability_is_reproducible_and_well_formed():
    """Verify seeded stability outputs are reproducible and structurally valid."""
    X_train, X_test, y_train, y_test = _data()
    kwargs = dict(
        train_model_fn=_fit,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=X_train.columns,
        n_bootstrap=5,
        bootstrap_seed=42,
        scoring="neg_log_loss",
        n_repeats=2,
        permutation_seed=42,
        top_k=2,
    )
    first = run_stability_analysis(**kwargs)
    second = run_stability_analysis(**kwargs)
    for first_frame, second_frame in zip(first, second):
        pd.testing.assert_frame_equal(first_frame, second_frame)
    raw, summary, pairwise = first
    assert len(raw) == 5 * X_train.shape[1]
    assert raw["Bootstrap"].nunique() == 5
    assert len(pairwise) == 10
    assert summary["TopKFrequency"].between(0, 1).all()
    assert pairwise["TopKOverlap"].between(0, 1).all()


def test_bootstrap_refits_change_deterministic_logistic_models():
    """Verify resampled training composition changes fitted importance vectors."""
    X_train, X_test, y_train, y_test = _data()
    raw, _, _ = run_stability_analysis(
        _fit,
        X_train,
        X_test,
        y_train,
        y_test,
        X_train.columns,
        n_bootstrap=6,
        n_repeats=2,
    )
    importance_vectors = raw.pivot(
        index="Bootstrap", columns="Feature", values="Importance"
    )
    assert len(importance_vectors.drop_duplicates()) > 1


def test_stability_requires_multiple_bootstrap_refits():
    """Verify pairwise stability analysis rejects a single requested refit."""
    X_train, X_test, y_train, y_test = _data()
    with pytest.raises(ValueError, match="at least 2"):
        run_stability_analysis(
            _fit,
            X_train,
            X_test,
            y_train,
            y_test,
            X_train.columns,
            n_bootstrap=1,
        )
