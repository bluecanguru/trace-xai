"""Tests selection mechanics only, not validity of test-set model selection.

Use validation data for selection; reserve the final test data for evaluation.
This legacy selector is not required by the notebook-driven fitted-model API.
"""
import numpy as np

from models.selector import select_best_model


class Candidate:
    """Minimal candidate estimator with controllable ranking direction."""

    def __init__(self, reverse=False):
        """Initialise a candidate and its fit-call counter.

        Args:
            reverse: Whether to invert the first-column probability signal.
        """
        self.reverse = reverse
        self.fit_calls = 0

    def fit(self, X, y):
        """Record one fit call and return the estimator unchanged."""
        self.fit_calls += 1
        return self

    def predict_proba(self, X):
        """Return first-column probabilities, optionally reversed."""
        p = np.asarray(X)[:, 0]
        if self.reverse:
            p = 1 - p
        return np.column_stack([1 - p, p])

    def predict(self, X):
        """Convert positive-class probabilities to labels at 0.5."""
        return (self.predict_proba(X)[:, 1] >= .5).astype(int)


def test_selects_highest_validation_auc():
    """Verify candidate selection retains the model with highest validation AUC."""
    X = np.array([[.1], [.9], [.2], [.8]])
    y = np.array([0, 1, 0, 1])
    bad, good = Candidate(True), Candidate()
    winner, metrics, task = select_best_model([bad, good], X, X, y, y)
    assert winner is good
    assert metrics["auc"] == 1.
    assert task == "classification"
    assert bad.fit_calls == good.fit_calls == 1
