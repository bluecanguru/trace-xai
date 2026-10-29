"""Adapt fitted Statsmodels binary models to the scikit-learn interface."""

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score


class StatsModelsWrapper(ClassifierMixin, BaseEstimator):
    """Expose a fitted Statsmodels binary model as a scikit-learn estimator.

    The adapter is intended for explanation and evaluation utilities that
    expect ``predict_proba``, ``predict``, ``score``, and ``classes_``. The
    underlying result is already fitted; calling :meth:`fit` does not estimate
    new parameters.

    Attributes:
        model: Fitted Statsmodels result object.
        classes_: Binary class labels in scikit-learn order.
    """

    def __init__(self, model):
        """Initialise the adapter around a fitted result.

        Args:
            model: Fitted Statsmodels binary-result object whose ``predict``
                method accepts a design matrix and ``transform=False``.
        """
        self.model = model
        self.classes_ = np.array([0, 1])

    def fit(self, X, y=None):
        """Satisfy the estimator protocol without refitting the model.

        Args:
            X: Ignored feature matrix, accepted for scikit-learn compatibility.
            y: Ignored target values.

        Returns:
            This unchanged adapter instance.

        Notes:
            The wrapped Statsmodels parameters remain unchanged.
        """
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        """Return two-column probabilities for the binary classes.

        Args:
            X: Design matrix aligned with the fitted Statsmodels model.

        Returns:
            NumPy array with class-0 probabilities in the first column and
            class-1 probabilities in the second.
        """
        p = np.asarray(self.model.predict(X, transform=False))

        return np.column_stack([
            1 - p,
            p
        ])

    def predict(self, X):
        """Return binary predictions at the fixed 0.5 threshold.

        Args:
            X: Design matrix aligned with the fitted Statsmodels model.

        Returns:
            Integer NumPy array containing zeros and ones.
        """
        p = np.asarray(self.model.predict(X, transform=False))

        return (p >= 0.5).astype(int)

    def score(self, X, y):
        """Calculate ROC AUC from the model's positive-class probabilities.

        Args:
            X: Design matrix aligned with the fitted Statsmodels model.
            y: Observed binary outcomes.

        Returns:
            Area under the receiver operating characteristic curve.

        Raises:
            ValueError: If ROC AUC cannot be calculated, such as when ``y``
                contains only one class.
        """
        p = np.asarray(self.model.predict(X, transform=False))

        return roc_auc_score(y, p)
