import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score


class StatsModelsWrapper(ClassifierMixin, BaseEstimator):
    """
    Adapter allowing a fitted statsmodels binary classification model
    to be used by scikit-learn utilities.

    The underlying statsmodels model is already fitted.
    """

    def __init__(self, model):
        """Wrap a fitted Statsmodels result with the framework prediction interface."""
        self.model = model
        self.classes_ = np.array([0, 1])

    def fit(self, X, y=None):
        """
        Dummy fit method required by sklearn estimator utilities.

        The wrapped statsmodels model is already fitted, so no
        refitting is performed here.
        """
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        """
        Return probabilities for classes 0 and 1.
        """
        p = np.asarray(self.model.predict(X, transform=False))

        return np.column_stack([
            1 - p,
            p
        ])

    def predict(self, X):
        """
        Return binary predictions using threshold 0.5.
        """
        p = np.asarray(self.model.predict(X, transform=False))

        return (p >= 0.5).astype(int)

    def score(self, X, y):
        """
        Default score: ROC AUC.
        """
        p = np.asarray(self.model.predict(X, transform=False))

        return roc_auc_score(y, p)
