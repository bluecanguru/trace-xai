"""Predictive metric calculation and lightweight task detection."""

from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_squared_error, r2_score
)
import numpy as np

def evaluate_model(model, X_test, y_test, task="classification"):

    """Evaluate a fitted model using task-appropriate predictive metrics.

    Args:
        model: Fitted estimator exposing ``predict`` and, optionally,
            ``predict_proba``.
        X_test: Evaluation feature matrix accepted by ``model``.
        y_test: Reference outcomes aligned with ``X_test``.
        task: Either ``"classification"`` or a regression task label. Any
            value other than ``"classification"`` follows the regression
            branch.

    Returns:
        Dictionary containing accuracy, F1, and ROC AUC for classification, or
        RMSE and R-squared for regression.

    Raises:
        ValueError: If a requested metric cannot be calculated from the
            supplied outcomes, for example ROC AUC with a single class.
    """
    y_pred = model.predict(X_test)

    if task == "classification":
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = y_pred

        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "auc": roc_auc_score(y_test, y_prob)
        }

    else:
        return {
            "rmse": mean_squared_error(y_test, y_pred, squared=False),
            "r2": r2_score(y_test, y_pred)
        }
        

def detect_task_type(y):

    """Infer whether a target resembles classification or regression.

    Args:
        y: One-dimensional array-like target values.

    Returns:
        ``"regression"`` for floating targets with more than 20 unique values;
        otherwise ``"classification"``.

    Notes:
        This is a lightweight heuristic used by the optional model selector,
        not a general-purpose task inference system.
    """
    y = np.asarray(y)

    unique = np.unique(y)

    # Detect regression-like targets.
    if len(unique) > 20 and np.issubdtype(y.dtype, np.floating):
        return "regression"

    return "classification"
