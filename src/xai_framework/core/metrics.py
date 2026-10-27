from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_squared_error, r2_score
)
import numpy as np

def evaluate_model(model, X_test, y_test, task="classification"):

    """Evaluate a fitted model using task-appropriate predictive metrics."""
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

    """Infer the supported prediction task from the target values."""
    y = np.asarray(y)

    unique = np.unique(y)

    # Detect regression-like targets.
    if len(unique) > 20 and np.issubdtype(y.dtype, np.floating):
        return "regression"

    return "classification"
