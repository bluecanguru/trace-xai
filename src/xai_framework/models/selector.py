
"""Fit and compare candidate estimators using task-appropriate metrics."""

from core.metrics import evaluate_model, detect_task_type

def select_best_model(models, X_train, X_test, y_train, y_test):
    """Fit candidate models and retain the highest-scoring estimator.

    Args:
        models: Iterable of unfitted estimators exposing ``fit`` and ``predict``.
        X_train: Feature matrix used to fit each candidate.
        X_test: Feature matrix used for model comparison.
        y_train: Training outcomes used for fitting and task detection.
        y_test: Outcomes aligned with ``X_test``.

    Returns:
        Tuple ``(best_model, best_metrics, task)`` containing the fitted winning
        estimator, its metric dictionary, and the inferred task label. If the
        iterable is empty, the first two values are ``None``.

    Notes:
        Classification candidates are ranked by ROC AUC when available and
        accuracy otherwise. Regression candidates are ranked by R-squared.
        Selection uses the supplied test data and is therefore intended as a
        convenience utility, not as a replacement for nested validation.
    """
    task = detect_task_type(y_train)

    best_model = None
    best_score = float("-inf")
    best_metrics = None

    for model in models:

        model.fit(X_train, y_train)

        metrics = evaluate_model(
            model,
            X_test,
            y_test,
            task=task
        )

        # Use the configured scoring rule for every candidate.
        if task == "classification":
            score = metrics.get("auc", metrics.get("accuracy", 0))
        else:
            score = metrics.get("r2", metrics.get("mse", 0))

        if score > best_score:
            best_score = score
            best_model = model
            best_metrics = metrics

    return best_model, best_metrics, task
