# selection/model_selector.py

from core.metrics import evaluate_model, detect_task_type

def select_best_model(models, X_train, X_test, y_train, y_test):

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

        # unified scoring rule
        if task == "classification":
            score = metrics.get("auc", metrics.get("accuracy", 0))
        else:
            score = metrics.get("r2", metrics.get("mse", 0))

        if score > best_score:
            best_score = score
            best_model = model
            best_metrics = metrics

    return best_model, best_metrics, task