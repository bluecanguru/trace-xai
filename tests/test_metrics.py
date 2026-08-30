import numpy as np
import pytest

from core.metrics import detect_task_type, evaluate_model


class FixedPredictions:
    def predict(self, X):
        return np.array([0, 1, 0, 1])

    def predict_proba(self, X):
        p = np.array([.1, .9, .2, .8])
        return np.column_stack([1 - p, p])


def test_binary_metrics_on_perfect_predictions():
    result = evaluate_model(FixedPredictions(), np.zeros((4, 1)), [0, 1, 0, 1])
    assert set(result) == {"accuracy", "f1", "auc"}
    assert all(v == pytest.approx(1.) for v in result.values())


def test_task_detection_for_current_binary_scope():
    assert detect_task_type([0, 1, 0, 1]) == "classification"


def test_continuous_target_heuristic():
    assert detect_task_type(np.linspace(0., 1., 30)) == "regression"
