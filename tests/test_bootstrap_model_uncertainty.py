"""Unit tests for src/analysis/bootstrap_model_uncertainty.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "analysis"
    / "bootstrap_model_uncertainty.py"
)


def load_module():
    """Load the analysis script without requiring analysis to be a package."""
    spec = importlib.util.spec_from_file_location(
        "bootstrap_model_uncertainty",
        SCRIPT_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import analysis script: {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def uncertainty_module():
    assert SCRIPT_PATH.exists(), f"Missing analysis script: {SCRIPT_PATH}"
    return load_module()


def test_specificity_score(uncertainty_module):
    y_true = np.array([0, 0, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1, 0])

    observed = uncertainty_module.specificity_score(y_true, y_pred)

    assert observed == pytest.approx(2 / 3)


def test_perfect_predictions_produce_perfect_metrics(uncertainty_module):
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred = y_true.copy()
    y_prob = np.array([0.05, 0.95, 0.10, 0.90, 0.20, 0.80])

    metrics = uncertainty_module.calculate_metrics(y_true, y_pred, y_prob)

    assert set(metrics) == set(uncertainty_module.METRIC_ORDER)
    for value in metrics.values():
        assert value == pytest.approx(1.0)


def test_bootstrap_is_reproducible_and_well_formed(uncertainty_module):
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 0, 1, 0, 1, 1])
    y_prob = np.array([0.05, 0.20, 0.65, 0.30, 0.85, 0.40, 0.70, 0.95])

    first = uncertainty_module.bootstrap_intervals(
        y_true,
        y_pred,
        y_prob,
        n_bootstrap=20,
        seed=42,
    )
    second = uncertainty_module.bootstrap_intervals(
        y_true,
        y_pred,
        y_prob,
        n_bootstrap=20,
        seed=42,
    )

    first_point, first_intervals, first_valid = first
    second_point, second_intervals, second_valid = second

    assert first_point == second_point
    assert first_intervals == second_intervals
    assert first_valid == second_valid
    assert 0 < first_valid <= 20

    for metric in uncertainty_module.METRIC_ORDER:
        estimate = first_point[metric]
        lower, upper = first_intervals[metric]
        assert 0.0 <= lower <= upper <= 1.0
        assert 0.0 <= estimate <= 1.0


def test_single_class_input_is_rejected(uncertainty_module):
    y_true = np.zeros(10, dtype=int)
    y_pred = np.zeros(10, dtype=int)
    y_prob = np.full(10, 0.1)

    with pytest.warns(UserWarning, match="Only one class is present"):
        with pytest.raises(RuntimeError, match="No valid bootstrap samples"):
            uncertainty_module.bootstrap_intervals(
                y_true,
                y_pred,
                y_prob,
                n_bootstrap=20,
                seed=42,
            )


def test_latex_export_contains_expected_structure(uncertainty_module):
    row = {"Dataset": "Heart"}
    row.update(
        {
            metric: "0.900 [0.800, 0.950]"
            for metric in uncertainty_module.METRIC_ORDER
        }
    )
    summary = pd.DataFrame([row])

    latex = uncertainty_module.dataframe_to_latex(summary)

    assert r"\label{tab:bootstrap_performance}" in latex
    assert r"95\% bootstrap confidence intervals" in latex
    assert "Heart" in latex
    assert r"\begin{tabular}" in latex