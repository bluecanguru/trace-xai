"""Tests for the isolated optional HbA1c experiment."""

import sys
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

EXPERIMENT_DIR = (
    Path(__file__).resolve().parents[1] / "experiments" / "hba1c_optional"
)
sys.path.insert(0, str(EXPERIMENT_DIR))

import hba1c_grouped_stability as pilot


class SumProbabilityModel:
    """Convert the sum of two design columns to a logistic probability."""

    def predict(self, X, transform=False):
        """Return deterministic logistic probabilities for grouped tests."""
        linear = np.asarray(X)[:, 0] + np.asarray(X)[:, 1]
        return 1 / (1 + np.exp(-linear))


def test_grouped_permutation_is_reproducible_and_does_not_mutate_input():
    """Verify grouped permutation is seeded and leaves the design matrix intact."""
    rng = np.random.default_rng(4)
    signal = rng.normal(size=120)
    X = np.column_stack([signal, signal, rng.normal(size=120)])
    y = (signal > 0).astype(float)
    original = X.copy()
    groups = OrderedDict([("joint_signal", (0, 1)), ("noise", (2,))])

    first = pilot.grouped_permutation_importance(
        SumProbabilityModel(), X, y, groups,
        n_repeats=4, scoring="neg_log_loss", random_seed=9,
    )
    second = pilot.grouped_permutation_importance(
        SumProbabilityModel(), X, y, groups,
        n_repeats=4, scoring="neg_log_loss", random_seed=9,
    )

    pd.testing.assert_frame_equal(first, second)
    np.testing.assert_array_equal(X, original)
    assert first.iloc[0]["Group"] == "joint_signal"
    assert first.iloc[0]["DesignColumns"] == 2


def test_term_groups_exclude_intercept_and_cover_every_other_column():
    """Verify Patsy terms group all non-intercept design columns exactly once."""
    design_info = SimpleNamespace(
        term_name_slices=OrderedDict(
            [
                ("Intercept", slice(0, 1)),
                ("category", slice(1, 3)),
                ("continuous", slice(3, 4)),
            ]
        )
    )
    model = SimpleNamespace(
        model=SimpleNamespace(data=SimpleNamespace(design_info=design_info))
    )
    groups = pilot.extract_term_groups(
        model, ["Intercept", "category_b", "category_c", "continuous"]
    )
    assert groups == OrderedDict(
        [("category", (1, 2)), ("continuous", (3,))]
    )


def test_stratified_bootstrap_preserves_class_counts():
    """Verify stratified resampling preserves the original per-class sizes."""
    y = np.array([0] * 17 + [1] * 8)
    indices = pilot.stratified_bootstrap_indices(y, np.random.default_rng(12))
    sampled = y[indices]
    assert len(indices) == len(y)
    assert np.sum(sampled == 0) == 17
    assert np.sum(sampled == 1) == 8


def test_pairwise_summary_has_expected_perfect_agreement():
    """Verify proportional rankings yield perfect pairwise stability measures."""
    raw = pd.DataFrame(
        {
            "Bootstrap": [1, 1, 1, 2, 2, 2],
            "Group": ["a", "b", "c", "a", "b", "c"],
            "DesignColumns": [1] * 6,
            "Importance": [3.0, 2.0, 1.0, 6.0, 4.0, 2.0],
        }
    )
    summary, pairwise = pilot.summarise_bootstrap(raw, top_k=2)
    assert list(summary["Group"]) == ["a", "b", "c"]
    assert pairwise.loc[0, "SpearmanRankCorrelation"] == pytest.approx(1.0)
    assert pairwise.loc[0, "TopKOverlap"] == pytest.approx(1.0)


def test_output_protection_rejects_canonical_paths():
    """Verify the optional pilot cannot write into canonical HbA1c results."""
    canonical = Path("C:/isolated-test/results/hba1c")
    bundle = canonical / "bundle"
    with pytest.raises(ValueError, match="separate"):
        pilot.ensure_isolated_output(bundle, canonical)


def test_full_mode_requires_explicit_confirmation():
    """Verify a full optional run requires an explicit confirmation argument."""
    with pytest.raises(SystemExit):
        pilot.parse_args(["--mode", "full"])


def test_pilot_assessment_recommends_full_only_when_feasible():
    """Verify a converged, fast pilot receives the proceed recommendation."""
    log = pd.DataFrame(
        {
            "Status": ["success"] * 5,
            "Converged": [True] * 5,
            "FitSeconds": [2.0] * 5,
            "PermutationSeconds": [3.0] * 5,
            "TotalSeconds": [5.0] * 5,
        }
    )
    assessment = pilot.assess_pilot(
        log,
        pd.DataFrame({"SpearmanRankCorrelation": [0.8]}),
        requested=5,
        full_target=100,
        max_projected_hours=1.0,
    )
    assert assessment["decision"] == "PROCEED_TO_FULL_RUN"
    assert assessment["projected_full_runtime_hours"] == pytest.approx(500 / 3600)
