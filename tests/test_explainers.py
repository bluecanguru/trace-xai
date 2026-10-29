"""Unit tests for case selection, LIME summaries, and native importance."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from explainers.local_explain import get_representative_cases
from explainers.lime_explainer import summarize_lime, summarize_lime_seed_stability
from explainers.native import run_native


class ProbabilityColumnModel:
    """Treat the first input column as the positive-class probability."""

    def predict_proba(self, X):
        """Return complementary binary probabilities from the first column."""
        p = np.asarray(X)[:, 0]
        return np.column_stack([1 - p, p])


@pytest.mark.parametrize("as_dataframe", [True, False])
def test_representative_cases_preserve_positions_not_dataframe_labels(as_dataframe):
    """Verify case indices use row positions and do not mutate input labels."""
    X = pd.DataFrame({"p": [.95, .05, .90, .10, .51]}, index=[10, 20, 30, 40, 50])
    original = X.copy(deep=True)
    y = np.array([1, 0, 0, 1, 1])
    cases = get_representative_cases(
        ProbabilityColumnModel(), X if as_dataframe else X.to_numpy(), y, ["p"]
    )
    assert cases.set_index("CaseType")["CaseIndex"].to_dict() == {
        "TP": 0, "TN": 1, "FP": 2, "FN": 3, "Borderline": 4
    }
    np.testing.assert_array_equal(cases.index, cases.CaseIndex)
    pd.testing.assert_frame_equal(X, original)


def test_missing_categories_and_duplicate_borderline_are_handled():
    """Verify unavailable categories are omitted and selected rows remain unique."""
    cases = get_representative_cases(
        ProbabilityColumnModel(), np.array([[.9], [.1]]), [1, 0], ["p"]
    )
    assert set(cases.CaseType) == {"TP", "TN"}
    assert cases.CaseIndex.is_unique


def test_mismatched_label_length_raises():
    """Verify case selection rejects misaligned feature and label lengths."""
    with pytest.raises(ValueError, match="different numbers"):
        get_representative_cases(
            ProbabilityColumnModel(), np.array([[.8], [.2]]), [1], ["p"]
        )


def test_custom_threshold_is_used():
    """Verify representative-case predictions honour a custom threshold."""
    cases = get_representative_cases(
        ProbabilityColumnModel(), np.array([[.6], [.8]]), [0, 1], ["p"], threshold=.7
    )
    assert cases.set_index("CaseIndex").loc[0, "y_pred"] == 0


def test_multiple_cases_are_selected_in_confidence_order():
    """Verify multiple cases are ranked by confidence within each category."""
    X = np.array([[.99], [.90], [.80], [.01], [.10], [.20]])
    y = np.array([1, 1, 1, 0, 0, 0])
    cases = get_representative_cases(
        ProbabilityColumnModel(), X, y, ["p"], n_per_type=2
    )
    assert cases.loc[cases.CaseType == "TP", "CaseIndex"].tolist() == [0, 1]
    assert cases.loc[cases.CaseType == "TN", "CaseIndex"].tolist() == [3, 4]
    assert cases.groupby("CaseType").CaseRank.apply(list).to_dict() == {
        "Borderline": [1, 2], "TN": [1, 2], "TP": [1, 2],
    }


def test_lime_summary_aggregates_cases_and_seeds():
    """Verify LIME summaries aggregate weights, cases, seeds, and frequency."""
    raw = pd.DataFrame({
        "CaseType": ["TP"] * 4,
        "CaseIndex": [1, 1, 2, 2],
        "LIMESeed": [0, 1, 0, 1],
        "Rule": ["x > 0"] * 4,
        "Weight": [.2, .4, -.1, .1],
    })
    summary = summarize_lime(raw, {"TP": 2}).iloc[0]
    assert summary.NExplanations == 4
    assert summary.NCases == 2
    assert summary.NSeeds == 2
    assert summary.SelectionFrequency == 1
    assert summary.MeanWeight == pytest.approx(.15)
    assert summary.MeanAbsWeight == pytest.approx(.2)


def test_lime_seed_stability_reports_rank_and_sign_agreement():
    """Verify identical seed rankings yield perfect overlap and sign agreement."""
    raw = pd.DataFrame({
        "CaseType": ["TP"] * 6,
        "CaseIndex": [1] * 6,
        "CaseRank": [1] * 6,
        "LIMESeed": [0, 0, 0, 1, 1, 1],
        "Rule": ["a", "b", "c", "a", "b", "c"],
        "Weight": [.5, -.3, .1, .4, -.2, .05],
    })
    summary = summarize_lime_seed_stability(raw, top_k=2).iloc[0]
    assert summary.NSeeds == 2
    assert summary.MeanPairwiseSpearman == pytest.approx(1)
    assert summary.MeanTopKOverlap == pytest.approx(1)
    assert summary.MeanRuleSignConsistency == pytest.approx(1)


def test_native_coefficients_use_magnitude_and_keep_sign():
    """Verify coefficient importance uses magnitude while retaining direction."""
    result = run_native(SimpleNamespace(coef_=np.array([[-2., .5]])), ["a", "b"])
    assert result.Feature.tolist() == ["a", "b"]
    assert result.iloc[0].Importance == 2.
    assert result.iloc[0].RawValue == -2.


def test_statsmodels_constants_are_excluded():
    """Verify Statsmodels intercept and constant terms are not ranked as features."""
    model = SimpleNamespace(params=pd.Series([10., 20., -3.], index=["Intercept", "const", "a"]))
    result = run_native(model, ["Intercept", "const", "a"])
    assert result.Feature.tolist() == ["a"]
    assert result.iloc[0].Importance == 3.


def test_unsupported_native_model_returns_none():
    """Verify models without native importance return an unavailable result."""
    assert run_native(object(), ["a"]) is None
