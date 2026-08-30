from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from explainers.local_explain import get_representative_cases
from explainers.native import run_native


class ProbabilityColumnModel:
    def predict_proba(self, X):
        p = np.asarray(X)[:, 0]
        return np.column_stack([1 - p, p])


@pytest.mark.parametrize("as_dataframe", [True, False])
def test_representative_cases_preserve_positions_not_dataframe_labels(as_dataframe):
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
    cases = get_representative_cases(
        ProbabilityColumnModel(), np.array([[.9], [.1]]), [1, 0], ["p"]
    )
    assert set(cases.CaseType) == {"TP", "TN"}
    assert cases.CaseIndex.is_unique


def test_mismatched_label_length_raises():
    with pytest.raises(ValueError, match="different numbers"):
        get_representative_cases(
            ProbabilityColumnModel(), np.array([[.8], [.2]]), [1], ["p"]
        )


def test_custom_threshold_is_used():
    cases = get_representative_cases(
        ProbabilityColumnModel(), np.array([[.6], [.8]]), [0, 1], ["p"], threshold=.7
    )
    assert cases.set_index("CaseIndex").loc[0, "y_pred"] == 0


def test_native_coefficients_use_magnitude_and_keep_sign():
    result = run_native(SimpleNamespace(coef_=np.array([[-2., .5]])), ["a", "b"])
    assert result.Feature.tolist() == ["a", "b"]
    assert result.iloc[0].Importance == 2.
    assert result.iloc[0].RawValue == -2.


def test_statsmodels_constants_are_excluded():
    model = SimpleNamespace(params=pd.Series([10., 20., -3.], index=["Intercept", "const", "a"]))
    result = run_native(model, ["Intercept", "const", "a"])
    assert result.Feature.tolist() == ["a"]
    assert result.iloc[0].Importance == 3.


def test_unsupported_native_model_returns_none():
    assert run_native(object(), ["a"]) is None
