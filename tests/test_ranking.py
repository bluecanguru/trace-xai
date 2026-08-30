import numpy as np
import pandas as pd
import pytest

from core.ranking import compare_importance_ranks


def tables():
    native = pd.DataFrame({"Feature": ["a", "b", "c"], "Importance": [3., 2., 1.]})
    shap = native.rename(columns={"Importance": "MeanAbsSHAP"})
    return native, shap, native.copy()


def test_feature_alignment_does_not_depend_on_row_order():
    native, shap, perm = tables()
    merged, agreement = compare_importance_ranks(
        native.iloc[::-1], shap.iloc[[1, 2, 0]], perm
    )
    assert set(merged.Feature) == {"a", "b", "c"}
    assert all(value == pytest.approx(1.) for value in agreement.values())


def test_reverse_ranking_has_negative_correlation():
    native, shap, perm = tables()
    perm["Importance"] = [1., 2., 3.]
    _, agreement = compare_importance_ranks(native, shap, perm)
    assert agreement["Native_vs_Permutation"] == pytest.approx(-1.)


def test_only_common_features_are_compared():
    native, shap, perm = tables()
    native = native[native.Feature != "c"]
    merged, _ = compare_importance_ranks(native, shap, perm)
    assert set(merged.Feature) == {"a", "b"}


def test_native_importance_is_optional():
    _, shap, perm = tables()
    merged, agreement = compare_importance_ranks(None, shap, perm)
    assert len(merged) == 3
    assert set(agreement) == {"SHAP_vs_Permutation"}
    assert agreement["SHAP_vs_Permutation"] == pytest.approx(1.)


def test_tied_importances_receive_average_ranks():
    """Regression test: currently expected to FAIL, not silently xfail."""
    native, shap, perm = tables()
    native["Importance"] = perm["Importance"] = [2., 2., 1.]
    shap["MeanAbsSHAP"] = [2., 2., 1.]
    merged, agreement = compare_importance_ranks(native, shap.iloc[::-1], perm)
    ranked = merged.set_index("Feature")
    for column in ["NativeRank", "SHAPRank", "PermutationRank"]:
        np.testing.assert_allclose(ranked.loc[["a", "b"], column], [1.5, 1.5])
    assert agreement["Native_vs_SHAP"] == pytest.approx(1.)
