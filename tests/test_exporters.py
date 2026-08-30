import json

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

from exporters.export import export_xai_results, save_xai_bundle


def test_csv_keeps_caseindex_and_nested_agreement_is_json(tmp_path):
    cases = pd.DataFrame({"CaseIndex": [11, 35], "CaseType": ["TP", "TN"]})
    export_xai_results({
        "representative_cases": cases,
        "rank_agreement": {"Native_vs_SHAP": .75},
        "not_available": None
    }, tmp_path)
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "representative_cases.csv"), cases)
    assert json.loads((tmp_path / "rank_agreement" / "Native_vs_SHAP.json").read_text()) == .75
    assert not list(tmp_path.glob("not_available*"))


def test_bundle_roundtrip_preserves_predictions_and_features(tmp_path):
    X = np.array([[0., 1.], [1., 0.], [2., 1.], [3., 0.]])
    y = np.array([0, 1, 0, 1])
    model = DummyClassifier(strategy="prior").fit(X, y)
    save_xai_bundle(model, X, X, y, y, ["a", "b"], tmp_path, {"dataset": "synthetic"})
    bundle = tmp_path / "bundle"
    assert {p.name for p in bundle.iterdir()} == {
        "model.pkl", "X_train.csv", "X_test.csv", "y_train.csv", "y_test.csv", "metadata.json"
    }
    metadata = json.loads((bundle / "metadata.json").read_text())
    assert metadata["feature_names"] == ["a", "b"]
    assert metadata["n_features"] == 2
    np.testing.assert_allclose(pd.read_csv(bundle / "X_test.csv"), X)
    np.testing.assert_array_equal(pd.read_csv(bundle / "y_test.csv").iloc[:, 0], y)
    # Load only the model this test just created, never an untrusted pickle.
    restored = joblib.load(bundle / "model.pkl")
    np.testing.assert_allclose(restored.predict_proba(X), model.predict_proba(X))
