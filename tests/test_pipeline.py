import json
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest


def test_runner_calls_run_before_export(monkeypatch, tmp_path):
    import run_pipeline as runner

    events = []
    experiment = Mock()
    experiment.run.side_effect = lambda: events.append("run")
    experiment.export.side_effect = lambda output: events.append(("export", output))
    factory = Mock(return_value=experiment)
    monkeypatch.setattr(runner, "XAIExperiment", factory)
    config = {"output_dir": str(tmp_path)}
    assert runner.run_pipeline(config) is experiment
    factory.assert_called_once_with(config)
    assert events == ["run", ("export", str(tmp_path))]


@pytest.mark.integration
def test_real_small_binary_pipeline(tmp_path):
    """Real Native/Permutation/SHAP/LIME/export path; no thesis data or writes."""
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from run_pipeline import run_pipeline

    X, y = make_classification(
        n_samples=60, n_features=3, n_informative=2, n_redundant=0, random_state=42
    )
    names = ["a", "b", "c"]
    X = pd.DataFrame(X, columns=names)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.3, random_state=42, stratify=y
    )
    model = LogisticRegression(random_state=42, max_iter=500).fit(X_train, y_train)
    config = dict(
        dataset_name="synthetic", model=model, X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test, feature_names=names,
        scoring="roc_auc", output_dir=str(tmp_path)
    )
    experiment = run_pipeline(config)
    # The configuration intentionally omits a retraining callback.
    assert experiment.state.results["stability"] is None
    for filename in ["native_importance", "permutation_importance", "shap_importance"]:
        table = pd.read_csv(tmp_path / (filename + ".csv"))
        assert set(table.Feature) == set(names)
    cases = pd.read_csv(tmp_path / "representative_cases.csv")
    lime = pd.read_csv(tmp_path / "lime_explanations.csv")
    lime_summary = pd.read_csv(tmp_path / "lime_summary.csv")
    lime_seed_stability = pd.read_csv(tmp_path / "lime_seed_stability.csv")
    assert not cases.empty and not lime.empty
    assert not lime_summary.empty and not lime_seed_stability.empty
    assert cases.CaseIndex.is_unique
    assert set(lime.CaseIndex) == set(cases.CaseIndex)
    assert set(lime.LIMESeed) == set(range(20))
    assert (lime.groupby("CaseIndex").LIMESeed.nunique() == 20).all()
    assert (lime_seed_stability.NSeeds == 20).all()
    for idx, group in lime.groupby("CaseIndex"):
        expected = model.predict_proba(X_test.iloc[[int(idx)]])[0, 1]
        np.testing.assert_allclose(group.Probability, expected)
    metadata = json.loads((tmp_path / "bundle" / "metadata.json").read_text())
    assert metadata["dataset"] == "synthetic"
    assert metadata["n_test"] == len(X_test)
    timing = json.loads((tmp_path / "execution_times.json").read_text())
    assert timing["global_seconds"] > 0
    assert timing["local_seconds"] > 0
    assert timing["stability_performed"] is False
    assert timing["total_seconds"] >= timing["analysis_seconds"]
    assert metadata["execution_times"]["total_seconds"] == timing["total_seconds"]
