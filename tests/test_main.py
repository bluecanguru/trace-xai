import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

import main


def validated_fixture(tmp_path, datasets=("heart",)):
    result = {}
    for dataset in datasets:
        bundle = tmp_path / "canonical" / dataset / "bundle"
        bundle.mkdir(parents=True)
        config = {"dataset_name": dataset, "bundle_dir": bundle,
                  "output_dir": tmp_path / "unused"}
        metadata = {"model_class": "MockModel"}
        data = {"X_train": pd.DataFrame({"a": [0, 1]}),
                "X_test": pd.DataFrame({"a": [0, 1]}),
                "y_train": np.array([0, 1]), "y_test": np.array([0, 1])}
        result[dataset] = (config, metadata, data, ["a"], "roc_auc")
    return result


def fake_pipeline(config):
    output = Path(config["output_dir"])
    output.mkdir(parents=True)
    pd.DataFrame({"CaseIndex": [0]}).to_csv(
        output / "representative_cases.csv", index=False
    )
    pd.DataFrame({"CaseIndex": [0]}).to_csv(
        output / "lime_explanations.csv", index=False
    )


def test_run_requires_explicit_trust():
    with pytest.raises(SystemExit) as exc:
        main.main(["run"])
    assert exc.value.code == 2


def test_validate_does_not_prepare_models_or_create_outputs(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "config_paths", lambda *args: {"heart": Path("heart.yaml")})
    monkeypatch.setattr(main, "validate_configs", lambda paths: validated_fixture(tmp_path))
    prepare = Mock()
    monkeypatch.setattr(main, "prepare_runtime", prepare)
    assert main.main(["validate", "--datasets", "heart"]) == 0
    prepare.assert_not_called()
    assert not (tmp_path / "unused").exists()


def test_successful_batch_writes_manifest(monkeypatch, tmp_path):
    validated = validated_fixture(tmp_path, ("heart", "stroke"))
    monkeypatch.setattr(main, "prepare_runtime", lambda config, *args: {
        "output_dir": str(config["output_dir"])
    })
    monkeypatch.setattr(main, "run_pipeline", fake_pipeline)
    output = tmp_path / "batch"
    assert main.run_batch(validated, output) == 0
    manifest = json.loads((output / "batch_manifest.json").read_text())
    assert manifest["status"] == "PASSED"
    assert set(manifest["datasets"]) == {"heart", "stroke"}
    assert all(v["status"] == "PASSED" for v in manifest["datasets"].values())


def test_failure_stops_by_default_and_preserves_partial_output(monkeypatch, tmp_path):
    validated = validated_fixture(tmp_path, ("heart", "stroke"))
    monkeypatch.setattr(main, "prepare_runtime", lambda config, *args: {
        "output_dir": str(config["output_dir"])
    })
    monkeypatch.setattr(main, "run_pipeline", Mock(side_effect=RuntimeError("boom")))
    output = tmp_path / "batch"
    assert main.run_batch(validated, output) == 1
    manifest = json.loads((output / "batch_manifest.json").read_text())
    assert manifest["datasets"]["heart"]["status"] == "FAILED"
    assert "stroke" not in manifest["datasets"]
    assert output.exists()


def test_continue_on_error_attempts_remaining_datasets(monkeypatch, tmp_path):
    validated = validated_fixture(tmp_path, ("heart", "stroke"))
    monkeypatch.setattr(main, "prepare_runtime", lambda config, *args: {
        "output_dir": str(config["output_dir"])
    })
    calls = []
    def pipeline(config):
        calls.append(Path(config["output_dir"]).name)
        if len(calls) == 1:
            raise RuntimeError("first failed")
        fake_pipeline(config)
    monkeypatch.setattr(main, "run_pipeline", pipeline)
    output = tmp_path / "batch"
    assert main.run_batch(validated, output, continue_on_error=True) == 1
    assert calls == ["heart", "stroke"]


def test_existing_batch_directory_is_refused(tmp_path):
    output = tmp_path / "batch"
    output.mkdir()
    with pytest.raises(ValueError, match="already exists"):
        main.run_batch(validated_fixture(tmp_path), output)
