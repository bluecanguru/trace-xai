"""Use with the existing tests/conftest.py framework path setup."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
import yaml

from xai_framework import cli

@pytest.fixture
def fixture_bundle(tmp_path):
    """Create a minimal saved bundle and YAML configuration for CLI tests.

    Args:
        tmp_path: Pytest-managed temporary directory.

    Returns:
        Path to the generated YAML configuration. The bundle deliberately
        contains an invalid pickle so validation tests detect unsafe loading.
    """
    root = tmp_path / "project"
    bundle = root / "results" / "toy" / "bundle"
    bundle.mkdir(parents=True)
    # Deliberately invalid pickle: validation-only MUST NOT deserialize this.
    (bundle / "model.pkl").write_bytes(b"not a pickle")
    metadata = dict(dataset="toy", feature_names=["a", "b"], n_features=2,
                    model_class="Toy", n_train=4, n_test=4, scoring="roc_auc")
    (bundle / "metadata.json").write_text(json.dumps(metadata))
    for split in ("train", "test"):
        pd.DataFrame({"a": [0., 1., 2., 3.], "b": [1., 0., 1., 0.]}).to_csv(
            bundle / f"X_{split}.csv", index=False)
        pd.Series([0, 1, 0, 1]).to_csv(bundle / f"y_{split}.csv", index=False)
    cfg_dir = root / "src" / "xai_framework" / "configs"
    cfg_dir.mkdir(parents=True)
    cfg_file = cfg_dir / "toy.yaml"
    cfg_file.write_text(yaml.safe_dump(dict(
        project_root="../../..", dataset_name="toy", bundle_dir="results/toy/bundle",
        output_dir="results/cli_runs/toy_run_01")))
    return cfg_file


def test_paths_are_not_relative_to_shell(fixture_bundle, monkeypatch, tmp_path):
    """Verify configuration paths resolve from the YAML location, not the shell."""
    monkeypatch.chdir(tmp_path)
    cfg = cli.load_config(fixture_bundle)
    assert cfg["bundle_dir"].is_dir()
    assert cfg["output_dir"].name == "toy_run_01"


def test_validation_never_loads_pickle_or_writes_output(fixture_bundle):
    """Verify validation-only mode neither deserialises nor creates output."""
    cfg = cli.load_config(fixture_bundle)
    assert cli.main(["--config", str(fixture_bundle), "--validate-only"]) == 0
    assert not cfg["output_dir"].exists()


def test_trust_acknowledgement_required(fixture_bundle):
    """Verify a model rerun requires explicit acknowledgement of pickle trust."""
    with pytest.raises(SystemExit) as exc:
        cli.main(["--config", str(fixture_bundle)])
    assert exc.value.code == 2


def test_column_order_mismatch_is_rejected(fixture_bundle):
    """Verify bundle validation rejects matrices with reordered feature columns."""
    cfg = cli.load_config(fixture_bundle)
    path = cfg["bundle_dir"] / "X_test.csv"
    pd.read_csv(path)[["b", "a"]].to_csv(path, index=False)
    with pytest.raises(ValueError, match="column names/order"):
        cli.read_bundle(cfg)


def test_dataset_mismatch_is_rejected(fixture_bundle):
    """Verify the requested dataset identifier must match bundle metadata."""
    cfg = cli.load_config(fixture_bundle)
    cfg["dataset_name"] = "wrong"
    with pytest.raises(ValueError, match="dataset_name"):
        cli.read_bundle(cfg)


def test_existing_output_is_never_overwritten(fixture_bundle):
    """Verify nonempty rerun output is preserved and rejected as a destination."""
    cfg = cli.load_config(fixture_bundle)
    cfg["output_dir"].mkdir(parents=True)
    (cfg["output_dir"] / "keep.txt").write_text("keep")
    with pytest.raises(ValueError, match="already exists"):
        cli.read_bundle(cfg)
    assert (cfg["output_dir"] / "keep.txt").read_text() == "keep"


def test_input_dataset_cannot_be_output(fixture_bundle):
    """Verify rerun output cannot overlap the canonical source-result folder."""
    cfg = cli.load_config(fixture_bundle)
    cfg["output_dir"] = cfg["bundle_dir"].parent
    with pytest.raises(ValueError, match="separate"):
        cli.read_bundle(cfg)


def test_runtime_loads_fitted_model_without_training(fixture_bundle, monkeypatch):
    """Verify runtime preparation loads but never refits the trusted model."""
    import joblib
    cfg = cli.load_config(fixture_bundle)
    metadata, data, names, scoring = cli.read_bundle(cfg)
    class Toy:
        """Represent a fitted binary estimator that fails if refitting occurs."""

        classes_ = np.array([0, 1])
        n_features_in_ = 2
        fit = Mock(side_effect=AssertionError("Must not retrain"))
    model = Toy()
    monkeypatch.setattr(joblib, "load", lambda path: model)
    runtime = cli.prepare_runtime(cfg, metadata, data, names, scoring)
    assert runtime["model"] is model
    assert isinstance(runtime["X_test"], np.ndarray)
    assert "train_model_fn" not in runtime
    model.fit.assert_not_called()


def test_cli_dispatches_existing_pipeline(fixture_bundle, monkeypatch):
    """Verify a trusted CLI rerun dispatches the pipeline and writes a manifest."""
    from xai_framework import run_pipeline
    cfg = cli.load_config(fixture_bundle)
    sentinel = {"output_dir": str(cfg["output_dir"])}
    monkeypatch.setattr(cli, "prepare_runtime", lambda *args: sentinel)
    def fake_run(runtime):
        """Write the minimum complete local-output pair expected by the CLI."""
        assert runtime is sentinel
        out = Path(runtime["output_dir"])
        out.mkdir(parents=True)
        pd.DataFrame({"CaseIndex": [0]}).to_csv(out / "representative_cases.csv", index=False)
        pd.DataFrame({"CaseIndex": [0]}).to_csv(out / "lime_explanations.csv", index=False)
    monkeypatch.setattr(run_pipeline, "run_pipeline", fake_run)
    assert cli.main(["--config", str(fixture_bundle), "--trust-model"]) == 0
    manifest = json.loads((cfg["output_dir"] / "cli_run.json").read_text())
    assert manifest["retraining"] is False
