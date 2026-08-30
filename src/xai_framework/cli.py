"""Rerun the existing XAI pipeline from a trusted saved experiment bundle.

Run from any directory using --config. Relative bundle/output paths are resolved
against project_root; project_root is relative to the YAML file, not the shell.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

try:
    from xai_framework.versioning import runtime_versions
except ImportError:
    from versioning import runtime_versions


def load_config(filename):
    path = Path(filename).expanduser().resolve()
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("YAML must contain a mapping.")
    allowed = {"project_root", "dataset_name", "bundle_dir", "output_dir", "scoring"}
    unknown = set(config) - allowed
    if unknown:
        raise ValueError(f"Unsupported YAML fields: {sorted(unknown)}")
    for key in ("dataset_name", "bundle_dir", "output_dir"):
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise ValueError(f"YAML requires a nonempty string: {key}")
    root_value = config.get("project_root", ".")
    if not isinstance(root_value, str):
        raise ValueError("project_root must be a path string.")
    root = (path.parent / Path(root_value).expanduser()).resolve()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    replacements = {
        "dataset": config["dataset_name"],
        "timestamp": timestamp
    }
    for key in ("bundle_dir", "output_dir"):
        try:
            rendered = config[key].format(**replacements)
        except KeyError as exc:
            raise ValueError(
                "Path templates support only {dataset} and {timestamp}."
            ) from exc
        config[key] = (root / Path(rendered).expanduser()).resolve()
    if "scoring" in config and not isinstance(config["scoring"], str):
        raise ValueError("scoring must be a scikit-learn scoring name.")
    return config


def validate_output(bundle, output):
    bundle, output = Path(bundle).resolve(), Path(output).resolve()
    source_results = bundle.parent
    # Never write over, inside, or above the input dataset results.
    if (output == source_results or source_results in output.parents
            or output in source_results.parents):
        raise ValueError("Output must be separate from the source dataset results.")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(
            f"Output already exists and is not an empty directory: {output}. "
            "Choose a new output_dir in YAML. No files were deleted."
        )


def read_bundle(config):
    """Validate CSV/JSON only. Does not deserialize model.pkl or write files."""
    bundle = config["bundle_dir"]
    required = ["model.pkl", "metadata.json", "X_train.csv", "X_test.csv",
                "y_train.csv", "y_test.csv"]
    missing = [name for name in required if not (bundle / name).is_file()]
    if missing:
        raise ValueError(f"Bundle is missing files: {missing}")
    validate_output(bundle, config["output_dir"])
    metadata = json.loads((bundle / "metadata.json").read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("metadata.json must contain an object.")
    if metadata.get("dataset") != config["dataset_name"]:
        raise ValueError("YAML dataset_name does not match bundle metadata.dataset.")
    names = metadata.get("feature_names")
    if (not isinstance(names, list) or not names
            or not all(isinstance(name, str) for name in names)
            or len(names) != len(set(names))):
        raise ValueError("metadata.feature_names must be a unique nonempty list of strings.")
    if metadata.get("n_features") != len(names):
        raise ValueError("metadata.n_features differs from feature_names.")
    data = {}
    for split in ("train", "test"):
        X = pd.read_csv(bundle / f"X_{split}.csv", float_precision="round_trip")
        y_frame = pd.read_csv(bundle / f"y_{split}.csv", float_precision="round_trip")
        if X.columns.tolist() != names:
            raise ValueError(f"X_{split} column names/order differ from metadata.")
        if X.empty or y_frame.shape[1] != 1 or len(X) != len(y_frame):
            raise ValueError(f"Invalid or mismatched X/y dimensions for {split}.")
        if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes):
            raise ValueError(f"X_{split} must be the numeric preprocessed/design matrix.")
        if not np.isfinite(X.to_numpy(dtype=float)).all():
            raise ValueError(f"X_{split} contains nonfinite values.")
        y = y_frame.iloc[:, 0]
        if y.isna().any() or not set(y.unique()).issubset({0, 1}):
            raise ValueError(f"y_{split} must contain binary 0/1 labels.")
        if metadata.get(f"n_{split}", len(X)) != len(X):
            raise ValueError(f"metadata.n_{split} does not match the CSV row count.")
        data[f"X_{split}"] = X
        data[f"y_{split}"] = y.to_numpy()
    scoring = config.get("scoring", metadata.get("scoring", "roc_auc"))
    if scoring == "roc_auc" and len(np.unique(data["y_test"])) != 2:
        raise ValueError("ROC AUC requires both classes in y_test.")
    return metadata, data, names, scoring


def prepare_runtime(config, metadata, data, names, scoring):
    """Call only after the user has explicitly acknowledged pickle trust."""
    import joblib
    try:
        model = joblib.load(config["bundle_dir"] / "model.pkl")
    except Exception as exc:
        raise RuntimeError(
            "Could not load model.pkl. Use the Python/library environment that "
            "created the bundle. Formula-based statsmodels objects may also "
            "require their original formula dependencies."
        ) from exc
    if type(model).__name__ != metadata.get("model_class"):
        raise ValueError("Loaded model class differs from bundle metadata.")
    if hasattr(model, "params"):
        expected = list(model.model.exog_names)
        if expected != names:
            raise ValueError("Statsmodels design terms/order differ from the saved matrix.")
        # Keep Intercept and all interactions exactly as exported. No Patsy rebuild.
    else:
        if getattr(model, "n_features_in_", len(names)) != len(names):
            raise ValueError("Loaded model feature count differs from the matrix.")
        classes = getattr(model, "classes_", None)
        if classes is None or list(classes) != [0, 1]:
            raise ValueError("This CLI supports fitted binary classifiers with classes [0, 1].")
        expected = getattr(model, "feature_names_in_", None)
        if expected is not None and list(expected) != names:
            raise ValueError("Loaded model feature names/order differ from the matrix.")
        if expected is None:
            data = dict(data)
            data["X_train"] = data["X_train"].to_numpy()
            data["X_test"] = data["X_test"].to_numpy()
    return dict(data, dataset_name=config["dataset_name"], model=model,
                feature_names=names, scoring=scoring, output_dir=str(config["output_dir"]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a dataset YAML file")
    parser.add_argument("--validate-only", action="store_true",
                        help="Check CSV/JSON files without loading the model or writing results")
    parser.add_argument("--trust-model", action="store_true",
                        help="Acknowledge that model.pkl is trusted; pickle loading can execute code")
    args = parser.parse_args(argv)
    if not args.validate_only and not args.trust_model:
        parser.error("Use --validate-only first, or --trust-model for a bundle you trust.")
    try:
        config = load_config(args.config)
        metadata, data, names, scoring = read_bundle(config)
        print(f"Dataset: {config['dataset_name']}")
        print(f"Bundle: {config['bundle_dir']}")
        print(f"Output: {config['output_dir']}")
        print(f"Features/design columns: {len(names)}; scoring: {scoring}")
        print("Stability: not rerun (no retraining callback in a saved bundle).")
        if args.validate_only:
            print("CSV/JSON validation passed. Model loading and predictions are not verified.")
            return 0
        runtime = prepare_runtime(config, metadata, data, names, scoring)
        if __package__:
            from .run_pipeline import run_pipeline
        else:
            from run_pipeline import run_pipeline
        run_pipeline(runtime)
        # The framework can catch individual LIME errors; detect incomplete exports.
        output = config["output_dir"]
        cases = pd.read_csv(output / "representative_cases.csv")
        lime = pd.read_csv(output / "lime_explanations.csv")
        if (cases.empty or lime.empty or "CaseIndex" not in lime
                or set(cases["CaseIndex"]) != set(lime["CaseIndex"])):
            raise RuntimeError("LIME output does not cover every selected case.")
        manifest = {
            "source_bundle": str(config["bundle_dir"]),
            "dataset": config["dataset_name"], "scoring": scoring,
            "stability": "not rerun: no train_model_fn", "retraining": False,
            "note": "CSV roundtrips and library versions may affect numeric reproducibility."
        }
        manifest["runtime"] = runtime_versions()
        (output / "cli_run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print("Bundle-based XAI rerun completed. Original results were not overwritten.")
        return 0
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("If a run started, its output may be partial; use a fresh output_dir on retry.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
