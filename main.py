"""Safely validate or rerun XAI for multiple saved dataset bundles.

This orchestrates the existing tested xai_framework package. It never trains
models, never overwrites canonical results, and never deletes partial outputs.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from xai_framework.cli import load_config, prepare_runtime, read_bundle
from xai_framework.compare_results import compare_directories
from xai_framework.run_pipeline import run_pipeline
from xai_framework.versioning import runtime_versions


DATASETS = ("heart", "stroke", "diabetes", "ckd", "hba1c")


def utc_id():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def config_paths(configs_dir, datasets):
    directory = Path(configs_dir).expanduser().resolve()
    paths = {dataset: directory / f"{dataset}.yaml" for dataset in datasets}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise ValueError(f"Missing configuration files: {missing}")
    return paths


def validate_configs(paths):
    validated = {}
    for dataset, path in paths.items():
        config = load_config(path)
        if config["dataset_name"] != dataset:
            raise ValueError(
                f"{path}: dataset_name={config['dataset_name']!r}, expected {dataset!r}"
            )
        metadata, data, names, scoring = read_bundle(config)
        validated[dataset] = (config, metadata, data, names, scoring)
        print(
            f"VALID {dataset}: {len(names)} features/design columns, "
            f"{len(data['X_test'])} test rows, scoring={scoring}"
        )
    return validated


def check_local_outputs(output):
    cases_path = output / "representative_cases.csv"
    lime_path = output / "lime_explanations.csv"
    if not cases_path.is_file() or not lime_path.is_file():
        raise RuntimeError("Representative-case or LIME export is missing.")
    cases, lime = pd.read_csv(cases_path), pd.read_csv(lime_path)
    if (cases.empty or lime.empty or "CaseIndex" not in cases
            or "CaseIndex" not in lime
            or set(cases["CaseIndex"]) != set(lime["CaseIndex"])):
        raise RuntimeError("LIME output does not cover every selected case.")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def run_batch(validated, batch_root, continue_on_error=False,
              compare=False, rtol=0.005, atol=0.0005):
    batch_root = Path(batch_root).resolve()
    if batch_root.exists():
        raise ValueError(f"Batch output already exists: {batch_root}")
    batch_root.mkdir(parents=True)
    manifest = {
        "batch_id": batch_root.name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": runtime_versions(),
        "retraining": False,
        "stability": "not rerun: saved bundles contain no train_model_fn",
        "datasets": {}
    }
    manifest_path = batch_root / "batch_manifest.json"
    write_json(manifest_path, manifest)

    for dataset, (config, metadata, data, names, scoring) in validated.items():
        output = batch_root / dataset
        state = {
            "status": "RUNNING",
            "source_bundle": str(config["bundle_dir"]),
            "output": str(output),
            "scoring": scoring
        }
        manifest["datasets"][dataset] = state
        write_json(manifest_path, manifest)
        try:
            run_config = dict(config)
            run_config["output_dir"] = output
            runtime = prepare_runtime(
                run_config, metadata, data, names, scoring
            )
            run_pipeline(runtime)
            check_local_outputs(output)
            state["status"] = "PASSED"
            state["completed_at_utc"] = datetime.now(timezone.utc).isoformat()

            if compare:
                reference = config["bundle_dir"].parent
                report = compare_directories(
                    reference, output, rtol=rtol, atol=atol
                )
                report_path = batch_root / "comparisons" / f"{dataset}.json"
                write_json(report_path, report)
                state["comparison"] = report["summary"]
                state["comparison_report"] = str(report_path)

            write_json(output / "cli_run.json", {
                "source_bundle": str(config["bundle_dir"]),
                "dataset": dataset,
                "scoring": scoring,
                "batch_id": batch_root.name,
                "runtime": runtime_versions(),
                "retraining": False,
                "stability": "not rerun: no train_model_fn"
            })
            print(f"PASSED {dataset}: {output}")
        except Exception as exc:
            state["status"] = "FAILED"
            state["error_type"] = type(exc).__name__
            state["error"] = str(exc)
            print(f"FAILED {dataset}: {type(exc).__name__}: {exc}", file=sys.stderr)
            if not continue_on_error:
                write_json(manifest_path, manifest)
                break
        finally:
            write_json(manifest_path, manifest)

    statuses = [item["status"] for item in manifest["datasets"].values()]
    manifest["status"] = (
        "PASSED" if statuses and all(status == "PASSED" for status in statuses)
        else "FAILED"
    )
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(manifest_path, manifest)
    print("Batch manifest:", manifest_path)
    return 0 if manifest["status"] == "PASSED" else 1


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate", "run"))
    parser.add_argument(
        "--datasets", nargs="+", choices=DATASETS, default=list(DATASETS),
        help="Datasets to process in order (default: all five)."
    )
    parser.add_argument(
        "--configs-dir", default="src/xai_framework/configs"
    )
    parser.add_argument(
        "--output-root", default="results/cli_runs/batches"
    )
    parser.add_argument(
        "--trust-models", action="store_true",
        help="Required for run: acknowledge that every selected pickle is trusted."
    )
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument(
        "--compare", action="store_true",
        help="Compare each successful rerun with its canonical dataset results."
    )
    parser.add_argument("--rtol", type=float, default=0.005)
    parser.add_argument("--atol", type=float, default=0.0005)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run" and not args.trust_models:
        parser.error("run requires --trust-models for your own trusted bundles.")
    if args.command == "validate" and (
        args.trust_models or args.continue_on_error or args.compare
    ):
        parser.error(
            "--trust-models, --continue-on-error, and --compare apply only to run."
        )
    if args.rtol < 0 or args.atol < 0:
        parser.error("Comparison tolerances must be nonnegative.")
    try:
        paths = config_paths(args.configs_dir, args.datasets)
        validated = validate_configs(paths)  # Validate all before loading any pickle.
        if args.command == "validate":
            print(f"All {len(validated)} selected bundles passed CSV/JSON validation.")
            return 0
        batch_id = f"batch_{utc_id()}"
        batch_root = Path(args.output_root).expanduser().resolve() / batch_id
        return run_batch(
            validated, batch_root,
            continue_on_error=args.continue_on_error,
            compare=args.compare, rtol=args.rtol, atol=args.atol
        )
    except KeyboardInterrupt:
        print("Interrupted. Existing partial output was preserved.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
