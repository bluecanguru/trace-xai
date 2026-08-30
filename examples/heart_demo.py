"""Small demonstration of the bundle-based XAI framework using Heart data.

Examples
--------
Validate the saved bundle without loading the model:

    python examples/heart_demo.py

Rerun the trusted saved model and summarize the generated explanations:

    python examples/heart_demo.py --run
"""

import argparse
import time
from pathlib import Path

import pandas as pd

from xai_framework.cli import main as cli_main


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT
    / "src"
    / "xai_framework"
    / "configs"
    / "heart.yaml"
)
CLI_RUNS_DIR = PROJECT_ROOT / "results" / "cli_runs"


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Validate the Heart reproducibility bundle or rerun its trusted "
            "saved model and display a concise XAI summary."
        )
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help=(
            "Load the trusted saved model and rerun XAI. Without this flag, "
            "the demo performs validation only."
        ),
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="Number of leading global features to display (default: 5).",
    )
    return parser


def read_csv(output_dir, filename):
    path = output_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"Expected demo output is missing: {path}")
    return pd.read_csv(path)


def print_table(title, table):
    print(f"\n{title}")
    print("-" * len(title))
    print(table.to_string(index=False))


def summarize_results(output_dir, top_n):
    native = read_csv(output_dir, "native_importance.csv")
    permutation = read_csv(output_dir, "permutation_importance.csv")
    shap = read_csv(output_dir, "shap_importance.csv")
    ranks = read_csv(output_dir, "rank_compare.csv")
    cases = read_csv(output_dir, "representative_cases.csv")
    lime = read_csv(output_dir, "lime_explanations.csv")

    native_columns = [
        column
        for column in ["Feature", "Importance", "RawValue", "Method"]
        if column in native.columns
    ]
    permutation_columns = [
        column
        for column in ["Feature", "Importance", "Std"]
        if column in permutation.columns
    ]
    shap_columns = [
        column
        for column in ["Feature", "MeanAbsSHAP", "MeanSHAP"]
        if column in shap.columns
    ]
    rank_columns = [
        column
        for column in [
            "Feature",
            "NativeRank",
            "SHAPRank",
            "PermutationRank",
        ]
        if column in ranks.columns
    ]
    case_columns = [
        column
        for column in [
            "CaseType",
            "CaseIndex",
            "y_true",
            "y_pred",
            "y_prob",
            "distance",
        ]
        if column in cases.columns
    ]

    print_table(
        f"Top {top_n} native features",
        native.loc[:, native_columns].head(top_n),
    )
    print_table(
        f"Top {top_n} permutation features",
        permutation.loc[:, permutation_columns].head(top_n),
    )
    print_table(
        f"Top {top_n} SHAP features",
        shap.loc[:, shap_columns].head(top_n),
    )
    print_table(
        "Global rank comparison",
        ranks.loc[:, rank_columns].head(top_n),
    )
    print_table(
        "Representative cases",
        cases.loc[:, case_columns],
    )

    if {"CaseIndex", "Rule", "Weight"}.issubset(lime.columns):
        case_types = cases.loc[:, ["CaseIndex", "CaseType"]]

        lime_summary = (
            lime.merge(
                case_types,
                on="CaseIndex",
                how="left",
                validate="many_to_one",
            )
            .assign(
                AbsoluteWeight=lambda frame: frame["Weight"].abs()
            )
            .sort_values(
                ["CaseIndex", "AbsoluteWeight"],
                ascending=[True, False],
            )
            .groupby("CaseIndex", as_index=False)
            .head(3)
        )

        print_table(
            "Three strongest LIME rules per representative case",
            lime_summary.loc[
                :,
                [
                    "CaseType",
                    "CaseIndex",
                    "Rule",
                    "Weight",
                ],
            ],
        )

    print("\nDemo completed successfully.")
    print(f"Generated output: {output_dir}")
    print(
        "The canonical results under results/heart were not overwritten."
    )
    print(
        "Repeated-fit stability was not rerun because the saved bundle does "
        "not contain the notebook retraining callback."
    )


def find_created_output(existing_directories, started_at):
    if not CLI_RUNS_DIR.is_dir():
        raise RuntimeError("The CLI output directory was not created.")

    candidates = [
        path
        for path in CLI_RUNS_DIR.iterdir()
        if (
            path.is_dir()
            and path.name.startswith("heart_")
            and path not in existing_directories
            and path.stat().st_mtime >= started_at
        )
    ]

    if not candidates:
        raise RuntimeError(
            "The Heart run completed but its new output directory could not "
            "be identified."
        )

    return max(candidates, key=lambda path: path.stat().st_mtime)


def main():
    args = build_parser().parse_args()

    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(
            f"Heart configuration was not found: {CONFIG_PATH}"
        )

    print("Heart XAI framework demonstration")
    print(f"Configuration: {CONFIG_PATH}")

    if not args.run:
        exit_code = cli_main(
            [
                "--config",
                str(CONFIG_PATH),
                "--validate-only",
            ]
        )
        if exit_code != 0:
            raise SystemExit(exit_code)

        print("\nValidation demo completed successfully.")
        print("The serialized model was not loaded and no results were written.")
        print(
            "Run again with --run only if the saved Heart model is trusted."
        )
        return

    existing_directories = (
        set(CLI_RUNS_DIR.iterdir())
        if CLI_RUNS_DIR.is_dir()
        else set()
    )
    started_at = time.time()

    exit_code = cli_main(
        [
            "--config",
            str(CONFIG_PATH),
            "--trust-model",
        ]
    )
    if exit_code != 0:
        raise SystemExit(exit_code)

    output_dir = find_created_output(
        existing_directories,
        started_at,
    )
    summarize_results(output_dir, args.top)


if __name__ == "__main__":
    main()