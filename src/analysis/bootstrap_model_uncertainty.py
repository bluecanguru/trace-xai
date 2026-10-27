"""Bootstrap uncertainty for the four held-out thesis models.

Run from the thesis repository root with the same trusted Windows Python
environment that created the bundles. The script loads only the user's own
saved models and never overwrites canonical dataset results.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


DATASETS = ("heart", "stroke", "diabetes", "ckd")
DISPLAY_NAMES = {
    "heart": "Heart",
    "stroke": "Stroke",
    "diabetes": "Diabetes",
    "ckd": "CKD",
}
METRIC_ORDER = (
    "Accuracy",
    "Precision",
    "Recall",
    "Specificity",
    "F1",
    "ROC_AUC",
    "PR_AUC",
)


def specificity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return specificity for binary labels, or NaN when no negatives exist."""
    negatives = y_true == 0
    if not negatives.any():
        return float("nan")
    return float(np.mean(y_pred[negatives] == 0))


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, float]:
    """Calculate the supported binary predictive-performance metrics."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "Specificity": specificity_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_true, y_prob),
        "PR_AUC": average_precision_score(y_true, y_prob),
    }


def bootstrap_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> tuple[dict[str, float], dict[str, tuple[float, float]], int]:
    """Return point estimates and two-sided 95% percentile intervals."""
    point = calculate_metrics(y_true, y_pred, y_prob)
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {name: [] for name in METRIC_ORDER}
    n = len(y_true)
    valid = 0

    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        sampled_true = y_true[indices]

        # ROC AUC and PR AUC require both classes. With the current test sets,
        # a single-class resample is extremely unlikely, but it is rejected
        # explicitly rather than assigned an artificial value.
        if np.unique(sampled_true).size != 2:
            continue

        values = calculate_metrics(
            sampled_true,
            y_pred[indices],
            y_prob[indices],
        )
        for name, value in values.items():
            if np.isfinite(value):
                samples[name].append(float(value))
        valid += 1

    if valid == 0:
        raise RuntimeError("No valid bootstrap samples contained both classes.")

    intervals = {
        name: tuple(np.percentile(values, [2.5, 97.5]).astype(float))
        for name, values in samples.items()
    }
    return point, intervals, valid


def load_predictions(bundle_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a trusted model bundle and return aligned labels and predictions."""
    x_test = pd.read_csv(bundle_dir / "X_test.csv")
    y_true = pd.read_csv(bundle_dir / "y_test.csv").iloc[:, 0].to_numpy(dtype=int)
    model = joblib.load(bundle_dir / "model.pkl")

    # Preserve DataFrame names when the fitted estimator recorded them; use an
    # array for estimators fitted without names. In both cases, the bundle CSV
    # retains the feature order recorded in metadata.json.
    model_input = x_test if hasattr(model, "feature_names_in_") else x_test.to_numpy()
    y_pred = np.asarray(model.predict(model_input), dtype=int)
    probabilities = np.asarray(model.predict_proba(model_input), dtype=float)

    classes = np.asarray(model.classes_)
    positive_positions = np.flatnonzero(classes == 1)
    if positive_positions.size != 1:
        raise ValueError(f"Expected exactly one positive class in {classes!r}.")
    y_prob = probabilities[:, int(positive_positions[0])]

    if not (len(y_true) == len(y_pred) == len(y_prob)):
        raise ValueError("Prediction and label lengths are not aligned.")
    return y_true, y_pred, y_prob


def format_interval(point: float, low: float, high: float) -> str:
    """Format a point estimate and confidence interval for reporting."""
    return f"{point:.3f} [{low:.3f}, {high:.3f}]"


def dataframe_to_latex(summary: pd.DataFrame) -> str:
    """Render the uncertainty summary as a compact LaTeX table."""
    columns = ["Dataset", *METRIC_ORDER]
    header = " & ".join(name.replace("_", " ") for name in columns) + r" \\"
    rows = [header, r"\midrule"]
    for _, row in summary.loc[:, columns].iterrows():
        rows.append(" & ".join(str(row[name]) for name in columns) + r" \\")
    return "\n".join(
        [
            r"\begin{table}[htbp]",
            r"    \centering",
            r"    \scriptsize",
            r"    \caption{Held-out predictive performance with 95\% bootstrap confidence intervals.}",
            r"    \label{tab:bootstrap_performance}",
            r"    \resizebox{\textwidth}{!}{%",
            r"    \begin{tabular}{@{}lrrrrrrr@{}}",
            r"        \toprule",
            "        " + rows[0],
            "        " + rows[1],
            *("        " + item for item in rows[2:]),
            r"        \bottomrule",
            r"    \end{tabular}%",
            r"    }",
            r"\end{table}",
        ]
    )


def export_roc_auc_figure(long_df: pd.DataFrame, output_path: Path) -> None:
    """Export ROC-AUC estimates and percentile intervals as a thesis figure."""
    roc = long_df.loc[long_df["Metric"] == "ROC_AUC"].copy()
    roc["Dataset"] = roc["Dataset"].map(DISPLAY_NAMES)
    roc = roc.set_index("Dataset").loc[list(DISPLAY_NAMES.values())].reset_index()

    estimates = roc["Estimate"].to_numpy(dtype=float)
    lower = roc["CI95_Lower"].to_numpy(dtype=float)
    upper = roc["CI95_Upper"].to_numpy(dtype=float)
    yerr = np.vstack([estimates - lower, upper - estimates])

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    positions = np.arange(len(roc))
    ax.errorbar(
        positions,
        estimates,
        yerr=yerr,
        fmt="o",
        color="#506F5C",
        ecolor="#789080",
        elinewidth=1.6,
        capsize=5,
        markersize=7,
    )
    ax.set_xticks(positions, roc["Dataset"])
    ax.set_ylabel("ROC AUC")
    ax.set_ylim(0.70, 1.02)
    ax.grid(axis="y", color="#D9DFDA", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Run the four-dataset bootstrap analysis and export tables and figures."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Thesis repository root; defaults to the current directory.",
    )
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results") / "uncertainty",
    )
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    long_rows: list[dict[str, object]] = []
    wide_rows: list[dict[str, object]] = []
    run_details: dict[str, object] = {
        "method": "nonparametric observation-level percentile bootstrap",
        "confidence_level": 0.95,
        "n_requested": args.n_bootstrap,
        "seed": args.seed,
        "scope": "test-sample uncertainty conditional on each fitted model",
        "datasets": {},
    }

    for offset, dataset in enumerate(DATASETS):
        bundle_dir = project_root / "results" / dataset / "bundle"
        y_true, y_pred, y_prob = load_predictions(bundle_dir)
        point, intervals, valid = bootstrap_intervals(
            y_true,
            y_pred,
            y_prob,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed + offset,
        )

        wide: dict[str, object] = {"Dataset": DISPLAY_NAMES[dataset]}
        for metric in METRIC_ORDER:
            low, high = intervals[metric]
            wide[metric] = format_interval(point[metric], low, high)
            long_rows.append(
                {
                    "Dataset": dataset,
                    "Metric": metric,
                    "Estimate": point[metric],
                    "CI95_Lower": low,
                    "CI95_Upper": high,
                    "ValidBootstrapSamples": valid,
                }
            )
        wide_rows.append(wide)
        run_details["datasets"][dataset] = {
            "n_test": int(len(y_true)),
            "n_positive": int(np.sum(y_true == 1)),
            "n_negative": int(np.sum(y_true == 0)),
            "valid_bootstrap_samples": valid,
        }

    long_df = pd.DataFrame(long_rows)
    wide_df = pd.DataFrame(wide_rows)
    long_df.to_csv(output_dir / "bootstrap_metrics_long.csv", index=False)
    wide_df.to_csv(output_dir / "bootstrap_metrics_table.csv", index=False)
    (output_dir / "bootstrap_metrics_table.tex").write_text(
        dataframe_to_latex(wide_df),
        encoding="utf-8",
    )
    (output_dir / "bootstrap_run.json").write_text(
        json.dumps(run_details, indent=2),
        encoding="utf-8",
    )
    export_roc_auc_figure(
        long_df,
        project_root / "images" / "results" / "bootstrap_roc_auc.pdf",
    )

    print(wide_df.to_string(index=False))
    print(f"\nExported uncertainty analysis to: {output_dir}")


if __name__ == "__main__":
    main()