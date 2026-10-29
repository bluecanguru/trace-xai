"""Optional HbA1c bootstrap-refit and grouped-permutation experiment.

This experiment is deliberately separate from the canonical XAI pipeline. It
reads the trusted HbA1c bundle and creates a new timestamped output directory.
It never edits the bundle, the HbA1c notebook, or results/hba1c.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import warnings
from collections import OrderedDict
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
import statsmodels
import statsmodels.api as sm
from scipy.stats import spearmanr
from sklearn.metrics import log_loss, roc_auc_score


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_BUNDLE = PROJECT_ROOT / "results" / "hba1c" / "bundle"
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "results" / "optional_hba1c_grouped_stability"
)


def _json_default(value):
    """Convert a supported nonstandard value for JSON serialisation.

    Args:
        value: NumPy scalar or path-like value encountered by ``json.dumps``.

    Returns:
        Native numeric scalar or string path representation.

    Raises:
        TypeError: If the value has no supported conversion.
    """
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialise {type(value).__name__}")


def _write_json(path: Path, data: dict) -> None:
    """Write an indented UTF-8 JSON document using scientific conversions.

    Args:
        path: Destination file path.
        data: Mapping to serialise.

    Returns:
        None.

    Raises:
        OSError: If the destination cannot be written.
        TypeError: If an unsupported value cannot be serialised.
    """
    path.write_text(
        json.dumps(data, indent=2, default=_json_default), encoding="utf-8"
    )


def ensure_isolated_output(bundle_dir: Path, output_root: Path) -> None:
    """Ensure optional output cannot overwrite canonical HbA1c results.

    Args:
        bundle_dir: Canonical trusted-bundle directory.
        output_root: Proposed parent directory for optional experiment runs.

    Returns:
        None.

    Raises:
        ValueError: If either path contains the other or both resolve to the
            same canonical result location.
    """
    canonical_results = bundle_dir.resolve().parent
    output_root = output_root.resolve()
    if (
        output_root == canonical_results
        or canonical_results in output_root.parents
        or output_root in canonical_results.parents
    ):
        raise ValueError(
            "Output must be separate from results/hba1c and cannot contain it."
        )


def validate_bundle_files(bundle_dir: Path) -> dict:
    """Validate the required files and metadata of an HbA1c bundle.

    Args:
        bundle_dir: Directory containing the saved model and aligned CSV files.

    Returns:
        Validated metadata mapping.

    Raises:
        ValueError: If files are missing or dataset and feature metadata are
            inconsistent.
    """
    required = (
        "model.pkl",
        "metadata.json",
        "X_train.csv",
        "X_test.csv",
        "y_train.csv",
        "y_test.csv",
    )
    missing = [name for name in required if not (bundle_dir / name).is_file()]
    if missing:
        raise ValueError(f"Bundle is missing files: {missing}")
    metadata = json.loads(
        (bundle_dir / "metadata.json").read_text(encoding="utf-8")
    )
    if metadata.get("dataset") != "hba1c":
        raise ValueError("The supplied bundle is not labelled as the HbA1c dataset.")
    names = metadata.get("feature_names")
    if not isinstance(names, list) or not names or len(names) != len(set(names)):
        raise ValueError("metadata.feature_names must be a unique non-empty list.")
    if metadata.get("n_features") != len(names):
        raise ValueError("metadata.n_features differs from feature_names.")
    return metadata


def load_bundle(bundle_dir: Path):
    """Load a trusted HbA1c bundle and validate its aligned design matrices.

    Args:
        bundle_dir: Directory containing the trusted saved bundle.

    Returns:
        Tuple of metadata, fitted Statsmodels result, and aligned train/test
        matrices and labels.

    Raises:
        ValueError: If schemas, values, labels, model class, or design terms do
            not agree with the metadata.

    Warning:
        ``model.pkl`` uses pickle-based deserialisation and must come from a
        trusted local source.
    """
    metadata = validate_bundle_files(bundle_dir)
    feature_names = metadata["feature_names"]
    data = {}
    for split in ("train", "test"):
        X = pd.read_csv(
            bundle_dir / f"X_{split}.csv", float_precision="round_trip"
        )
        y_frame = pd.read_csv(
            bundle_dir / f"y_{split}.csv", float_precision="round_trip"
        )
        if X.columns.tolist() != feature_names:
            raise ValueError(f"X_{split} columns differ from bundle metadata.")
        if X.empty or y_frame.shape[1] != 1 or len(X) != len(y_frame):
            raise ValueError(f"Invalid X/y dimensions for {split}.")
        X_array = X.to_numpy(dtype=float)
        if not np.isfinite(X_array).all():
            raise ValueError(f"X_{split} contains non-finite values.")
        y = y_frame.iloc[:, 0].to_numpy(dtype=float)
        if not set(np.unique(y)).issubset({0.0, 1.0}):
            raise ValueError(f"y_{split} must contain binary 0/1 labels.")
        data[f"X_{split}"] = X_array
        data[f"y_{split}"] = y

    model = joblib.load(bundle_dir / "model.pkl")
    if type(model).__name__ != metadata.get("model_class"):
        raise ValueError("Loaded model class differs from bundle metadata.")
    if list(model.model.exog_names) != feature_names:
        raise ValueError("Saved model terms differ from the matrix columns.")
    return metadata, model, data


def extract_term_groups(model, feature_names) -> OrderedDict[str, tuple[int, ...]]:
    """Extract complete non-intercept Patsy terms for grouped permutation.

    Args:
        model: Fitted Statsmodels result retaining Patsy design information.
        feature_names: Ordered saved design-matrix column names.

    Returns:
        Ordered mapping from original Patsy term names to design-column indices.

    Raises:
        ValueError: If term slices are unavailable, empty, overlapping, out of
            range, or fail to cover every non-intercept design column.
    """
    design_info = getattr(getattr(model.model, "data", None), "design_info", None)
    term_slices = getattr(design_info, "term_name_slices", None)
    if not term_slices:
        raise ValueError(
            "The model does not retain Patsy term slices; grouped permutation "
            "cannot be reconstructed safely."
        )

    groups: OrderedDict[str, tuple[int, ...]] = OrderedDict()
    covered = set()
    for term_name, term_slice in term_slices.items():
        indices = tuple(range(term_slice.start, term_slice.stop))
        if term_name.strip().lower() == "intercept":
            continue
        if not indices:
            raise ValueError(f"Empty design slice for term {term_name!r}.")
        if min(indices) < 0 or max(indices) >= len(feature_names):
            raise ValueError(f"Invalid design slice for term {term_name!r}.")
        if covered.intersection(indices):
            raise ValueError("Patsy term slices overlap.")
        covered.update(indices)
        groups[str(term_name)] = indices

    non_intercept = {
        index
        for index, name in enumerate(feature_names)
        if name.strip().lower() != "intercept"
    }
    if covered != non_intercept:
        missing = sorted(non_intercept - covered)
        extra = sorted(covered - non_intercept)
        raise ValueError(
            f"Term slices do not cover the non-intercept design columns; "
            f"missing={missing}, extra={extra}."
        )
    return groups


def predict_probabilities(model, X: np.ndarray) -> np.ndarray:
    """Return validated positive-class probabilities from Statsmodels.

    Args:
        model: Fitted Statsmodels binary-result object.
        X: Design matrix aligned with the fitted specification.

    Returns:
        One-dimensional probability array clipped away from zero and one.

    Raises:
        ValueError: If predictions have the wrong length or contain nonfinite
            values.
    """
    try:
        probabilities = model.predict(X, transform=False)
    except TypeError:
        probabilities = model.predict(X)
    probabilities = np.asarray(probabilities, dtype=float).reshape(-1)
    if len(probabilities) != len(X) or not np.isfinite(probabilities).all():
        raise ValueError("Model returned invalid probabilities.")
    return np.clip(probabilities, 1e-12, 1 - 1e-12)


def prediction_score(y, probabilities, scoring: str) -> float:
    """Evaluate probabilities with ROC AUC or negative log loss.

    Args:
        y: Observed binary outcomes.
        probabilities: Positive-class probabilities aligned with ``y``.
        scoring: Either ``"roc_auc"`` or ``"neg_log_loss"``.

    Returns:
        Continuous performance score, oriented so larger values are better.

    Raises:
        ValueError: If the scoring name is unsupported or the metric cannot be
            evaluated from the supplied outcomes.
    """
    if scoring == "roc_auc":
        return float(roc_auc_score(y, probabilities))
    if scoring == "neg_log_loss":
        return float(-log_loss(y, probabilities, labels=[0.0, 1.0]))
    raise ValueError("scoring must be 'roc_auc' or 'neg_log_loss'.")


def grouped_permutation_importance(
    model,
    X,
    y,
    groups,
    *,
    n_repeats: int,
    scoring: str,
    random_seed: int,
) -> pd.DataFrame:
    """Measure grouped permutation importance for complete Patsy terms.

    Args:
        model: Fitted Statsmodels result to evaluate.
        X: Fixed evaluation design matrix.
        y: Outcomes aligned with ``X``.
        groups: Ordered term-to-column-index mapping.
        n_repeats: Joint permutations performed for each term.
        scoring: Continuous scoring rule used to measure performance loss.
        random_seed: Seed controlling the permutation order.

    Returns:
        DataFrame containing term size, mean importance, variability, baseline
        score, scorer, and repeat count, sorted by decreasing importance.

    Raises:
        ValueError: If repeat count or input dimensions are invalid.

    Notes:
        Columns belonging to one encoded term are permuted together, preserving
        their internal relationship while breaking their association with the
        outcome.
    """
    if n_repeats < 1:
        raise ValueError("n_repeats must be at least 1.")
    base = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if base.ndim != 2 or len(base) != len(y):
        raise ValueError("X and y dimensions are incompatible.")
    working = base.copy()
    baseline = prediction_score(y, predict_probabilities(model, base), scoring)
    rng = np.random.default_rng(random_seed)
    rows = []

    for group_name, column_indices in groups.items():
        columns = np.asarray(column_indices, dtype=int)
        values = []
        for _ in range(n_repeats):
            permutation = rng.permutation(len(base))
            working[:, columns] = base[permutation][:, columns]
            permuted_score = prediction_score(
                y, predict_probabilities(model, working), scoring
            )
            values.append(baseline - permuted_score)
            working[:, columns] = base[:, columns]
        rows.append(
            {
                "Group": group_name,
                "DesignColumns": len(columns),
                "Importance": float(np.mean(values)),
                "ImportanceStd": float(np.std(values, ddof=1))
                if len(values) > 1
                else 0.0,
                "BaselineScore": baseline,
                "Scoring": scoring,
                "Repeats": n_repeats,
            }
        )
    return pd.DataFrame(rows).sort_values("Importance", ascending=False)


def stratified_bootstrap_indices(y, rng) -> np.ndarray:
    """Draw class-stratified bootstrap indices at original class sizes.

    Args:
        y: Binary outcomes to stratify.
        rng: NumPy random generator controlling sampling and shuffling.

    Returns:
        Shuffled integer indices sampled with replacement within each class.
    """
    y = np.asarray(y)
    sampled = []
    for label in np.unique(y):
        class_indices = np.flatnonzero(y == label)
        sampled.append(
            rng.choice(class_indices, size=len(class_indices), replace=True)
        )
    indices = np.concatenate(sampled)
    rng.shuffle(indices)
    return indices


def fit_bootstrap_logit(X, y, start_params, maxiter: int):
    """Fit one bootstrap logistic model with the canonical specification.

    Args:
        X: Bootstrapped design matrix.
        y: Bootstrapped binary outcomes.
        start_params: Initial coefficient vector supplied to the optimiser.
        maxiter: Maximum Newton iterations.

    Returns:
        Tuple containing the fitted result, convergence flag, elapsed seconds,
        and captured warning messages.
    """
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fitted = sm.Logit(y, X).fit(
            start_params=start_params,
            method="newton",
            maxiter=maxiter,
            disp=False,
            full_output=True,
            warn_convergence=True,
        )
    elapsed = time.perf_counter() - start
    converged = bool(fitted.mle_retvals.get("converged", False))
    messages = [str(item.message) for item in caught]
    return fitted, converged, elapsed, messages


def summarise_bootstrap(raw: pd.DataFrame, top_k: int):
    """Summarise term rankings and pairwise agreement across successful refits.

    Args:
        raw: Bootstrap-level grouped-permutation importance rows.
        top_k: Number of leading terms used for persistence and overlap.

    Returns:
        Pair of DataFrames containing term-level stability summaries and all
        pairwise Spearman/top-k comparisons. Empty input yields two empty tables.
    """
    if raw.empty:
        return pd.DataFrame(), pd.DataFrame()
    raw = raw.copy()
    raw["Rank"] = raw.groupby("Bootstrap")["Importance"].rank(
        method="average", ascending=False
    )
    k = min(top_k, raw["Group"].nunique())
    raw["InTopK"] = raw.groupby("Bootstrap")["Rank"].transform(
        lambda ranks: ranks <= k
    )
    summary = (
        raw.groupby("Group")
        .agg(
            DesignColumns=("DesignColumns", "first"),
            MeanImportance=("Importance", "mean"),
            StdImportance=("Importance", "std"),
            MeanRank=("Rank", "mean"),
            StdRank=("Rank", "std"),
            TopKFrequency=("InTopK", "mean"),
        )
        .reset_index()
        .sort_values(["MeanRank", "MeanImportance"], ascending=[True, False])
    )

    pairwise_rows = []
    bootstrap_ids = sorted(raw["Bootstrap"].unique())
    for first, second in combinations(bootstrap_ids, 2):
        first_ranks = raw.loc[
            raw["Bootstrap"] == first, ["Group", "Rank"]
        ].rename(columns={"Rank": "Rank_A"})
        second_ranks = raw.loc[
            raw["Bootstrap"] == second, ["Group", "Rank"]
        ].rename(columns={"Rank": "Rank_B"})
        merged = first_ranks.merge(second_ranks, on="Group", validate="one_to_one")
        rho = spearmanr(merged["Rank_A"], merged["Rank_B"]).correlation
        top_first = set(merged.nsmallest(k, "Rank_A")["Group"])
        top_second = set(merged.nsmallest(k, "Rank_B")["Group"])
        pairwise_rows.append(
            {
                "Bootstrap_A": int(first),
                "Bootstrap_B": int(second),
                "SpearmanRankCorrelation": float(rho),
                "TopK": k,
                "TopKOverlap": len(top_first.intersection(top_second)) / k,
            }
        )
    return summary, pd.DataFrame(pairwise_rows)


def make_run_directory(output_root: Path, mode: str) -> Path:
    """Create a unique timestamped directory for one optional run.

    Args:
        output_root: Parent directory reserved for optional experiments.
        mode: Run mode included in the directory name.

    Returns:
        Newly created run directory.

    Raises:
        OSError: If the parent or unique run directory cannot be created.
    """
    output_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir = output_root / f"{mode}_{stamp}"
    run_dir.mkdir(exist_ok=False)
    return run_dir


def save_progress(run_dir, raw_rows, run_rows, top_k):
    """Persist current optional-run tables and recompute stability summaries.

    Args:
        run_dir: Isolated run directory.
        raw_rows: Completed grouped-importance DataFrames.
        run_rows: Per-bootstrap status and timing records.
        top_k: Number of leading terms used by the summary.

    Returns:
        Raw importance, run-log, feature-summary, and pairwise DataFrames.
    """
    raw = pd.concat(raw_rows, ignore_index=True) if raw_rows else pd.DataFrame()
    run_log = pd.DataFrame(run_rows)
    raw.to_csv(run_dir / "bootstrap_grouped_importance.csv", index=False)
    run_log.to_csv(run_dir / "bootstrap_run_log.csv", index=False)
    summary, pairwise = summarise_bootstrap(raw, top_k)
    summary.to_csv(run_dir / "bootstrap_grouped_summary.csv", index=False)
    pairwise.to_csv(run_dir / "bootstrap_pairwise_stability.csv", index=False)
    return raw, run_log, summary, pairwise


def assess_pilot(
    run_log,
    pairwise,
    *,
    requested,
    full_target,
    max_projected_hours,
):
    """Assess whether a pilot justifies attempting the requested full run.

    Args:
        run_log: Per-bootstrap status, convergence, and timing table.
        pairwise: Pairwise ranking-stability table from successful refits.
        requested: Number of bootstraps requested in the pilot.
        full_target: Bootstrap count planned for a full run.
        max_projected_hours: Maximum acceptable projected duration.

    Returns:
        Decision mapping containing convergence, failures, median timings,
        projected runtime, and the exploratory mean pairwise correlation.

    Notes:
        The recommendation concerns computational feasibility only and does not
        establish statistical convergence of the stability estimate.
    """
    successful = int((run_log.get("Status") == "success").sum())
    converged = int(run_log.get("Converged", pd.Series(dtype=bool)).fillna(False).sum())
    convergence_rate = converged / requested if requested else 0.0
    successful_rows = run_log.loc[run_log.get("Status") == "success"]
    if successful_rows.empty:
        median_fit = None
        median_permutation = None
        median_total = None
        projected_hours = None
    else:
        median_fit = float(successful_rows["FitSeconds"].median())
        median_permutation = float(successful_rows["PermutationSeconds"].median())
        median_total = float(successful_rows["TotalSeconds"].median())
        projected_hours = median_total * full_target / 3600

    acceptable = (
        successful == requested
        and converged == requested
        and projected_hours is not None
        and projected_hours <= max_projected_hours
    )
    mean_rho = None
    if not pairwise.empty:
        mean_rho = float(pairwise["SpearmanRankCorrelation"].mean())
    return {
        "decision": "PROCEED_TO_FULL_RUN"
        if acceptable
        else "KEEP_CANONICAL_RESULTS_AND_REVIEW_PILOT",
        "decision_scope": (
            "Computational feasibility only: optimiser convergence, failures, "
            "and projected runtime. A small pilot does not establish statistical "
            "convergence of the stability estimates."
        ),
        "requested_bootstraps": requested,
        "successful_bootstraps": successful,
        "failed_bootstraps": requested - successful,
        "converged_fits": converged,
        "convergence_rate": convergence_rate,
        "median_fit_seconds": median_fit,
        "median_grouped_permutation_seconds": median_permutation,
        "median_total_seconds_per_bootstrap": median_total,
        "projected_full_bootstraps": full_target,
        "projected_full_runtime_hours": projected_hours,
        "maximum_acceptable_projected_hours": max_projected_hours,
        "pilot_mean_pairwise_spearman": mean_rho,
        "canonical_results_changed": False,
    }


def runtime_metadata():
    """Collect timestamped Python, platform, and dependency provenance.

    Returns:
        Mapping suitable for inclusion in the optional run manifest.
    """
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__,
            "statsmodels": statsmodels.__version__,
            "joblib": joblib.__version__,
        },
    }


def parse_args(argv=None):
    """Parse and validate optional HbA1c experiment arguments.

    Args:
        argv: Optional command-line argument sequence. ``None`` reads the
            process arguments.

    Returns:
        Populated argument namespace with mode-specific defaults applied.

    Raises:
        SystemExit: If arguments are invalid or full mode lacks explicit
            confirmation.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pilot", "full"), default="pilot")
    parser.add_argument("--confirm-full", action="store_true")
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Validate the trusted bundle and term groups without writing or fitting.",
    )
    parser.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--n-bootstrap", type=int)
    parser.add_argument("--bootstrap-repeats", type=int)
    parser.add_argument("--current-repeats", type=int)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--permutation-seed", type=int, default=42)
    parser.add_argument("--maxiter", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--full-bootstrap-target", type=int, default=100)
    parser.add_argument("--max-projected-hours", type=float, default=12.0)
    args = parser.parse_args(argv)
    if args.mode == "full" and not args.confirm_full:
        parser.error("Full mode requires --confirm-full.")
    defaults = {
        "pilot": (5, 3, 5),
        "full": (100, 10, 20),
    }
    default_bootstraps, default_bootstrap_repeats, default_current_repeats = (
        defaults[args.mode]
    )
    args.n_bootstrap = args.n_bootstrap or default_bootstraps
    args.bootstrap_repeats = args.bootstrap_repeats or default_bootstrap_repeats
    args.current_repeats = args.current_repeats or default_current_repeats
    for name in (
        "n_bootstrap",
        "bootstrap_repeats",
        "current_repeats",
        "maxiter",
        "top_k",
        "full_bootstrap_target",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1.")
    return args


def run(args) -> Path | None:
    """Execute or inspect the isolated grouped-permutation experiment.

    Args:
        args: Validated namespace returned by :func:`parse_args`.

    Returns:
        Created run directory, or ``None`` for inspection-only mode.

    Raises:
        ValueError: If bundle contents, term groups, settings, or output paths
            are invalid.
        OSError: If required inputs cannot be read or outputs cannot be written.

    Notes:
        Canonical HbA1c results are never modified. Progress is saved after
        every attempted refit so interrupted optional runs remain auditable.
    """
    bundle_dir = args.bundle_dir.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    ensure_isolated_output(bundle_dir, output_root)
    metadata, reference_model, data = load_bundle(bundle_dir)
    feature_names = metadata["feature_names"]
    groups = extract_term_groups(reference_model, feature_names)

    print(f"Trusted bundle: {bundle_dir}")
    print(f"Rows: {len(data['X_train']):,}; design columns: {len(feature_names)}")
    print(f"Grouped non-intercept Patsy terms: {len(groups)}")
    print(f"Canonical results protected: {bundle_dir.parent}")
    if args.inspect_only:
        for term, indices in groups.items():
            print(f"  {term}: {len(indices)} design column(s)")
        print("Inspection passed. No output directory was created and no model was fitted.")
        return None

    run_dir = make_run_directory(output_root, args.mode)
    term_rows = []
    for term, indices in groups.items():
        term_rows.append(
            {
                "Group": term,
                "DesignColumns": len(indices),
                "ColumnIndices": ";".join(map(str, indices)),
                "ColumnNames": " | ".join(feature_names[index] for index in indices),
            }
        )
    pd.DataFrame(term_rows).to_csv(run_dir / "term_groups.csv", index=False)

    manifest = {
        "status": "running",
        "mode": args.mode,
        "source_bundle": str(bundle_dir),
        "output_directory": str(run_dir),
        "canonical_results_directory": str(bundle_dir.parent),
        "canonical_results_changed": False,
        "evaluation_design": (
            "The saved HbA1c analytical configuration uses the same full cohort "
            "as its training and explanation/evaluation matrix. These optional "
            "results are therefore in-sample and are not directly comparable "
            "with held-out stability estimates."
        ),
        "settings": {
            "n_bootstrap": args.n_bootstrap,
            "bootstrap_repeats": args.bootstrap_repeats,
            "current_model_repeats": args.current_repeats,
            "bootstrap_scoring": "neg_log_loss",
            "current_model_scoring": "roc_auc",
            "bootstrap_seed": args.bootstrap_seed,
            "permutation_seed": args.permutation_seed,
            "maxiter": args.maxiter,
            "top_k": args.top_k,
        },
        "runtime": runtime_metadata(),
    }
    _write_json(run_dir / "run_metadata.json", manifest)

    try:
        current_start = time.perf_counter()
        current = grouped_permutation_importance(
            reference_model,
            data["X_test"],
            data["y_test"],
            groups,
            n_repeats=args.current_repeats,
            scoring="roc_auc",
            random_seed=args.permutation_seed,
        )
        current_seconds = time.perf_counter() - current_start
        current.to_csv(
            run_dir / "grouped_permutation_current_model.csv", index=False
        )

        master_rng = np.random.default_rng(args.bootstrap_seed)
        raw_rows = []
        run_rows = []
        start_params = np.asarray(reference_model.params, dtype=float)

        for bootstrap_id in range(1, args.n_bootstrap + 1):
            sample_seed = int(master_rng.integers(0, np.iinfo(np.uint32).max))
            sample_rng = np.random.default_rng(sample_seed)
            indices = stratified_bootstrap_indices(data["y_train"], sample_rng)
            X_bootstrap = data["X_train"][indices]
            y_bootstrap = data["y_train"][indices]
            run_row = {
                "Bootstrap": bootstrap_id,
                "BootstrapSampleSeed": sample_seed,
                "Status": "failed",
                "Converged": False,
                "FitSeconds": np.nan,
                "PermutationSeconds": np.nan,
                "TotalSeconds": np.nan,
                "Iterations": np.nan,
                "Warnings": "",
                "Error": "",
            }
            total_start = time.perf_counter()
            try:
                fitted, converged, fit_seconds, fit_warnings = fit_bootstrap_logit(
                    X_bootstrap,
                    y_bootstrap,
                    start_params=start_params,
                    maxiter=args.maxiter,
                )
                run_row["FitSeconds"] = fit_seconds
                run_row["Converged"] = converged
                run_row["Iterations"] = fitted.mle_retvals.get("iterations", np.nan)
                run_row["Warnings"] = " | ".join(fit_warnings)
                if not converged:
                    raise RuntimeError("Statsmodels optimiser did not converge.")

                permutation_start = time.perf_counter()
                importance = grouped_permutation_importance(
                    fitted,
                    data["X_test"],
                    data["y_test"],
                    groups,
                    n_repeats=args.bootstrap_repeats,
                    scoring="neg_log_loss",
                    random_seed=args.permutation_seed,
                )
                run_row["PermutationSeconds"] = (
                    time.perf_counter() - permutation_start
                )
                importance["Bootstrap"] = bootstrap_id
                importance["BootstrapSampleSeed"] = sample_seed
                raw_rows.append(importance)
                run_row["Status"] = "success"
            except Exception as exc:
                run_row["Error"] = f"{type(exc).__name__}: {exc}"
            finally:
                run_row["TotalSeconds"] = time.perf_counter() - total_start
                run_rows.append(run_row)
                _, run_log, _, pairwise = save_progress(
                    run_dir, raw_rows, run_rows, args.top_k
                )
                print(
                    f"Bootstrap {bootstrap_id}/{args.n_bootstrap}: "
                    f"{run_row['Status']} ({run_row['TotalSeconds']:.1f} s)"
                )

        assessment = assess_pilot(
            run_log,
            pairwise,
            requested=args.n_bootstrap,
            full_target=args.full_bootstrap_target,
            max_projected_hours=args.max_projected_hours,
        )
        assessment["current_model_grouped_permutation_seconds"] = current_seconds
        _write_json(run_dir / "pilot_assessment.json", assessment)
        manifest["status"] = "completed"
        manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["assessment"] = assessment
        _write_json(run_dir / "run_metadata.json", manifest)
        print(f"Completed isolated {args.mode} run: {run_dir}")
        print(f"Decision: {assessment['decision']}")
        return run_dir
    except BaseException as exc:
        manifest["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        _write_json(run_dir / "run_metadata.json", manifest)
        raise


def main(argv=None):
    """Run the optional experiment command-line interface.

    Args:
        argv: Optional argument sequence passed to :func:`parse_args`.

    Returns:
        Zero after successful inspection or execution and one after a handled
        experiment error.
    """
    args = parse_args(argv)
    try:
        run(args)
        return 0
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
