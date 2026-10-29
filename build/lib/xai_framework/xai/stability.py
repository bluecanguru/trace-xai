
"""Measure permutation-ranking persistence across bootstrap model refits."""

import numpy as np
import pandas as pd

from itertools import combinations
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance


def _take_rows(values, indices):
    """Select rows while preserving pandas metadata when available.

    Args:
        values: Pandas object or NumPy-compatible array.
        indices: Integer positional indices to select.

    Returns:
        Independent row subset of the same broad representation as ``values``.
    """
    if hasattr(values, "iloc"):
        return values.iloc[indices].copy()
    return np.asarray(values)[indices].copy()


def _stratified_bootstrap_indices(y, rng):
    """Draw class-stratified bootstrap indices.

    Args:
        y: One-dimensional class labels.
        rng: NumPy random generator controlling the resample and final shuffle.

    Returns:
        Integer array containing one with-replacement sample per observed class,
        each retaining that class's original sample size.
    """
    y_array = np.asarray(y)
    sampled = []
    for label in np.unique(y_array):
        class_indices = np.flatnonzero(y_array == label)
        sampled.append(
            rng.choice(class_indices, size=len(class_indices), replace=True)
        )
    indices = np.concatenate(sampled)
    rng.shuffle(indices)
    return indices


def run_stability_analysis(
    train_model_fn,
    X_train,
    X_test,
    y_train,
    y_test,
    feature_names,
    n_bootstrap=100,
    bootstrap_seed=42,
    scoring="neg_log_loss",
    n_repeats=10,
    permutation_seed=42,
    top_k=3,
):
    """Evaluate permutation-ranking stability across bootstrap model refits.

    Each replicate draws a class-stratified bootstrap sample from the fixed
    training matrix, refits the final estimator on that sample, and calculates
    permutation importance on the unchanged test partition. Model and
    permutation seeds are held fixed so that the intended source of variation
    is training-sample composition.

    Args:
        train_model_fn: Callback receiving one bootstrapped ``(X, y)`` pair and
            returning a newly fitted estimator with the intended specification.
        X_train: Training matrix sampled during every bootstrap replicate.
        X_test: Fixed evaluation matrix used for permutation importance.
        y_train: Class labels aligned with ``X_train``.
        y_test: Outcomes aligned with ``X_test``.
        feature_names: Ordered names for all matrix columns.
        n_bootstrap: Number of stratified model refits.
        bootstrap_seed: Master seed used to derive bootstrap-sample seeds.
        scoring: Scikit-learn scoring rule used for permutation importance.
        n_repeats: Feature permutations performed within each refit.
        permutation_seed: Fixed seed used by permutation importance.
        top_k: Number of highest-ranked features used for overlap comparisons.

    Returns:
        Three DataFrames: bootstrap-level feature importances and ranks,
        feature-level stability summaries, and pairwise Spearman/top-k
        comparisons among fitted bootstrap models.

    Raises:
        ValueError: If fewer than two bootstraps or one permutation are
            requested, feature names do not match the training width, or the
            configured scorer cannot evaluate a refitted model.

    Notes:
        The fixed permutation seed limits the intended source of between-refit
        variation to training-sample composition. Pairwise results contain
        ``n_bootstrap * (n_bootstrap - 1) / 2`` rows.
    """
    if n_bootstrap < 2:
        raise ValueError("n_bootstrap must be at least 2")
    if n_repeats < 1:
        raise ValueError("n_repeats must be at least 1")

    feature_names = list(feature_names)
    if len(feature_names) != np.asarray(X_train).shape[1]:
        raise ValueError("feature_names must match the number of training columns")

    rows = []
    master_rng = np.random.default_rng(bootstrap_seed)

    for bootstrap_id in range(1, n_bootstrap + 1):
        sample_seed = int(master_rng.integers(0, np.iinfo(np.uint32).max))
        sample_rng = np.random.default_rng(sample_seed)
        indices = _stratified_bootstrap_indices(y_train, sample_rng)
        X_bootstrap = _take_rows(X_train, indices)
        y_bootstrap = _take_rows(y_train, indices)
        model = train_model_fn(X_bootstrap, y_bootstrap)

        perm = permutation_importance(
            model,
            X_test,
            y_test,
            n_repeats=n_repeats,
            scoring=scoring,
            random_state=permutation_seed,
        )

        df = pd.DataFrame(
            {
                "Feature": feature_names,
                "Importance": perm.importances_mean,
                "ImportanceStd": perm.importances_std,
                "Bootstrap": bootstrap_id,
                "BootstrapSampleSeed": sample_seed,
            }
        )
        df["Rank"] = df["Importance"].rank(method="average", ascending=False)
        k = min(top_k, len(df))
        top_features = set(df.nsmallest(k, "Rank")["Feature"])
        df["InTopK"] = df["Feature"].isin(top_features)
        rows.append(df)

    stability_df = pd.concat(rows, ignore_index=True)
    stability_summary = (
        stability_df.groupby("Feature")
        .agg(
            MeanImportance=("Importance", "mean"),
            StdImportance=("Importance", "std"),
            MeanRank=("Rank", "mean"),
            StdRank=("Rank", "std"),
            TopKFrequency=("InTopK", "mean"),
        )
        .reset_index()
        .sort_values(["MeanRank", "MeanImportance"], ascending=[True, False])
    )

    rank_rows = []
    bootstrap_values = list(range(1, n_bootstrap + 1))
    for bootstrap_a, bootstrap_b in combinations(bootstrap_values, 2):
        a = stability_df[stability_df["Bootstrap"] == bootstrap_a][
            ["Feature", "Rank"]
        ].rename(columns={"Rank": "Rank_A"})
        b = stability_df[stability_df["Bootstrap"] == bootstrap_b][
            ["Feature", "Rank"]
        ].rename(columns={"Rank": "Rank_B"})
        merged = a.merge(b, on="Feature")
        rho = spearmanr(merged["Rank_A"], merged["Rank_B"]).correlation
        k = min(top_k, len(merged))
        top_a = set(merged.nsmallest(k, "Rank_A")["Feature"])
        top_b = set(merged.nsmallest(k, "Rank_B")["Feature"])
        rank_rows.append(
            {
                "Bootstrap_A": bootstrap_a,
                "Bootstrap_B": bootstrap_b,
                "SpearmanRankCorrelation": rho,
                "TopK": k,
                "TopKOverlap": len(top_a.intersection(top_b)) / k,
            }
        )

    return stability_df, stability_summary, pd.DataFrame(rank_rows)
