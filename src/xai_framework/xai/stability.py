# xai/stability.py

import numpy as np
import pandas as pd

from itertools import combinations
from scipy.stats import spearmanr
from sklearn.inspection import permutation_importance


def run_stability_analysis(
    train_model_fn,
    X_train,
    X_test,
    y_train,
    y_test,
    feature_names,
    seeds=(1, 2, 3, 4, 5),
    scoring="roc_auc",
    n_repeats=10,
    top_k=3
):
    """
    Evaluate permutation-importance stability across repeated model fits.

    Stability is assessed at two levels:

    1. Importance-value stability
       - mean permutation importance
       - standard deviation of permutation importance

    2. Ranking stability
       - feature rank for each seed
       - pairwise Spearman rank correlation between seeds
       - top-k feature overlap between seeds

    Parameters
    ----------
    train_model_fn : callable
        Function receiving a seed and returning a fitted model.

    X_train, X_test :
        Training and test feature matrices.

    y_train, y_test :
        Training and test labels.

    feature_names : list
        Names of the predictors.

    seeds : iterable
        Random seeds used for repeated runs.

    scoring : str
        Scikit-learn scoring metric.

    n_repeats : int
        Number of permutation repetitions per seed.

    top_k : int
        Number of highest-ranked features used for top-k overlap.

    Returns
    -------
    stability_df : pandas.DataFrame
        Feature-level importance and rank for every seed.

    stability_summary : pandas.DataFrame
        Mean and standard deviation of importance and rank by feature.

    rank_stability : pandas.DataFrame
        Pairwise Spearman rank correlations and top-k overlap
        between seeds.
    """

    rows = []

    # ============================================================
    # 1. COMPUTE PERMUTATION IMPORTANCE FOR EACH MODEL/SEED
    # ============================================================

    for seed in seeds:

        model = train_model_fn(seed)

        perm = permutation_importance(
            model,
            X_test,
            y_test,
            n_repeats=n_repeats,
            scoring=scoring,
            random_state=seed
        )

        df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": perm.importances_mean,
            "ImportanceStd": perm.importances_std,
            "Seed": seed
        })

        # Rank 1 = most important
        df["Rank"] = (
            df["Importance"]
            .rank(
                method="average",
                ascending=False
            )
        )

        rows.append(df)

    stability_df = pd.concat(
        rows,
        ignore_index=True
    )

    # ============================================================
    # 2. FEATURE-LEVEL STABILITY SUMMARY
    # ============================================================

    stability_summary = (
        stability_df
        .groupby("Feature")
        .agg(
            MeanImportance=("Importance", "mean"),
            StdImportance=("Importance", "std"),
            MeanRank=("Rank", "mean"),
            StdRank=("Rank", "std")
        )
        .reset_index()
        .sort_values(
            "MeanImportance",
            ascending=False
        )
    )

    # ============================================================
    # 3. PAIRWISE RANK STABILITY
    # ============================================================

    rank_rows = []

    seed_values = list(seeds)

    for seed_a, seed_b in combinations(seed_values, 2):

        a = (
            stability_df[
                stability_df["Seed"] == seed_a
            ][["Feature", "Rank", "Importance"]]
            .rename(
                columns={
                    "Rank": "Rank_A",
                    "Importance": "Importance_A"
                }
            )
        )

        b = (
            stability_df[
                stability_df["Seed"] == seed_b
            ][["Feature", "Rank", "Importance"]]
            .rename(
                columns={
                    "Rank": "Rank_B",
                    "Importance": "Importance_B"
                }
            )
        )

        merged = a.merge(
            b,
            on="Feature"
        )

        rho = spearmanr(
            merged["Rank_A"],
            merged["Rank_B"]
        ).correlation

        # Top-k overlap
        k = min(
            top_k,
            len(merged)
        )

        top_a = set(
            merged
            .nsmallest(k, "Rank_A")["Feature"]
        )

        top_b = set(
            merged
            .nsmallest(k, "Rank_B")["Feature"]
        )

        overlap = len(
            top_a.intersection(top_b)
        ) / k

        rank_rows.append({
            "Seed_A": seed_a,
            "Seed_B": seed_b,
            "SpearmanRankCorrelation": rho,
            "TopK": k,
            "TopKOverlap": overlap
        })

    rank_stability = pd.DataFrame(rank_rows)

    return (
        stability_df,
        stability_summary,
        rank_stability
    )