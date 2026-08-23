import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def compare_importance_ranks(
    native_df,
    shap_df,
    perm_df
):
    """
    Compare rankings obtained from:
        - Native importance
        - SHAP importance
        - Permutation importance

    Returns
    -------
    merged_rankings : DataFrame

    agreement : dict
        Spearman correlations between ranking methods.
    """

    # -------------------------
    # Keep common features
    # -------------------------

    common = set(shap_df["Feature"]) & set(perm_df["Feature"])

    if native_df is not None:
        common &= set(native_df["Feature"])

    common = list(common)

    # -------------------------
    # SHAP ranking
    # -------------------------

    shap_rank = (
        shap_df[
            shap_df["Feature"].isin(common)
        ]
        .sort_values(
            "MeanAbsSHAP",
            ascending=False
        )
        .reset_index(drop=True)
    )

    shap_rank["SHAPRank"] = np.arange(
        1,
        len(shap_rank)+1
    )

    # -------------------------
    # Permutation ranking
    # -------------------------

    perm_rank = (
        perm_df[
            perm_df["Feature"].isin(common)
        ]
        .sort_values(
            "Importance",
            ascending=False
        )
        .reset_index(drop=True)
    )

    perm_rank["PermutationRank"] = np.arange(
        1,
        len(perm_rank)+1
    )

    # -------------------------
    # Native ranking
    # -------------------------

    if native_df is None:

        merged = (
            shap_rank
            .merge(
                perm_rank,
                on="Feature"
            )
        )

        agreement = {

            "SHAP_vs_Permutation":

                spearmanr(

                    merged["SHAPRank"],

                    merged["PermutationRank"]

                ).correlation

        }

        return merged, agreement

    native_rank = (
        native_df[
            native_df["Feature"].isin(common)
        ]
        .sort_values(
            "Importance",
            ascending=False
        )
        .reset_index(drop=True)
    )

    native_rank["NativeRank"] = np.arange(
        1,
        len(native_rank)+1
    )

    merged = (

        native_rank

        .merge(
            shap_rank,
            on="Feature"
        )

        .merge(
            perm_rank,
            on="Feature"
        )

    )

    agreement = {

        "Native_vs_SHAP":

            spearmanr(
                merged["NativeRank"],
                merged["SHAPRank"]
            ).correlation,

        "Native_vs_Permutation":

            spearmanr(
                merged["NativeRank"],
                merged["PermutationRank"]
            ).correlation,

        "SHAP_vs_Permutation":

            spearmanr(
                merged["SHAPRank"],
                merged["PermutationRank"]
            ).correlation

    }

    return merged, agreement