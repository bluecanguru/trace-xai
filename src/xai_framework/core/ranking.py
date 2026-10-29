"""Cross-method feature-ranking alignment and agreement statistics."""

from scipy.stats import spearmanr


def compare_importance_ranks(
    native_df,
    shap_df,
    perm_df
):
    """Compare aligned native, SHAP, and permutation feature rankings.

    Args:
        native_df: Native-importance table with ``Feature`` and ``Importance``
            columns, or ``None`` when native importance is unavailable.
        shap_df: SHAP table with ``Feature`` and ``MeanAbsSHAP`` columns.
        perm_df: Permutation-importance table with ``Feature`` and
            ``Importance`` columns.

    Returns:
        A pair ``(rank_table, agreement)``. ``rank_table`` contains the
        method-specific values and average ranks for features shared by every
        available method. ``agreement`` maps each method pair to its Spearman
        rank correlation.

    Notes:
        Rankings are calculated only over common features. Ties receive average
        ranks, and correlations may be ``NaN`` when a ranking is constant or
        contains too few features.
    """

    # Retain features shared by the compared methods.

    common = set(shap_df["Feature"]) & set(perm_df["Feature"])

    if native_df is not None:
        common &= set(native_df["Feature"])

    common = list(common)

    # Construct the SHAP ranking.

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

    shap_rank["SHAPRank"] = shap_rank["MeanAbsSHAP"].rank(
        ascending=False,
        method="average"
    )

    # Construct the permutation ranking.

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

    perm_rank["PermutationRank"] = perm_rank["Importance"].rank(
        ascending=False,
        method="average"
    )

    # Construct the native ranking.

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

    native_rank["NativeRank"] = native_rank["Importance"].rank(
        ascending=False,
        method="average"
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
