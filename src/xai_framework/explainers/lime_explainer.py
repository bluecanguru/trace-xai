import numpy as np
import pandas as pd

from lime.lime_tabular import LimeTabularExplainer
from core.statsmodels_wrapper import StatsModelsWrapper


def get_probabilities(model, X):
    """
    Returns probabilities for the positive class.

    Supports:
    - predict_proba()
    - decision_function()
    - regression-like probability outputs
    """

    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X)[:, 1])

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return 1 / (1 + np.exp(-scores))

    return np.asarray(model.predict(X))


def explain_lime(
    model,
    X_train,
    X_test,
    cases,
    feature_names,
    seeds=range(20),
    num_features=None,
    threshold=0.5,
):
    """
    Generate repeated LIME explanations for selected cases.

    The explainer's random state is reset for every seed so the exported rows
    preserve variability introduced by LIME's perturbation sampling without
    repeatedly rebuilding the training-data discretizer.

    Returns
    -------
    DataFrame
        CaseIndex
        CaseType
        CaseRank
        LIMESeed
        Prediction
        Probability
        Rule
        Weight
    """

    if hasattr(model, "params"):
        model = StatsModelsWrapper(model)

    # LIME works internally with NumPy arrays
    X_train_np = np.asarray(X_train)
    X_test_np = np.asarray(X_test)

    def lime_predict_proba(X):
        """Return class probabilities in the format expected by LIME."""
        X_df = pd.DataFrame(
            X,
            columns=feature_names
        )

        probs = get_probabilities(
            model,
            X_df
        )

        return np.column_stack([
            1 - probs,
            probs
        ])
    
    seeds = list(seeds)
    if not seeds:
        raise ValueError("At least one LIME seed is required.")

    if num_features is None:
        num_features = min(10, len(feature_names))
    if not isinstance(num_features, int) or num_features < 1:
        raise ValueError("num_features must be a positive integer.")

    rows = []

    explainer = LimeTabularExplainer(
        training_data=X_train_np,
        feature_names=feature_names,
        mode="classification",
        random_state=seeds[0],
    )

    for seed in seeds:
        explainer.random_state = np.random.RandomState(seed)

        for idx, case in cases.iterrows():
            try:

            # idx corresponds to the row position in X_test
                x_row = X_test_np[idx]

                x_row_df = pd.DataFrame(
                    x_row.reshape(1, -1),
                    columns=feature_names
                )

                prob = get_probabilities(
                    model,
                    x_row_df
                )[0]

                pred = int(prob >= threshold)

                exp = explainer.explain_instance(
                    x_row,
                    lime_predict_proba,
                    num_features=num_features,
                )

                for rank, (rule, weight) in enumerate(exp.as_list(), start=1):

                    rows.append({
                        "CaseIndex": idx,
                        "CaseType": case["CaseType"],
                        "CaseRank": case["CaseRank"],
                        "LIMESeed": seed,
                        "Prediction": pred,
                        "Probability": prob,
                        "RuleRank": rank,
                        "Rule": rule,
                        "Weight": weight,
                    })

            except Exception as e:
                print(f"LIME failed for case {idx}, seed {seed}: {e}")

    return pd.DataFrame(rows)


def summarize_lime(lime_df, n_selected_cases):
    """Aggregate repeated LIME weights by case category and rule."""
    columns = [
        "CaseType", "Rule", "MeanWeight", "StdWeight", "MeanAbsWeight",
        "MedianWeight", "PositiveWeightFraction", "NExplanations", "NCases",
        "NSeeds", "SelectionFrequency",
    ]
    if lime_df.empty:
        return pd.DataFrame(columns=columns)

    summary = (
        lime_df.groupby(["CaseType", "Rule"], as_index=False)
        .agg(
            MeanWeight=("Weight", "mean"),
            StdWeight=("Weight", "std"),
            MeanAbsWeight=("Weight", lambda values: values.abs().mean()),
            MedianWeight=("Weight", "median"),
            PositiveWeightFraction=("Weight", lambda values: (values > 0).mean()),
            NExplanations=("Weight", "size"),
            NCases=("CaseIndex", "nunique"),
            NSeeds=("LIMESeed", "nunique"),
        )
    )
    denominators = {
        case_type: count * lime_df.loc[
            lime_df["CaseType"] == case_type, "LIMESeed"
        ].nunique()
        for case_type, count in n_selected_cases.items()
    }
    summary["SelectionFrequency"] = summary.apply(
        lambda row: row["NExplanations"] / denominators[row["CaseType"]],
        axis=1,
    )
    return summary.sort_values(
        ["CaseType", "MeanAbsWeight"], ascending=[True, False]
    ).reset_index(drop=True)


def summarize_lime_seed_stability(lime_df, top_k=3):
    """Summarize agreement among repeated LIME seeds for each observation."""
    columns = [
        "CaseType", "CaseIndex", "CaseRank", "NSeeds",
        "MeanPairwiseSpearman", "MeanTopKOverlap", "TopK",
        "MeanRuleWeightSD", "MeanRuleSignConsistency",
    ]
    if lime_df.empty:
        return pd.DataFrame(columns=columns)
    if not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer.")

    rows = []
    for (case_type, case_index, case_rank), group in lime_df.groupby(
        ["CaseType", "CaseIndex", "CaseRank"], sort=False
    ):
        weights = group.pivot_table(
            index="Rule", columns="LIMESeed", values="Weight", fill_value=0.0
        )
        seeds = list(weights.columns)
        rank_correlations = weights.rank(axis=0).corr()
        upper_triangle = np.triu_indices_from(rank_correlations, k=1)
        correlations = rank_correlations.to_numpy()[upper_triangle]
        correlations = correlations[np.isfinite(correlations)]
        overlaps = []
        effective_k = min(top_k, len(weights.index))

        for left_position, left_seed in enumerate(seeds):
            left = weights[left_seed]
            left_top = set(left.abs().nlargest(effective_k).index)
            for right_seed in seeds[left_position + 1:]:
                right = weights[right_seed]
                right_top = set(right.abs().nlargest(effective_k).index)
                overlaps.append(len(left_top & right_top) / effective_k)

        rule_stds = weights.std(axis=1, ddof=1)
        sign_consistency = weights.apply(
            lambda values: max((values > 0).mean(), (values < 0).mean()), axis=1
        )
        rows.append({
            "CaseType": case_type,
            "CaseIndex": case_index,
            "CaseRank": case_rank,
            "NSeeds": len(seeds),
            "MeanPairwiseSpearman": correlations.mean() if correlations.size else np.nan,
            "MeanTopKOverlap": np.mean(overlaps) if overlaps else np.nan,
            "TopK": effective_k,
            "MeanRuleWeightSD": rule_stds.mean(),
            "MeanRuleSignConsistency": sign_consistency.mean(),
        })

    return pd.DataFrame(rows, columns=columns)
