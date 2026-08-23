# core/stability.py

import pandas as pd
from sklearn.inspection import permutation_importance

def run_stability_analysis(
    train_model_fn,
    X_train,
    X_test,
    y_train,
    y_test,
    feature_names,
    seeds=[1,2,3,4,5],
    scoring="roc_auc"
):

    rows = []

    for seed in seeds:

        model = train_model_fn(seed)

        perm = permutation_importance(
            model,
            X_test,
            y_test,
            n_repeats=10,
            scoring=scoring,
            random_state=seed
        )

        df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": perm.importances_mean,
            "Seed": seed
        })

        rows.append(df)

    stability_df = pd.concat(rows, ignore_index=True)

    stability_summary = stability_df.groupby("Feature").agg(
        MeanImportance=("Importance", "mean"),
        StdImportance=("Importance", "std")
    ).reset_index()

    return stability_df, stability_summary