import numpy as np
import pandas as pd


def run_native(model, feature_names):

    if hasattr(model, "coef_"):

        coef = model.coef_.ravel()

        return pd.DataFrame({
            "Feature": feature_names,
            "Importance": np.abs(coef),
            "RawValue": coef,
            "Method": "coefficients"
        }).sort_values("Importance", ascending=False)

    if hasattr(model, "feature_importances_"):

        return pd.DataFrame({
            "Feature": feature_names,
            "Importance": model.feature_importances_,
            "Method": "tree_importance"
        }).sort_values("Importance", ascending=False)

    if hasattr(model, "params"):
        p = model.params.copy()

        # Exclude intercept/constant from feature importance
        p = p[
            ~p.index.str.lower().isin(
                ["intercept", "const"]
            )
        ]

        return pd.DataFrame({
            "Feature": p.index,
            "Importance": np.abs(p.values),
            "RawValue": p.values,
            "Method": "statsmodels"
        }).sort_values(
            "Importance",
            ascending=False
        )

    return None