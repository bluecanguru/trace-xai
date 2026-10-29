"""Extract model-native global feature-importance representations."""

import numpy as np
import pandas as pd


def run_native(model, feature_names):
    """Extract the native importance representation exposed by a fitted model.

    Args:
        model: Fitted estimator exposing ``coef_``, ``feature_importances_``,
            or Statsmodels ``params``.
        feature_names: Ordered feature names corresponding to coefficient- or
            tree-importance positions. Statsmodels parameter names are read
            directly from the fitted result.

    Returns:
        DataFrame sorted by decreasing absolute importance. Coefficient-based
        outputs also retain signed values in ``RawValue``. Returns ``None``
        when the model exposes no supported native importance.

    Notes:
        Statsmodels intercept terms named ``Intercept`` or ``const`` are
        excluded because they are not predictor features.
    """
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
