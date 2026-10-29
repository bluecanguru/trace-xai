"""Calculate model-agnostic global permutation importance."""

import pandas as pd
from sklearn.inspection import permutation_importance
from core.statsmodels_wrapper import StatsModelsWrapper

def run_permutation(model, X_test, y_test, feature_names, scoring):
    """Calculate global permutation importance on an evaluation partition.

    Args:
        model: Fitted scikit-learn estimator or Statsmodels result.
        X_test: Evaluation matrix accepted by the fitted model.
        y_test: Outcomes aligned with ``X_test``.
        feature_names: Ordered names for the columns of ``X_test``.
        scoring: Scikit-learn scorer name used to quantify performance loss
            after permutation.

    Returns:
        DataFrame containing each feature's mean importance and standard
        deviation across 20 permutations, sorted by decreasing importance.

    Raises:
        ValueError: If the scorer is invalid or the data do not support it.

    Notes:
        Permutations use a fixed random seed of 42. Fitted Statsmodels results
        are adapted to the scikit-learn estimator protocol before evaluation.
    """
    if hasattr(model, "params"):
        model = StatsModelsWrapper(model)
    
    perm = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=20,
        random_state=42,
        scoring=scoring
    )

    return pd.DataFrame({
        "Feature": feature_names,
        "Importance": perm.importances_mean,
        "Std": perm.importances_std
    }).sort_values("Importance", ascending=False)
