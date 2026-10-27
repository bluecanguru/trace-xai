import pandas as pd
from sklearn.inspection import permutation_importance
from core.statsmodels_wrapper import StatsModelsWrapper

def run_permutation(model, X_test, y_test, feature_names, scoring):
    """Calculate global permutation importance with the configured scorer."""
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