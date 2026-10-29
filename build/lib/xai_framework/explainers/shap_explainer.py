"""Select SHAP explainers and derive global attribution summaries."""

import warnings

from tqdm import TqdmWarning

warnings.filterwarnings(
    "ignore",
    message="IProgress not found.*",
    category=TqdmWarning,
)

import shap
import numpy as np
import pandas as pd


def detect_model_type(model):
    """Infer the model family used for SHAP explainer selection.

    Args:
        model: Fitted estimator or Statsmodels result.

    Returns:
        One of ``"tree"``, ``"statsmodels"``, ``"linear"``, or
        ``"generic"``.

    Notes:
        Detection relies on fitted attributes first and class-name inspection
        second. It controls explainer selection only; it does not validate the
        estimator's statistical specification.
    """
    if hasattr(model, "feature_importances_"):
        return "tree"
    
    if hasattr(model, "params"):
        return "statsmodels"

    if hasattr(model, "coef_"):
        return "linear"

    name = str(type(model)).lower()

    if "xgboost" in name:
        return "tree"

    if "lightgbm" in name:
        return "tree"
    
    if "catboost" in name:
        return "tree"

    return "generic"


def get_prediction_function(model):
    """Return a positive-class prediction callable for model-agnostic SHAP.

    Args:
        model: Fitted estimator or Statsmodels result.

    Returns:
        Callable accepting a feature matrix and returning one prediction or
        positive-class probability per observation.
    """
    # statsmodels formula-based fitted model
    if hasattr(model, "params"):
        return lambda X: np.asarray(
            model.predict(X, transform=False)
        )

    # sklearn classifier
    if hasattr(model, "predict_proba"):
        return lambda X: model.predict_proba(X)[:, 1]

    # fallback
    return model.predict


def run_shap(
    model,
    X_train,
    X_test,
    feature_names
):
    """Compute observation-level SHAP values and global importance summaries.

    Args:
        model: Fitted estimator or Statsmodels result to explain.
        X_train: Reference/background matrix used by model-agnostic explainers.
        X_test: Matrix containing the observations to explain.
        feature_names: Ordered feature or design-term names corresponding to
            the columns of the matrices.

    Returns:
        Dictionary containing the instantiated explainer, explainer class name,
        original SHAP explanation object, numeric attribution array, expected
        value when available, and a feature table with mean absolute and mean
        signed SHAP values.

    Notes:
        Tree models use ``TreeExplainer`` when possible. Linear, Statsmodels,
        and generic models use SHAP's automatic interface with ``X_train`` as
        background. If the preferred construction fails, the same automatic
        interface is used as a fallback. Three-dimensional binary-class output
        is reduced to the positive-class attribution slice.
    """
    model_type = detect_model_type(model)
    try:
        if model_type == "tree":

            explainer = shap.TreeExplainer(model)

        elif model_type in ["linear", "statsmodels"]:

            explainer = shap.Explainer(
                get_prediction_function(model),
                X_train
            )
            
        else:

            explainer = shap.Explainer(
                get_prediction_function(model),
                X_train
            )
        
    except Exception:

        explainer = shap.Explainer(
            get_prediction_function(model),
            X_train
        )

    if isinstance(explainer, shap.TreeExplainer):
        shap_values = explainer(X_test)
    else:
        shap_values = explainer(X_test, silent=True)

    values = shap_values.values

    if values.ndim == 3:

        values = values[:, :, 1]

    mean_abs = np.abs(values).mean(axis=0)
    mean_signed = values.mean(axis=0)

    shap_df = (
        pd.DataFrame({
            "Feature": feature_names,
            "MeanAbsSHAP": mean_abs,
            "MeanSHAP": mean_signed
        })
        .sort_values("MeanAbsSHAP", ascending=False)
        .reset_index(drop=True)
    )

    return {
        "explainer": explainer,
        "explainer_class": type(explainer).__name__,
        "shap_values": shap_values,
        "raw_values": values,
        "expected_value": getattr(explainer, "expected_value", None),
        "importance": shap_df
    }
