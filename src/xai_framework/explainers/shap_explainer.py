import shap
import numpy as np
import pandas as pd


def detect_model_type(model):
    """
    Detect the model family to select the most appropriate SHAP explainer.
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
    """
    Computes SHAP values and global feature importance.
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

    shap_values = explainer(X_test)

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
        "shap_values": shap_values,
        "raw_values": values,
        "expected_value": getattr(explainer, "expected_value", None),
        "importance": shap_df
    }