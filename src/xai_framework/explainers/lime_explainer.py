from pyexpat import model

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
    feature_names
):
    """
    Generates LIME explanations for representative cases.

    Returns
    -------
    DataFrame
        CaseIndex
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
    
    explainer = LimeTabularExplainer(
        training_data=X_train_np,
        feature_names=feature_names,
        mode="classification",
        random_state=42
    )

    rows = []

    for idx in cases.index:
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

            pred = int(prob >= 0.5)

            exp = explainer.explain_instance(
                x_row,
                lime_predict_proba
            )

            for rule, weight in exp.as_list():

                rows.append({
                    "CaseIndex": idx,
                    "Prediction": pred,
                    "Probability": prob,
                    "Rule": rule,
                    "Weight": weight
                })

        except Exception as e:
            print(f"LIME failed for case {idx}: {e}")

    return pd.DataFrame(rows)