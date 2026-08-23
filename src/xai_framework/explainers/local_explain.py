# explainers/local_explain.py

import numpy as np
import pandas as pd

from core.statsmodels_wrapper import StatsModelsWrapper


def get_probabilities(model, X):
    """
    Return probabilities for the positive class.

    Supports:
    - sklearn models with predict_proba()
    - sklearn models with decision_function()
    - models whose predict() already returns probabilities
    """

    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X)[:, 1])

    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X))
        return 1 / (1 + np.exp(-scores))

    preds = np.asarray(model.predict(X))

    # Some models, such as statsmodels, may already return probabilities.
    if preds.dtype.kind == "f":
        return preds

    return preds.astype(float)


def get_representative_cases(
    model,
    X_test,
    y_test,
    feature_names,
    threshold=0.5
):
    """
    Select representative observations for local explanations.

    Selected case types:
    - TP: strongest true positive
    - TN: strongest true negative
    - FP: strongest false positive
    - FN: strongest false negative
    - Borderline: observation closest to the decision threshold

    Parameters
    ----------
    model
        Fitted model.

    X_test
        Test feature matrix. Can be either a pandas DataFrame
        or a NumPy array.

    y_test
        True test labels.

    feature_names
        Names of the predictor variables.

    threshold
        Classification threshold. Default is 0.5.

    Returns
    -------
    pandas.DataFrame
        Representative observations and their prediction information.
    """

    # Adapt statsmodels objects to an sklearn-like interface.
    if hasattr(model, "params"):
        model = StatsModelsWrapper(model)

    # Predicted probabilities and classes.
    probs = get_probabilities(model, X_test)
    preds = (probs >= threshold).astype(int)

    # Convert input to DataFrame so downstream logic is consistent
    # regardless of whether the original data were NumPy or pandas.
    if isinstance(X_test, pd.DataFrame):
        case_df = X_test.copy()
    else:
        case_df = pd.DataFrame(
            X_test,
            columns=feature_names
        )

    # Important:
    # LIME later uses NumPy positional indexing.
    # Therefore the representative-case indices must correspond to
    # positions 0, 1, ..., n_test - 1 rather than original dataset indices.
    case_df = case_df.reset_index(drop=True)

    # Ensure y_test aligns positionally with X_test.
    y_true = np.asarray(y_test).reshape(-1)

    if len(case_df) != len(y_true):
        raise ValueError(
            "X_test and y_test have different numbers of observations: "
            f"{len(case_df)} != {len(y_true)}"
        )

    if len(case_df) != len(probs):
        raise ValueError(
            "The number of model predictions does not match X_test: "
            f"{len(probs)} != {len(case_df)}"
        )

    case_df["y_true"] = y_true
    case_df["y_pred"] = preds
    case_df["y_prob"] = probs
    case_df["distance"] = np.abs(probs - threshold)

    # Strongest correctly classified positive.
    tp = case_df[
        (case_df["y_true"] == 1) &
        (case_df["y_pred"] == 1)
    ].nlargest(1, "y_prob").copy()

    # Strongest correctly classified negative.
    tn = case_df[
        (case_df["y_true"] == 0) &
        (case_df["y_pred"] == 0)
    ].nsmallest(1, "y_prob").copy()

    # Most confident false positive.
    fp = case_df[
        (case_df["y_true"] == 0) &
        (case_df["y_pred"] == 1)
    ].nlargest(1, "y_prob").copy()

    # Most confident false negative.
    fn = case_df[
        (case_df["y_true"] == 1) &
        (case_df["y_pred"] == 0)
    ].nsmallest(1, "y_prob").copy()

    # Observation closest to the classification threshold.
    borderline = case_df.nsmallest(
        1,
        "distance"
    ).copy()

    # Add case labels only when the corresponding case exists.
    if not tp.empty:
        tp["CaseType"] = "TP"

    if not tn.empty:
        tn["CaseType"] = "TN"

    if not fp.empty:
        fp["CaseType"] = "FP"

    if not fn.empty:
        fn["CaseType"] = "FN"

    if not borderline.empty:
        borderline["CaseType"] = "Borderline"

    representative_cases = pd.concat(
        [tp, tn, fp, fn, borderline],
        axis=0,
        ignore_index=False
    )

    # Avoid returning the exact same observation twice, for example
    # when the borderline case is also the selected TP/TN/etc.
    representative_cases = representative_cases.loc[
        ~representative_cases.index.duplicated(keep="first")
    ]

    return representative_cases