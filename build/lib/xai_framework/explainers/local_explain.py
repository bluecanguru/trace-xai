
"""Select prediction-outcome cases for repeated local explanation."""

import numpy as np
import pandas as pd

from core.statsmodels_wrapper import StatsModelsWrapper


def get_probabilities(model, X):
    """Return positive-class probabilities from a supported fitted model.

    Args:
        model: Fitted estimator exposing ``predict_proba``,
            ``decision_function``, or ``predict``.
        X: Feature matrix accepted by ``model``.

    Returns:
        One-dimensional NumPy array of positive-class probabilities or
        probability-like predictions.

    Notes:
        Decision-function scores are converted with the logistic function.
        Floating outputs from ``predict`` are treated as probabilities; other
        outputs are converted to floating-point values without calibration.
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
    threshold=0.5,
    n_per_type=1,
):
    """Select outcome-stratified observations for local explanation.

    Args:
        model: Fitted classifier or Statsmodels binary-result object.
        X_test: Evaluation matrix as a DataFrame or NumPy-compatible array.
        y_test: Binary outcomes aligned positionally with ``X_test``.
        feature_names: Ordered names used when ``X_test`` is not a DataFrame.
        threshold: Probability threshold used to assign predicted classes.
        n_per_type: Maximum observations selected for each available category.

    Returns:
        DataFrame containing the selected features, observed and predicted
        classes, probability, distance from the threshold, case category,
        original positional index, and within-category rank.

    Raises:
        ValueError: If inputs have inconsistent row counts or ``n_per_type`` is
            not a positive integer.

    Notes:
        True positives and false positives are selected by highest positive
        probability; true negatives and false negatives by lowest probability.
        Borderline observations are the remaining cases nearest the threshold,
        preventing the same row from being assigned to two categories.
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

    if not isinstance(n_per_type, int) or n_per_type < 1:
        raise ValueError("n_per_type must be a positive integer.")

    # Most confident correctly classified positives.
    tp = case_df[
        (case_df["y_true"] == 1) &
        (case_df["y_pred"] == 1)
    ].nlargest(n_per_type, "y_prob").copy()

    # Most confident correctly classified negatives.
    tn = case_df[
        (case_df["y_true"] == 0) &
        (case_df["y_pred"] == 0)
    ].nsmallest(n_per_type, "y_prob").copy()

    # Most confident false positives.
    fp = case_df[
        (case_df["y_true"] == 0) &
        (case_df["y_pred"] == 1)
    ].nlargest(n_per_type, "y_prob").copy()

    # Most confident false negatives.
    fn = case_df[
        (case_df["y_true"] == 1) &
        (case_df["y_pred"] == 0)
    ].nsmallest(n_per_type, "y_prob").copy()

    # Borderline observations are selected after the four outcome categories so
    # the same observation is not explained twice under different labels.
    selected_indices = set(tp.index) | set(tn.index) | set(fp.index) | set(fn.index)
    borderline = case_df.loc[~case_df.index.isin(selected_indices)].nsmallest(
        n_per_type, "distance"
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
    representative_cases = representative_cases.copy()

    representative_cases["CaseIndex"] = representative_cases.index
    representative_cases["CaseRank"] = (
        representative_cases.groupby("CaseType", sort=False).cumcount() + 1
    )

    return representative_cases
