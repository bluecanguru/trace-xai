"""Validate canonical scorer settings and CKD partition integrity."""

from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]


def _row_keys(frame):
    """Convert complete rows into comparable, missing-value-safe keys.

    Args:
        frame: DataFrame whose rows should be represented uniquely.

    Returns:
        Set containing one pipe-delimited string per row.
    """
    return set(frame.fillna("<MISSING>").astype(str).agg("|".join, axis=1))


def test_canonical_tree_configs_use_roc_auc():
    """Verify Diabetes and CKD use the common continuous ROC-AUC scorer."""
    for dataset in ("diabetes", "ckd"):
        config_path = ROOT / "src" / "xai_framework" / "configs" / f"{dataset}.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert config["scoring"] == "roc_auc"


def test_ckd_has_no_exact_duplicate_or_cross_partition_profiles():
    """Verify cleaned CKD records are unique and partitions share no profiles."""
    data = pd.read_csv(ROOT / "data" / "ckd" / "kidney_disease.csv")
    data = data.replace(r"^\s+|\s+$", "", regex=True).replace("?", pd.NA)
    data["classification"] = (
        data["classification"].astype(str).str.strip().str.lower()
    )
    data = data[data["classification"].isin(["ckd", "notckd"])].copy()

    assert len(data) == 400
    assert data["id"].nunique() == 400
    assert not data["id"].duplicated().any()

    predictors = data.drop(columns=["id", "classification"])
    outcome = data["classification"]
    assert not pd.concat([predictors, outcome], axis=1).duplicated().any()

    train, test = train_test_split(
        predictors,
        test_size=0.30,
        random_state=42,
        stratify=outcome,
    )
    assert not (_row_keys(train) & _row_keys(test))
