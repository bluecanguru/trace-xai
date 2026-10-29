"""Tests for runtime provenance and tolerant result-folder comparison."""

import json
import re

import pandas as pd

from xai_framework.compare_results import compare_csv, compare_directories
from xai_framework.versioning import runtime_versions


def test_runtime_versions_has_required_provenance():
    """Verify runtime metadata records pinned key versions and a UTC timestamp."""
    result = runtime_versions()
    assert re.match(r"^3\.12\.", result["python"])
    assert result["packages"]["scikit-learn"] == "1.6.1"
    assert result["packages"]["shap"] == "0.51.0"
    assert result["created_at_utc"].endswith("+00:00")


def test_csv_comparison_accepts_small_numeric_differences(tmp_path):
    """Verify reordered rows and negligible numeric drift receive CLOSE status."""
    left, right = tmp_path / "left.csv", tmp_path / "right.csv"
    pd.DataFrame({"Feature": ["b", "a"], "Importance": [2., 1.]}).to_csv(left, index=False)
    pd.DataFrame({"Feature": ["a", "b"], "Importance": [1.000000001, 2.]}).to_csv(right, index=False)
    assert compare_csv(left, right)["status"] == "CLOSE"


def test_csv_comparison_detects_material_difference(tmp_path):
    """Verify a material numeric change receives DIFFERENT status."""
    left, right = tmp_path / "left.csv", tmp_path / "right.csv"
    pd.DataFrame({"Feature": ["a"], "Importance": [1.]}).to_csv(left, index=False)
    pd.DataFrame({"Feature": ["a"], "Importance": [2.]}).to_csv(right, index=False)
    assert compare_csv(left, right)["status"] == "DIFFERENT"


def test_directory_comparison_reports_missing_files(tmp_path):
    """Verify directory comparison reports files absent from one result set."""
    reference, candidate = tmp_path / "reference", tmp_path / "candidate"
    reference.mkdir(); candidate.mkdir()
    pd.DataFrame({"a": [1]}).to_csv(reference / "only.csv", index=False)
    report = compare_directories(reference, candidate)
    assert report["files"]["only.csv"]["status"] == "MISSING"
