"""Use the framework's existing imports; do not alter its package structure."""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK = Path(os.environ.get(
    "XAI_FRAMEWORK_DIR", str(ROOT / "src" / "xai_framework")
)).resolve()
if not (FRAMEWORK / "run_pipeline.py").is_file():
    raise pytest.UsageError(
        "Copy tests/ into thesis/, or set XAI_FRAMEWORK_DIR to "
        "the absolute path of thesis/src/xai_framework."
    )
sys.path.insert(0, str(FRAMEWORK))


def pytest_addoption(parser):
    parser.addoption("--run-integration", action="store_true", default=False)


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: real SHAP/LIME smoke test")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-integration"):
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(pytest.mark.skip(reason="Use --run-integration"))
