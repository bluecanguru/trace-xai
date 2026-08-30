"""Runtime provenance stored in bundles and CLI manifests."""
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version


PACKAGES = (
    "numpy", "pandas", "scikit-learn", "scipy", "shap", "lime",
    "statsmodels", "joblib", "PyYAML", "matplotlib"
)


def runtime_versions():
    packages = {}
    for package in PACKAGES:
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
    }
