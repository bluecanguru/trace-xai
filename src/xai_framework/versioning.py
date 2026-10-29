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
    """Collect timestamped runtime and dependency provenance.

    Returns:
        Dictionary containing the UTC creation timestamp, Python version and
        executable, platform description, and installed versions of the
        framework's principal dependencies. A missing dependency is recorded
        as ``None`` rather than raising an exception.
    """
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
