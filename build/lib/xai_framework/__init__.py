"""XAI framework package with compatibility aliases for legacy notebooks."""
import sys

from . import core, explainers, exporters, models, xai

# Existing notebooks add src/xai_framework to sys.path and import top-level
# modules. These aliases let the same unchanged implementation also work as an
# installed package. Remove them only after migrating all notebooks.
sys.modules.setdefault("core", core)
sys.modules.setdefault("explainers", explainers)
sys.modules.setdefault("exporters", exporters)
sys.modules.setdefault("models", models)
sys.modules.setdefault("xai", xai)

__version__ = "1.0.0"
