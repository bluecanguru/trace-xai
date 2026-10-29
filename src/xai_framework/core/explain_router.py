"""Route fitted estimators to the global explanation methods they support."""


class ExplainRouter:

    """Select global explanation methods supported by a fitted model.

    Permutation importance and SHAP are treated as model-agnostic defaults.
    Native importance is additionally selected when the estimator exposes
    coefficients or tree-based feature importances.

    Attributes:
        registry: External model registry supplied by the caller. It is stored
            for compatibility with routing extensions but is not consulted by
            the current selection rule.
    """

    def __init__(self, model_registry):
        """Initialise the router with an optional model registry.

        Args:
            model_registry: Registry or mapping reserved for model-specific
                routing extensions.
        """
        self.registry = model_registry

    def select_global_methods(self, model):

        """Return the applicable global explanation method names.

        Args:
            model: Fitted estimator whose public attributes are inspected.

        Returns:
            Ordered list containing ``"permutation"`` and ``"shap"``, plus
            ``"native"`` when coefficients or feature importances are
            available.
        """
        methods = ["permutation", "shap"]

        if hasattr(model, "coef_") or hasattr(model, "feature_importances_"):
            methods.append("native")

        return methods
