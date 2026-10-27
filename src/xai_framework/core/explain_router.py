class ExplainRouter:

    """Select explanation methods supported by a fitted model."""
    def __init__(self, model_registry):
        """Initialise the explanation router."""
        self.registry = model_registry

    def select_global_methods(self, model):

        """Return the applicable global explanation methods for a model."""
        methods = ["permutation", "shap"]

        if hasattr(model, "coef_") or hasattr(model, "feature_importances_"):
            methods.append("native")

        return methods