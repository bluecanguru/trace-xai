class ExplainRouter:

    def __init__(self, model_registry):
        self.registry = model_registry

    def select_global_methods(self, model):

        methods = ["permutation", "shap"]

        if hasattr(model, "coef_") or hasattr(model, "feature_importances_"):
            methods.append("native")

        return methods