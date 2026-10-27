class ExperimentState:

    """Store aligned inputs, settings, intermediate outputs, and metadata."""
    def __init__(self, config):

        """Initialise an empty experiment-state container."""
        self.config = config
        self.dataset_name = config["dataset_name"]
        self.model = config["model"]

        self.data = {}

        for key in [
            "X_train",
            "X_test",
            "y_train",
            "y_test",
            "X",
            "y",
            "analysis_df",
            "feature_names"
        ]:
            if key in config:
                self.data[key] = config[key]

        self.results = {
            "global": None,
            "local": None,
            "stability": None,
            "metrics": None
        }

        self.artifacts = {}