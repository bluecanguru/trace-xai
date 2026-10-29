"""State container shared by the TRACE-XAI pipeline stages."""


class ExperimentState:

    """Store configuration, aligned inputs, generated results, and artefacts.

    The container keeps the fitted model and data supplied by a single
    experiment together so that global, local, stability, and export stages
    operate on the same aligned objects.

    Attributes:
        config: Original experiment configuration mapping.
        dataset_name: Identifier used for metadata and exports.
        model: Fitted estimator or fitted Statsmodels result.
        data: Available aligned matrices, labels, and feature names.
        results: Stage-specific result objects populated by the pipeline.
        artifacts: Reserved mapping for additional generated artefacts.
    """

    def __init__(self, config):

        """Initialise the state from a pipeline configuration.

        Args:
            config: Mapping containing at least ``dataset_name`` and ``model``.
                Recognised data entries are copied into ``data`` without
                transformation so their alignment is preserved.

        Raises:
            KeyError: If ``dataset_name`` or ``model`` is absent.
        """
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
