"""Public entry point for executing one configured XAI experiment.

The replication notebooks are responsible for:

* loading the dataset;
* preprocessing and feature construction; and
* training or selecting the final model.

The framework is responsible for:

* global and local explanations;
* optional bootstrap-refit stability analysis; and
* reproducible result and model-bundle exports.
"""

from core.experiment import XAIExperiment


def run_pipeline(config):
    """Execute and export one complete XAI experiment.

    Args:
        config: Experiment configuration. Required entries are ``dataset_name``,
            ``model``, ``X_train``, ``X_test``, ``y_train``, ``y_test``,
            ``feature_names``, and ``output_dir``. Optional entries control the
            permutation scorer, representative-case threshold and sample size,
            LIME seeds and rule limits, the retraining callback, bootstrap and
            permutation settings, and stability top-k comparisons.

    Returns:
        Completed :class:`XAIExperiment`. Its state retains the same result
        objects that were written to ``config["output_dir"]``.

    Raises:
        KeyError: If a required configuration entry is absent.
        ValueError: If an analysis setting or aligned input is invalid.
        OSError: If the output directory or exported files cannot be written.

    Notes:
        Stages run sequentially. Bootstrap-refit stability is skipped when the
        configuration does not provide ``train_model_fn``; global and local
        explanations are still generated.
    """

    experiment = XAIExperiment(config)

    experiment.run()

    experiment.export(
        config["output_dir"]
    )

    return experiment
