"""
run_pipeline.py

Thin wrapper around the XAI framework.

The replication notebooks are responsible for:
    • loading the dataset
    • preprocessing
    • training/selecting the final model

The framework is responsible for:
    • global explanations
    • local explanations
    • stability analysis
    • exporting all artifacts
"""

from core.experiment import XAIExperiment


def run_pipeline(config):
    """
    Execute a complete XAI experiment.

    Parameters
    ----------
    config : dict

        Required keys
        -------------
        dataset_name
        model
        X_train
        X_test
        y_train
        y_test
        feature_names

        Optional
        --------
        scoring
        train_model_fn
        stability_n_bootstrap
        stability_bootstrap_seed
        stability_scoring
        stability_n_repeats
        stability_permutation_seed
        stability_top_k
        classification_threshold
        local_cases_per_type
        lime_seeds
        lime_num_features
        lime_stability_top_k
        output_dir
    """

    experiment = XAIExperiment(config)

    experiment.run()

    experiment.export(
        config["output_dir"]
    )

    return experiment
