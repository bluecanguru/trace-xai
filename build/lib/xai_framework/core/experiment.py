"""Orchestrate the global, local, stability, and export pipeline stages."""

from core.state import ExperimentState

import json
from pathlib import Path
from time import perf_counter

from explainers.native import run_native
from explainers.permutation import run_permutation
from explainers.shap_explainer import run_shap
from explainers.lime_explainer import (
    explain_lime,
    summarize_lime,
    summarize_lime_seed_stability,
)
from explainers.local_explain import get_representative_cases

from core.ranking import compare_importance_ranks

from xai.stability import run_stability_analysis

from exporters.export import (
    export_xai_results,
    save_xai_bundle
)

class XAIExperiment:

    """Coordinate the analysis and export stages for one XAI experiment.

    The experiment owns a validated :class:`ExperimentState` and populates it
    with global explanations, local explanations, optional bootstrap-refit
    stability results, and reproducibility metadata. Stage-level execution
    times are retained separately and are written during export.

    Attributes:
        state: Validated experiment inputs, configuration, and generated
            results.
        execution_times: Durations and status flags recorded for the analysis
            and export stages.
    """

    def __init__(self, config):

        """Initialise an experiment from a pipeline configuration.

        Args:
            config: Mapping containing the dataset name, fitted model, aligned
                analysis data, feature names, method settings, and optional
                retraining callback accepted by :class:`ExperimentState`.

        Raises:
            KeyError: If a required configuration entry is absent.
        """
        self.state = ExperimentState(config)
        self.execution_times = {}

    # Global explanations

    def run_global(self):

        """Generate and store the global explanation results.

        Native importance, permutation importance, and SHAP are calculated
        from the aligned experiment inputs. Their feature rankings are then
        compared using the shared ranking utilities.

        The resulting importance tables, SHAP values, explainer metadata, and
        rank-comparison outputs are stored under ``state.results["global"]``.
        Exceptions raised by an explainer or ranking method are propagated to
        the caller so that incomplete global results are not silently accepted.
        """
        model = self.state.model

        X_train = self.state.data["X_train"]
        X_test = self.state.data["X_test"]
        y_test = self.state.data["y_test"]

        feature_names = self.state.data["feature_names"]

        scoring = self.state.config.get(
            "scoring",
            "roc_auc"
        )


        native_df = run_native(
            model,
            feature_names
        )


        perm_df = run_permutation(
            model,
            X_test,
            y_test,
            feature_names,
            scoring
        )


        shap_results = run_shap(
            model,
            X_train,
            X_test,
            feature_names
        )


        rank_compare, rank_agreement = compare_importance_ranks(
            native_df,
            shap_results["importance"],
            perm_df
        )

        self.state.results["global"] = {

            "native_importance":
                native_df,

            "permutation_importance":
                perm_df,

            "shap_importance":
                shap_results["importance"],

            "shap_values":
                shap_results["shap_values"],
                
            "shap_raw_values":
                shap_results["raw_values"],

            "explainer":
                shap_results["explainer"],

            "shap_explainer_class":
                shap_results["explainer_class"],

            "rank_compare":
                rank_compare,

            "rank_agreement":
                rank_agreement,
        }

    # Local explanations

    def run_local(self):

        """Generate and store representative-case LIME explanations.

        Cases are selected by prediction outcome using the configured
        classification threshold and per-category limit. Each selected case is
        explained across the configured LIME seeds, after which rule summaries
        and per-observation seed-agreement measures are calculated.

        Representative cases, seed-level explanations, category summaries,
        and seed-stability results are stored under
        ``state.results["local"]``.
        """
        model = self.state.model

        X_train = self.state.data["X_train"]

        X_test = self.state.data["X_test"]

        y_test = self.state.data["y_test"]

        feature_names = self.state.data["feature_names"]

        representative_cases = get_representative_cases(
            model,
            X_test,
            y_test,
            feature_names,
            threshold=self.state.config.get("classification_threshold", 0.5),
            n_per_type=self.state.config.get("local_cases_per_type", 10),
        )

        lime_df = explain_lime(
            model,
            X_train,
            X_test,
            representative_cases,
            feature_names,
            seeds=self.state.config.get("lime_seeds", range(20)),
            num_features=self.state.config.get("lime_num_features"),
            threshold=self.state.config.get("classification_threshold", 0.5),
        )

        lime_summary = summarize_lime(
            lime_df,
            representative_cases["CaseType"].value_counts().to_dict(),
        )
        lime_seed_stability = summarize_lime_seed_stability(
            lime_df,
            top_k=self.state.config.get("lime_stability_top_k", 3),
        )

        self.state.results["local"] = {

            "cases":
                representative_cases,

            "lime":
                lime_df,

            "lime_summary":
                lime_summary,

            "lime_seed_stability":
                lime_seed_stability,

        }

    # Stability analysis

    def run_stability(self):

        """Evaluate and store bootstrap-refit ranking stability when enabled.

        Stability analysis is performed only when the configuration supplies a
        ``train_model_fn`` callback. The callback is used to refit the specified
        model on stratified bootstrap samples, after which permutation rankings
        are compared across successful refits.

        If no callback is available, the method returns without modifying the
        stability result. Otherwise, bootstrap-level rankings, summary values,
        and pairwise rank-stability statistics are stored under
        ``state.results["stability"]``.
        """
        cfg = self.state.config

        if "train_model_fn" not in cfg:

            return

        stability_df, stability_summary, rank_stability = run_stability_analysis(
            cfg["train_model_fn"],
            self.state.data["X_train"],
            self.state.data["X_test"],
            self.state.data["y_train"],
            self.state.data["y_test"],
            self.state.data["feature_names"],
            n_bootstrap=cfg.get("stability_n_bootstrap", 100),
            bootstrap_seed=cfg.get("stability_bootstrap_seed", 42),
            scoring=cfg.get("stability_scoring", "neg_log_loss"),
            n_repeats=cfg.get("stability_n_repeats", 10),
            permutation_seed=cfg.get("stability_permutation_seed", 42),
            top_k=cfg.get("stability_top_k", 3),
        )

        self.state.results["stability"] = {
            "raw": stability_df,
            "summary": stability_summary,
            "rank_stability": rank_stability
        }

    # Provenance metadata

    def build_metadata(self):

        """Construct and store metadata describing the completed analysis.

        The metadata records the dataset and model identifiers, matrix sizes,
        configured permutation scorer, and instantiated SHAP explainer class.
        It is stored under ``state.results["metadata"]`` and later included in
        the exported reproducibility bundle.
        """
        self.state.results["metadata"] = {

            "dataset":
                self.state.dataset_name,

            "model":
                type(self.state.model).__name__,

            "n_features":
                len(
                    self.state.data["feature_names"]
                ),

            "n_train":
                len(
                    self.state.data["X_train"]
                ),

            "n_test":
                len(
                    self.state.data["X_test"]
                ),

            "scoring":
                self.state.config.get(
                    "scoring",
                    "roc_auc"
                ),

            "shap_explainer_class":
                (self.state.results.get("global") or {}).get(
                    "shap_explainer_class"
                )

        }

    # Pipeline execution

    def run(self):

        """Execute the configured analysis stages and record their durations.

        Global explanations, local explanations, and optional stability are
        executed sequentially. The method records stage-level and total
        analysis durations, notes whether stability produced a result, and then
        builds the experiment metadata.

        This method does not write files. Call :meth:`export` after a successful
        run to create the result tables and reproducibility bundle.
        """
        run_started = perf_counter()

        stage_started = perf_counter()
        self.run_global()
        self.execution_times["global_seconds"] = perf_counter() - stage_started

        stage_started = perf_counter()
        self.run_local()
        self.execution_times["local_seconds"] = perf_counter() - stage_started

        stage_started = perf_counter()
        self.run_stability()
        self.execution_times["stability_seconds"] = perf_counter() - stage_started
        self.execution_times["stability_performed"] = bool(
            self.state.results.get("stability")
        )

        self.execution_times["analysis_seconds"] = perf_counter() - run_started
        self.build_metadata()

    # Result export

    def export(self, out_dir):

        """Export generated results, the fitted bundle, and runtime metadata.

        Args:
            out_dir: Destination directory for the canonical result tables,
                model-and-data bundle, ``execution_times.json``, and updated
                bundle metadata. The directory is created by the export
                utilities when required.

        The method exports all available global, local, and stability objects;
        unavailable optional results remain absent. It records export and total
        pipeline durations and copies the same timing information into the
        bundle metadata when that file exists.

        Raises:
            OSError: If an output directory or file cannot be created or
                written.
            TypeError: If a result cannot be serialised by the configured
                export utility.
        """
        export_started = perf_counter()
        global_results = self.state.results.get("global") or {}

        local_results = self.state.results.get("local") or {}

        stability_results = self.state.results.get("stability") or {}

        metadata = self.state.results.get("metadata") or {}

        # Save the reproducible experiment bundle.

        save_xai_bundle(

            model=self.state.model,

            X_train=self.state.data["X_train"],

            X_test=self.state.data["X_test"],

            y_train=self.state.data["y_train"],

            y_test=self.state.data["y_test"],

            feature_names=self.state.data["feature_names"],

            out_dir=out_dir,

            metadata=metadata

        )

        # Export the explanation results.

        export_results = {

            "native_importance":
                global_results.get(
                    "native_importance"
                ),

            "permutation_importance":
                global_results.get(
                    "permutation_importance"
                ),

            "shap_importance":
                global_results.get(
                    "shap_importance"
                ),

            "rank_compare":
                global_results.get(
                    "rank_compare"
                ),

            "rank_agreement":
                global_results.get(
                    "rank_agreement"
                ),

            "representative_cases":
                local_results.get(
                    "cases"
                ),

            "lime_explanations":
                local_results.get(
                    "lime"
                ),

            "lime_summary":
                local_results.get(
                    "lime_summary"
                ),

            "lime_seed_stability":
                local_results.get(
                    "lime_seed_stability"
                ),

            "stability":
                stability_results.get(
                    "raw"
                ),

            "stability_summary":
                stability_results.get(
                    "summary"
                ),
            
            "stability_rank":
                stability_results.get(
                    "rank_stability"
                ),

        }

        export_xai_results(
            export_results,
            out_dir
        )

        self.execution_times["export_seconds"] = perf_counter() - export_started
        self.execution_times["total_seconds"] = (
            self.execution_times.get("analysis_seconds", 0.0)
            + self.execution_times["export_seconds"]
        )

        output_path = Path(out_dir)
        timing_path = output_path / "execution_times.json"
        timing_path.write_text(
            json.dumps(self.execution_times, indent=4),
            encoding="utf-8",
        )

        metadata_path = output_path / "bundle" / "metadata.json"
        if metadata_path.is_file():
            bundle_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            bundle_metadata["execution_times"] = self.execution_times
            metadata_path.write_text(
                json.dumps(bundle_metadata, indent=4, default=str),
                encoding="utf-8",
            )
