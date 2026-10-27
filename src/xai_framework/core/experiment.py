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

    """Coordinate global, local, stability, metadata, and export stages."""
    def __init__(self, config):

        """Initialise an experiment around one validated state object."""
        self.state = ExperimentState(config)
        self.execution_times = {}

    # Global explanations

    def run_global(self):

        """Generate global importance outputs and rank comparisons."""
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

        """Select local cases and generate repeated-seed LIME outputs."""
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

        """Evaluate bootstrap-refit permutation-ranking stability when enabled."""
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

        """Construct reproducibility metadata for the current experiment."""
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

        """Execute every enabled analysis stage in sequence."""
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

        """Export the experiment state and reproducibility bundle."""
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
