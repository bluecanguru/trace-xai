from core.state import ExperimentState

from explainers.native import run_native
from explainers.permutation import run_permutation
from explainers.shap_explainer import run_shap
from explainers.lime_explainer import explain_lime
from explainers.local_explain import get_representative_cases

from core.ranking import compare_importance_ranks

from xai.stability import run_stability_analysis

from exporters.export import (
    export_xai_results,
    save_xai_bundle
)

class XAIExperiment:

    def __init__(self, config):

        self.state = ExperimentState(config)

    # ============================================================
    # GLOBAL EXPLANATIONS
    # ============================================================

    def run_global(self):

        model = self.state.model

        X_train = self.state.data["X_train"]
        X_test = self.state.data["X_test"]
        y_test = self.state.data["y_test"]

        feature_names = self.state.data["feature_names"]

        scoring = self.state.config.get(
            "scoring",
            "roc_auc"
        )

        # ---------------- Native ----------------

        native_df = run_native(
            model,
            feature_names
        )

        # ---------------- Permutation ----------------

        perm_df = run_permutation(
            model,
            X_test,
            y_test,
            feature_names,
            scoring
        )

        # ---------------- SHAP ----------------

        shap_results = run_shap(
            model,
            X_train,
            X_test,
            feature_names
        )

        # ---------------- Ranking comparison ----------------

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

            "rank_compare":
                rank_compare,

            "rank_agreement":
                rank_agreement,
        }

    # ============================================================
    # LOCAL EXPLANATIONS
    # ============================================================

    def run_local(self):

        model = self.state.model

        X_train = self.state.data["X_train"]

        X_test = self.state.data["X_test"]

        y_test = self.state.data["y_test"]

        feature_names = self.state.data["feature_names"]

        representative_cases = get_representative_cases(
            model,
            X_test,
            y_test,
            feature_names
        )

        lime_df = explain_lime(
            model,
            X_train,
            X_test,
            representative_cases,
            feature_names
        )

        self.state.results["local"] = {

            "cases":
                representative_cases,

            "lime":
                lime_df

        }

    # ============================================================
    # STABILITY
    # ============================================================

    def run_stability(self):

        cfg = self.state.config

        if "train_model_fn" not in cfg:

            return

        stability_df, stability_summary = run_stability_analysis(

            cfg["train_model_fn"],

            self.state.data["X_train"],

            self.state.data["X_test"],

            self.state.data["y_train"],

            self.state.data["y_test"],

            self.state.data["feature_names"],

            scoring=cfg.get(
                "scoring",
                "roc_auc"
            )
        )

        self.state.results["stability"] = {

            "raw":
                stability_df,

            "summary":
                stability_summary

        }

    # ============================================================
    # METADATA
    # ============================================================

    def build_metadata(self):

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
                )

        }

    # ============================================================
    # RUN EVERYTHING
    # ============================================================

    def run(self):

        self.run_global()

        self.run_local()

        self.run_stability()

        self.build_metadata()

    # ============================================================
    # EXPORT
    # ============================================================

    def export(self, out_dir):

        global_results = self.state.results.get("global") or {}

        local_results = self.state.results.get("local") or {}

        stability_results = self.state.results.get("stability") or {}

        metadata = self.state.results.get("metadata") or {}

        # ---------------------------------------------------------
        # Save reproducible experiment bundle
        # ---------------------------------------------------------

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

        # ---------------------------------------------------------
        # Export XAI results
        # ---------------------------------------------------------

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

            "stability":
                stability_results.get(
                    "raw"
                ),

            "stability_summary":
                stability_results.get(
                    "summary"
                )

        }

        export_xai_results(
            export_results,
            out_dir
        )