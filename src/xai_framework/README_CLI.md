# Bundle-based XAI CLI

Copy this package's src/xai_framework/cli.py and configs/*.yaml into the matching
locations inside thesis/, replacing the old CLI and YAML files. Copy test_cli.py
directly into thesis/tests/, beside the existing conftest.py (not another nested
tests directory). Keep a backup or commit of the old files. No changes to
run_pipeline.py, the replication notebooks, main.py, or __init__.py are needed.

Use the same environment that saved the models. PyYAML and joblib are already in
the framework requirements; install missing dependencies in that environment.

From the thesis root, run:

```bash
python -m pytest tests -v --run-integration
python src/xai_framework/cli.py --config src/xai_framework/configs/heart.yaml --validate-only
python src/xai_framework/cli.py --config src/xai_framework/configs/heart.yaml --trust-model
```

Replace heart.yaml with stroke.yaml, diabetes.yaml, ckd.yaml, or hba1c.yaml for
other datasets. Start with Heart. HbA1c uses the complete saved design matrix
and can be expensive; no subsampling or preprocessing is silently introduced.

## What happens

1. YAML is parsed with safe_load and unknown keys are rejected.
2. CSV/JSON schema, row counts, finite numeric data, binary labels, and feature
   order are checked. The Intercept is preserved for model prediction.
3. --trust-model permits loading the saved fitted model. Pickles can execute
   code: use this flag only for your own trusted bundles.
4. The same run_pipeline(config) used by the notebooks runs explanations and
   exports a new bundle and results. No model training occurs.
5. LIME case coverage is checked and cli_run.json records the source and scope.

Validation-only does not load the model or validate predictions. Successful
validation therefore does not guarantee successful model loading or XAI runs.

## Paths and outputs

project_root: ../../.. means three directories above the YAML in
src/xai_framework/configs. Bundle and output paths are relative to that root,
independent of the current shell directory. Use Linux/WSL-compatible paths when
running in WSL. Do not use Windows C:/ paths in a WSL-specific custom YAML.

Outputs go to results/cli_runs/<dataset>_run_01. Existing nonempty destinations
are refused, not overwritten. Change run_01 to run_02 for the next run. A failed
run may leave partial output; choose a new destination on retry. The analysis
notebook still reads the original results/<dataset> directories; this CLI does
not redirect analysis or replace your thesis results.

base.yaml is now only a copyable template. It is not automatically merged.
The optional scoring field overrides metadata.scoring; omit it to preserve the
saved experiment's scorer. The original pipeline still controls its own SHAP,
LIME, and permutation defaults; unsupported YAML knobs are rejected.

## Important limits

Repeated-fit stability is NOT rerun: a bundle does not store train_model_fn.
Original stability files remain untouched and are not copied into the rerun.
CSV roundtrips and library changes mean bit-for-bit reproduction is not promised.
Use the original Python/library versions for model loading. Statsmodels formula
pickles may require original formula dependencies during deserialization; if
loading fails, use the working replication environment rather than rebuilding
the design matrix or retraining silently.

This package includes nine additional tests (validation, safe output paths,
trust acknowledgement, fitted-model dispatch). They include mocked dispatch,
not a real CLI test on every saved model. The existing real pipeline integration
test still applies. Package Python syntax was checked; this environment lacks
PyYAML/joblib and the full framework runtime, so run the commands above in your
working thesis environment to verify behavior there.
