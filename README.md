# Reproducible XAI Framework for Cross-Dataset Replication

This repository contains five paper-replication notebooks and a reusable binary-classification XAI framework.

The replication notebooks own data preparation, model selection, training, and the optional retraining callback. The framework consumes the selected fitted model and prepared matrices, then produces native, permutation, SHAP, LIME, rank-agreement, representative-case, and optional stability outputs.

## Supported scope

The framework currently supports:

- Binary classification with class labels `0` and `1`.
- Scikit-learn estimators.
- The tested statsmodels binary-model adapter.
- NumPy arrays and pandas DataFrames.
- Already-created Patsy design matrices.
- Repeated-fit stability when the replication notebook supplies `train_model_fn`.

It does not claim general support for:

- Regression.
- Multiclass classification.
- Arbitrary statsmodels formulas.
- Untrusted serialized model files.
- Clinical validation or clinical decision-making.

## Reproducible environment

The final tested environment is Windows CPython 3.12.6.

Create and activate a dedicated virtual environment from the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the exact XAI dependency lock and the framework:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements-xai-lock.txt
python -m pip install -e . --no-deps
```

If PowerShell prevents activation, the environment’s Python executable can be used directly:

```powershell
& ".\.venv\Scripts\python.exe" -m pip install --upgrade pip
& ".\.venv\Scripts\python.exe" -m pip install -r requirements-xai-lock.txt
& ".\.venv\Scripts\python.exe" -m pip install -e . --no-deps
```

The dependency files have different purposes:

- `requirements-xai-lock.txt` records the focused, exact XAI runtime and is the recommended installation source.
- `requirements-environment-lock.txt` is the complete archival `pip freeze` of the final Windows interpreter, including unrelated packages. It is retained for provenance rather than normal installation.
- `pyproject.toml` defines the installable package and command-line entry points.

New bundles and CLI manifests record Python, platform, and relevant library versions. Bundles created before this functionality was added cannot acquire missing provenance retroactively.

## Project structure

```text
data/                         dataset inputs
src/replication/              model replication and selection notebooks
src/xai_framework/            reusable XAI implementation
src/xai_framework/configs/    bundle-based CLI configurations
src/analysis/                 cross-dataset results analysis
tests/                        unit and integration tests
results/<dataset>/            canonical notebook-generated results
results/cli_runs/             independent bundle-based reruns
examples/                     optional examples
chapter/                      thesis text
main.py                       multi-dataset batch orchestrator
main.tex                      thesis document
mybib.bib                     thesis bibliography
```

The older notebooks under `src/notebooks/` are prototypes and are not the canonical workflow.

Empty `__init__.py` files are valid Python package markers. Empty legacy result folders are not used by the dataset-oriented exporter.

## Canonical workflow

The canonical thesis workflow is:

1. Open each notebook under `src/replication/` using the thesis environment.
2. Restart the kernel and run all cells.
3. Confirm that each notebook selects its replication model and calls `run_pipeline(config)`.
4. Confirm that each dataset writes its canonical outputs to `results/<dataset>/`.
5. After all five datasets succeed, restart and run `src/analysis/xai_results_analysis_corrected.ipynb`.
6. Run the complete automated test suite.
7. Optionally use the bundle CLI or batch runner to independently reproduce and compare the saved XAI outputs.

Model-selection evidence remains in the replication notebooks. CLI-generated results must not replace the canonical notebook-generated results without an explicit reviewed decision.

## Tests

Activate the project environment and run the complete suite:

```powershell
python -m pytest tests -v --run-integration
```

If the environment is not activated, use its Python executable directly:

```powershell
& ".\.venv\Scripts\python.exe" -m pytest tests -v --run-integration
```

The complete suite currently contains 40 tests, including a small real SHAP/LIME integration pipeline.

The tests cover:

- Binary metric calculation.
- Task detection.
- Model selection.
- Native importance extraction.
- Feature alignment.
- Rank comparison and tie handling.
- Representative-case selection.
- CSV and JSON exports.
- Bundle round trips.
- Pipeline execution order.
- Runtime provenance.
- Result comparison.
- Batch validation and orchestration.
- A small real SHAP/LIME pipeline.

Passing tests support only the documented and tested scope. They are not clinical validation or proof of statistical explanation fidelity.

## Bundle CLI

The package can be executed with:

```powershell
python -m xai_framework
```

If the virtual environment is not activated, use:

```powershell
& ".\.venv\Scripts\python.exe" -m xai_framework
```

### Validate a bundle

Validate a trusted saved bundle without deserializing its model:

```powershell
python -m xai_framework `
  --config src/xai_framework/configs/heart.yaml `
  --validate-only
```

Validation checks the saved CSV and JSON data, feature order, labels, configuration, and expected bundle structure.

Validation does not load the model or verify predictions.

### Rerun XAI from a bundle

Rerun the XAI pipeline using the fitted model stored in the bundle:

```powershell
python -m xai_framework `
  --config src/xai_framework/configs/heart.yaml `
  --trust-model
```

Pickle and joblib files can execute code while loading. Use `--trust-model` only for bundles that you created and control.

The configured output path contains a UTC `{timestamp}`, so every execution receives a new output directory. Canonical results are not overwritten.

Validation and execution performed as separate commands display different prospective output timestamps. This is expected.

### CLI limitations

The bundle CLI:

- Does not retrain models.
- Cannot reproduce repeated-fit stability because bundles do not serialize `train_model_fn`.
- Preserves statsmodels design columns.
- Validates data and feature order.
- Checks LIME representative-case coverage.
- Writes a `cli_run.json` manifest with runtime provenance.

Warnings that a scikit-learn estimator was fitted without feature names can occur when a saved estimator was originally fitted with a NumPy array and is later evaluated using an equivalent DataFrame. These warnings should be reviewed, but they do not necessarily indicate a failed run.

## Result comparison

Canonical CSV exports can be compared with an independent CLI rerun using numerical tolerances:

```powershell
python -m xai_framework.compare_results `
  --reference results/heart `
  --candidate results/cli_runs/heart_YYYYMMDDTHHMMSSZ `
  --output results/comparisons/heart.json
```

Custom tolerances can be supplied when justified:

```powershell
python -m xai_framework.compare_results `
  --reference results/heart `
  --candidate results/cli_runs/heart_YYYYMMDDTHHMMSSZ `
  --atol 0.0005 `
  --rtol 0.005 `
  --output results/comparisons/heart_with_tolerance.json
```

Comparison statuses are:

- `EXACT`: text and numerical values match exactly.
- `CLOSE`: numerical values match within the configured tolerances.
- `DIFFERENT`: at least one comparable value exceeds the tolerances.
- `SCHEMA_DIFFERENT`: table columns differ.
- `SHAPE_DIFFERENT`: table dimensions differ.
- `MISSING`: a compared file is absent from one side.

SHAP and LIME results can vary slightly across executions. Differences must be reviewed rather than automatically increasing tolerances until a report passes.

The comparison covers exported CSV tables. It does not prove model equivalence, statistical explanation validity, or clinical validity.

## Batch orchestration

The root `main.py` validates and reruns multiple saved bundles through one command.

### Validate all datasets

Validate all five saved bundles without loading their serialized models:

```powershell
python main.py validate
```

Without an activated environment:

```powershell
& ".\.venv\Scripts\python.exe" main.py validate
```

### Run all datasets

Run all five trusted bundles and compare the generated results with the canonical results:

```powershell
python main.py run --trust-models --compare
```

Without an activated environment:

```powershell
& ".\.venv\Scripts\python.exe" main.py run --trust-models --compare
```

### Run selected datasets

Validate selected datasets:

```powershell
python main.py validate --datasets heart diabetes
```

Run and compare selected datasets:

```powershell
python main.py run `
  --datasets heart diabetes `
  --trust-models `
  --compare
```

### Batch outputs

Each batch receives a UTC timestamp and writes to:

```text
results/cli_runs/batches/batch_YYYYMMDDTHHMMSSZ/
  batch_manifest.json
  heart/
  stroke/
  diabetes/
  ckd/
  hba1c/
  comparisons/
```

The batch runner:

- Validates every selected bundle before loading any model.
- Refuses to overwrite an existing batch directory.
- Does not overwrite canonical results.
- Stops on the first execution failure by default.
- Preserves partial output after failures or interruptions.
- Records dataset progress and outcomes in `batch_manifest.json`.
- Can continue to later datasets with `--continue-on-error`.

A dataset marked `PASSED` completed its pipeline execution successfully. This does not necessarily mean that every generated file exactly matched the canonical version. The corresponding comparison report must still be inspected for `EXACT`, `CLOSE`, or `DIFFERENT` results.

Small numerical SHAP differences can exchange the order of nearly tied features. For example, a maximum rank difference of one can occur even when the underlying mean absolute SHAP values differ only by a few millionths.

The batch runner does not train models or reproduce repeated-fit stability. The HbA1c bundle contains approximately 70,000 test observations, and its SHAP execution can take approximately 30 minutes.

## Adding another dataset

To add another dataset:

1. Reproduce and select the model in a replication notebook.
2. Produce aligned `X_train`, `X_test`, `y_train`, and `y_test` objects.
3. Define unique, ordered feature names.
4. Supply the fitted binary model, scoring name, output directory, and optional `train_model_fn` to `run_pipeline`.
5. Run the canonical notebook pipeline successfully.
6. Add unit fixtures and an integration test appropriate to the model family.
7. Add a dataset YAML file only after a successful canonical bundle exists.
8. Add the dataset to the batch configuration.
9. Extend the cross-dataset analysis without silently changing established definitions.

## Outputs

Dataset outputs can include:

- A reproducibility bundle.
- Evaluation metrics.
- Native model importance.
- Permutation importance.
- SHAP importance.
- Rank-comparison tables.
- Agreement summaries.
- Representative cases.
- LIME explanations.
- Stability summaries when a retraining callback is available.
- Runtime and dependency metadata.

Canonical notebook outputs are stored under:

```text
results/<dataset>/
```

Independent CLI outputs are stored under:

```text
results/cli_runs/
```

Cross-dataset analysis outputs are stored under:

```text
results/cross_dataset/
```

Generated results, serialized models, caches, editable-install metadata, and local virtual environments remain excluded from version control.

## Methodological limitations

The framework and analysis have several important limitations:

- Raw importance magnitudes are not directly comparable across different XAI methods.
- Rank agreement measures ordering similarity rather than equality of explanation meaning.
- Representative cases are descriptive selections, not a population sample.
- Global/local overlap is based on lexical feature presence and is not formal SHAP–LIME agreement.
- LIME explanations are local surrogate approximations.
- Permutation importance depends on the selected evaluation data and scoring metric.
- SHAP values can vary slightly due to explainer behavior, sampling, software versions, and numerical precision.
- Repeated-fit stability is available only when the original notebook supplies a retraining callback.
- Bundle-based reruns evaluate an existing fitted model and do not repeat model selection.
- HbA1c explanations refer to Patsy design terms and interactions rather than only raw clinical variables.
- The framework does not establish causal relationships or clinical utility.

## Reproducibility record

For the final thesis archive, retain:

- The final Git commit identifier.
- `requirements-xai-lock.txt`.
- `requirements-environment-lock.txt`.
- `pyproject.toml`.
- The five executed replication notebooks.
- The executed cross-dataset analysis notebook.
- Canonical dataset results.
- Selected batch manifests and comparison reports.
- Python and library versions recorded in bundle metadata.
- Any institutionally required thesis deposit files.

Timestamped CLI and batch outputs are reproducibility evidence. They should not automatically replace the canonical notebook-generated thesis results.

## Citation and license

See `mybib.bib` for source-paper and software citations, `main.tex` for the thesis, and `LICENSE` for repository licensing terms.

Add the final thesis citation after institutional publication details are assigned.