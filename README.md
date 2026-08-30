# Reproducible XAI Framework for Cross-Dataset Replication

This repository contains five paper-replication notebooks and a reusable binary-classification XAI framework. Replication notebooks own data preparation, model selection, training, and the optional retraining callback. The framework consumes the selected fitted model and prepared matrices, then produces Native, Permutation, SHAP, LIME, rank-agreement, representative-case, and optional stability outputs.

## Supported scope

- Binary classification with class labels `0` and `1`.
- Scikit-learn estimators and the tested statsmodels binary model adapter.
- NumPy arrays, DataFrames, and already-created Patsy design matrices.
- Stability only when the notebook supplies `train_model_fn`.

It does not claim general support for regression, multiclass prediction, arbitrary formulas, or untrusted model files.

## Reproducible environment

The final tested environment is Windows CPython 3.12.6. Create a dedicated virtual environment and install the exact lock:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-xai-lock.txt
python -m pip install -e . --no-deps
```

`requirements-xai-lock.txt` records the focused, exact XAI runtime. `requirements-environment-lock.txt` is the complete archival `pip freeze` of the Windows interpreter, including unrelated packages; use it for provenance, not as the default clean installation. `pyproject.toml` exposes package and command-line entry points. Future bundles and CLI manifests record Python, platform, and relevant library versions; old bundles cannot acquire provenance retroactively.

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
```

The older notebooks under `src/notebooks/` are prototypes and are not the canonical workflow. Empty `__init__.py` files are valid package markers. Empty legacy result folders are not used by the dataset-oriented exporter.

## Canonical workflow

1. Open each notebook in `src/replication/` with the thesis environment.
2. Restart the kernel and run all cells.
3. Each notebook calls `run_pipeline(config)` and writes `results/<dataset>/`.
4. After all five datasets succeed, restart and run `src/analysis/xai_results_analysis_corrected.ipynb`.
5. Run the complete automated tests.

The model-selection evidence stays in replication notebooks. Do not replace canonical results with CLI reruns without an explicit reviewed decision.

## Tests

```powershell
python -m pytest tests -v --run-integration
```

The integration suite exercises a real small SHAP/LIME pipeline. Passing tests support only the tested scope; they are not clinical validation or proof of statistical explanation fidelity.

## Bundle CLI

Validate a trusted saved bundle without deserializing its model:

```powershell
python -m xai_framework --config src/xai_framework/configs/heart.yaml --validate-only
```

Rerun XAI from the fitted model:

```powershell
python -m xai_framework --config src/xai_framework/configs/heart.yaml --trust-model
```

Pickle/joblib files can execute code. Use `--trust-model` only for bundles you created and control. The configured output path contains UTC `{timestamp}`, so every invocation receives a new directory and canonical results are not overwritten. Validation and execution performed in separate invocations therefore show different prospective output timestamps; this is expected.

The CLI does not retrain models and cannot reproduce repeated-fit stability because bundles do not serialize `train_model_fn`. It validates data and feature order, preserves statsmodels design columns, checks LIME case coverage, and writes `cli_run.json` with runtime provenance.

## Result comparison

Compare canonical CSV exports with a CLI rerun using numerical tolerances:

```powershell
python -m xai_framework.compare_results `
  --reference results/heart `
  --candidate results/cli_runs/heart_YYYYMMDDTHHMMSSZ `
  --output results/comparisons/heart.json
```

Statuses are `EXACT`, `CLOSE`, `DIFFERENT`, `SCHEMA_DIFFERENT`, `SHAPE_DIFFERENT`, or `MISSING`. SHAP and LIME can vary slightly; review differences rather than weakening tolerances until a report passes. The comparison covers exported CSV tables, not model equivalence or explanation validity.

## Adding another dataset

1. Reproduce and select the model in a replication notebook.
2. Produce aligned `X_train`, `X_test`, `y_train`, `y_test`, and unique ordered `feature_names`.
3. Supply a fitted binary model, scoring name, output directory, and optional `train_model_fn` to `run_pipeline`.
4. Add unit fixtures and a real integration test appropriate to the model family.
5. Add a CLI YAML only after a successful canonical bundle exists.
6. Extend cross-dataset analysis without changing established definitions silently.

## Outputs and limitations

Dataset outputs include a reproducibility bundle, global importance tables, rank comparison/agreement, representative cases, LIME explanations, and stability tables when available. Raw importance magnitudes are not comparable across methods. Representative cases are descriptive selections rather than a population sample. Global/local overlap in analysis is lexical presence, not formal SHAP–LIME agreement. HbA1c explanations refer to Patsy design terms and interactions, and its bundle-based run does not add stability.

Generated results, model pickles, caches, and local environments remain excluded from version control. Record the final commit, exact lock, notebook outputs, and thesis archive according to institutional requirements.

## Citation and license

See `mybib.bib` for source-paper and software citations, `main.tex` for the thesis, and `LICENSE` for repository licensing terms. Add the final thesis citation after institutional publication details are assigned.
