# Optional HbA1c grouped-permutation stability experiment

This folder contains an isolated feasibility experiment for the optional HbA1c
extension suggested during supervision. It refits the saved 91-column
Statsmodels logistic design on stratified bootstrap samples and permutes all
dummy columns belonging to the same Patsy term together.

The experiment does **not** replace the current thesis results automatically.
It reads the trusted bundle in `results/hba1c/bundle` and writes each run to a
new timestamped folder under `results/optional_hba1c_grouped_stability`. It does
not write to `results/hba1c` or alter the HbA1c notebook.

The saved analytical configuration uses the complete cohort as both the fitting
and explanation/evaluation matrix. The optional results are therefore
in-sample and should not be described as directly comparable to held-out
stability estimates.

## Run from the repository root

First validate the bundle, the stored Patsy term definitions, and the path
protection. This loads the trusted local pickle but creates no output and fits
no model:

```powershell
python experiments/hba1c_optional/hba1c_grouped_stability.py --inspect-only
```

Run the small pilot (5 bootstrap refits, 3 grouped-permutation repeats):

```powershell
python experiments/hba1c_optional/hba1c_grouped_stability.py
```

Open the new timestamped folder and inspect `pilot_assessment.json`. The
decision is based on optimiser convergence, failures, and projected runtime.
Five bootstrap samples do not establish statistical convergence of the
stability estimates.

Only after an acceptable pilot, start the full run explicitly:

```powershell
python experiments/hba1c_optional/hba1c_grouped_stability.py --mode full --confirm-full
```

Full mode defaults to 100 bootstrap refits, 10 grouped-permutation repeats per
refit, and 20 repeats for the current fitted model. It saves progress after
every bootstrap, so an interrupted run retains completed results. Every retry
creates another timestamped folder; no earlier pilot or full run is overwritten.

## Observed full run

The retained full run requested 100 bootstrap refits. Fifty-eight converged and 42 failed. The median total time per attempted bootstrap was 22.10 seconds, and the observed runtime was computationally feasible. The pairwise Spearman estimate from the 58 successful fits is exploratory only and is not part of the canonical thesis stability comparison: excluding 42 failed fits could select a non-representative subset of resamples. The run is retained as sensitivity evidence under `results/optional_hba1c_grouped_stability/`.

## Output files

- `term_groups.csv`: exact mapping from Patsy terms to design columns.
- `grouped_permutation_current_model.csv`: grouped importance for the current
  fitted model using ROC-AUC.
- `bootstrap_grouped_importance.csv`: grouped importance for each successful
  refit using negative log loss.
- `bootstrap_grouped_summary.csv`: importance, rank, and top-k summaries.
- `bootstrap_pairwise_stability.csv`: pairwise Spearman correlation and top-k
  overlap between successful refits.
- `bootstrap_run_log.csv`: fit convergence, warnings, failures, and stage times.
- `pilot_assessment.json`: computational feasibility decision and projected
  full-run time.
- `run_metadata.json`: configuration, provenance, limitations, and final state.

Keep the canonical thesis results unless the full experiment completes and its
scientific interpretation is accepted. A failed or slow pilot is evidence for
retaining the current limitation statement.
