# Tests for the existing XAI framework

Copy the contents of `tests/` into your project's `thesis/tests/`. Replace the
four empty test files; add conftest.py, test_ranking.py, and test_exporters.py.
Do not replace any non-empty tests without reviewing them first.

Use the same Python environment as your working replication notebooks. Install
pytest there if needed:

```text
python -m pip install pytest
```

From the thesis project root:

```text
python -m pytest tests -v
python -m pytest tests -v --run-integration
```

The first command skips the real SHAP/LIME smoke test. The runner orchestration
test uses a mock; it does not prove numerical explainer correctness. The second
command runs real explainers on small synthetic data. All exports use pytest's
temporary directory, not your thesis results. Existing framework dependencies
(including SHAP and LIME) must already be installed in this environment.

## Known expected failure — do not hide it

`test_tied_importances_receive_average_ranks` tests correct tie handling. The
current core/ranking.py assigns sequential ranks after sorting, even for ties.
It should fail until that production bug is corrected. Replace the corresponding
three sequential-rank assignments with:

```python
shap_rank["SHAPRank"] = shap_rank["MeanAbsSHAP"].rank(
    ascending=False, method="average"
)
perm_rank["PermutationRank"] = perm_rank["Importance"].rank(
    ascending=False, method="average"
)
native_rank["NativeRank"] = native_rank["Importance"].rank(
    ascending=False, method="average"
)
```

These are suggested production changes, not changes applied by this test pack.
After changing ranking code, regenerate affected rank-agreement results before
using them in analysis. Do not weaken the test to retain arbitrary tie order.

## Scope and verification

Tests cover ranking, native importance, representative-case positions, binary
metrics, selector mechanics, CSV/JSON exports, bundle roundtrip, runner order,
and an optional real pipeline smoke test. They are a starting suite, not full
validation of clinical use, statsmodels design matrices, repeated-fit stability,
regression, multiclass tasks, or statistical explanation fidelity. Selector tests
use synthetic fixtures: select models using validation data, not final test data.

The files were syntax checked but not executed in the author's environment:
pytest/scikit-learn/scipy/SHAP/LIME were not installed there. Run both commands
in your working thesis environment and inspect failures; do not assume all pass.
