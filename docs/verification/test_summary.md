# Final automated verification

- Date: 2026-10-28
- Command: `python -m coverage run -m pytest -q --run-integration`
- Result: 62 passed, 0 failed
- Warnings: 1 expected scikit-learn warning from the deliberate single-class rejection test
- Runtime: 119.78 seconds
- Statement coverage: 595 of 724 statements (82%)
- Separate branch-enabled coverage: 78%

The full machine-readable statement-coverage report is stored in `coverage.json`, and the module-level text report is stored in `coverage_report.txt`.

