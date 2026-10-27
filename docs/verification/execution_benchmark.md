# Execution-time benchmark

Measured on 2026-10-23 on a Windows 11 workstation with an AMD Ryzen 7 3700U CPU and CPython 3.12.6.

Each trusted saved bundle was rerun once with its aligned saved matrices and fitted model. Runs used up to ten observations per available representative-case category and 20 LIME seeds per case. Model fitting and bootstrap-refit stability were excluded because saved fitted bundles do not contain the retraining callback. Stage durations were captured with Python's monotonic high-resolution timer and written automatically to each run's `execution_times.json`.

The values are descriptive single-run measurements. They depend on dataset size, feature count, explainer, model, configuration, hardware, and concurrent workload.

| Dataset | Global (s) | Local (s) | Export (s) | Total (s) |
|---|---:|---:|---:|---:|
| Heart | 14.6 | 37.9 | 0.2 | 52.7 |
| Stroke | 5.5 | 25.8 | 0.2 | 31.6 |
| Diabetes | 7.9 | 213.2 | 0.3 | 221.4 |
| CKD | 10.2 | 79.6 | 0.2 | 90.0 |
| HbA1c | 2,122.9 | 77.8 | 10.8 | 2,211.5 |
