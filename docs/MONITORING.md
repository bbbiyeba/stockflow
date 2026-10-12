# Monitoring and retraining

The forecast feeds safety stock and transfers, so a silently degrading model costs money
twice: it causes stock-outs where it under-forecasts and excess where it over-forecasts.
This is the operating plan.

## What is measured, and when

| Check | Source | Cadence | Alert when | Action |
|---|---|---|---|---|
| Forecast accuracy by ABC class | `python -m src.monitoring --actuals …` compares the published forecast with actual sales | Weekly, as actuals land | Live WAPE > held-out baseline × 1.2 (`monitoring.wape_degradation`) | Investigate; retrain if it persists 2 weeks |
| Forecast bias by ABC class | same | Weekly | \|bias\| > 10% (`monitoring.bias_limit`) | Retrain; check for a demand shift or data issue |
| Interval calibration | same: share of actuals inside P10–P90 | Weekly | Outside 80% ± 10 pts | Recalibrate (retrain refreshes residual quantiles) |
| Demand data drift | PSI of daily demand vs the last 365 training days | Weekly | PSI > 0.25 (`monitoring.psi_limit`) | Retrain; review new SKUs / branches |
| Data quality | `reports/data_quality_report.csv` from every pipeline run | Every run | Missing days, outliers or bad adjustments jump vs the last run | Check the extract before trusting outputs |
| Pipeline health | `outputs/run_manifest.json` (stage status, timings) | Every run | Any stage `ok: false`; runtime up > 50% | Fix before publishing recommendations |
| Optimiser sanity | `reports/optimizer_checks.csv` | Every run | Any non-zero row (the run fails anyway) | Fix before publishing |
| Decision outcomes | `decisions/transfer_decisions.csv` | Monthly | Planner rejection rate > 30% | Review cost/penalty settings in `config.yaml` with planners |

`src.monitoring` exits with code 1 when any alert fires, so a scheduler can page or open a
ticket without parsing the report. Results go to `reports/monitoring_accuracy.csv` and
`reports/monitoring_drift.csv`.

## Retraining

- **Routine:** the nightly or weekly batch (`make run --skip-generate` on fresh extracts) refits every model on the latest history. The global LightGBM model is trained once per run.
- **Incremental:** set `forecast.warm_start: true` to continue boosting the previous model for `warm_start_rounds` rounds instead of refitting. The previous model is only loaded if its SHA-256 matches the last run manifest. Do a full refit at least monthly and after any alert.
- **Promotion gate:** a retrained model's held-out WAPE (`reports/holdout_metrics_overall.csv`) must not be more than 5% worse than the previous run's before its recommendations are published. Compare the two manifests' reports.

## Ownership

| Role | Owns |
|---|---|
| Data science | Alerts on accuracy, bias, calibration and drift; retraining; threshold tuning |
| Inventory planning | Approving or rejecting transfers; the rejection-rate review |
| Data engineering | Extract freshness, data-quality alerts, pipeline failures |
