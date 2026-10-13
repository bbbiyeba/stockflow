# Model card: Stockflow demand forecast

## Summary

| | |
|---|---|
| **Task** | Daily unit demand for every branch × SKU, 28 days ahead, with a P10–P90 range |
| **Model** | One global LightGBM model (Tweedie objective), forecasting recursively; simple models where they win on held-out data |
| **Output** | `outputs/forecasts/` (P10 / P50 / P90 per day), `outputs/forecast_series_stats.parquet` (28-day totals, error σ) |
| **Consumers** | Safety stock and reorder points (`src/policy.py`), transfer optimiser (`src/optimizer.py`), dashboard |
| **Owner** | Bryce Biyeba, Data Science Intern |
| **Code** | `src/features.py`, `src/models.py`, `src/forecast.py` · settings in `config.yaml` → `forecast` |

## Intended use

- **Use it for:** planning inventory at branch × SKU level. That means safety stock, reorder points, target levels and transfer recommendations, which a planner reviews before anything moves.
- **Do not use it for:**
  - single-day staffing or delivery promises, since daily error is about 50%;
  - financial forecasts;
  - any automated action without a person approving it.
- **Users:** branch and inventory planners through the dashboard; supply chain analysts through the output files.

## Training data

All results below are on **synthetic data** from `data/generate.py`, built to behave like a distributor's ERP/WMS extracts:
- 25 branches, 300 SKUs, 4,889 active branch × SKU series;
- daily history from 2023-10-01 to 2026-09-30, three years.

| Treatment | Detail |
|---|---|
| Cleaning (`src/prep.py`) | Duplicates and returns removed, keying-error outliers capped, system-outage days imputed from past weeks only |
| Stock-out days | Excluded from training and scoring, because sales on those days understate demand |
| Training window | Last 540 days before each origin, capped at 4M rows (sampled by series × day on larger networks) |
| Data contract | Every extract is validated on load (`src/validation.py`, [DATA_CONTRACT.md](DATA_CONTRACT.md)) |

## Features

All features are known at forecast time; `tests/test_features.py` guards against leakage.

| Group | Features |
|---|---|
| Demand history | Lags 7/14/28/364/365 days; rolling means over 7, 28, 91 and 365 days; rolling standard deviations; share of days with demand; same-weekday mean of the last 4 weeks; year-over-year 14-day mean |
| Calendar | Day of week and month, week of year, seasonal sine/cosine, holidays and closures, days to and after a closure |
| Promotions and price | Discount, on-promotion flag, days into promotion, 7 days after, net price, price index |
| Branch and product | Branch, SKU, region, branch size, category, unit cost, list price, lead time, case pack, obsolescence risk |

The most important feature is the same-weekday mean of the last 4 weeks (71% of gain), followed by the 28-day standard deviation and mean. See `reports/feature_importance.csv`.

**Per-forecast explanations:** SHAP contributions are shown in plain language in the dashboard's Forecast explorer. The source is `outputs/forecast_explanations.parquet`.

## Evaluation

Evaluation is a rolling-origin backtest with 4 folds of 28 days each:
- the model for each segment (ABC class × demand pattern) is **chosen on folds 1–3**;
- results are **reported on fold 4** (2026-09-03 to 2026-09-30), which played no part in any choice.

| Model, held-out month | Daily WAPE | MAE (units/day) | Bias | MASE | 28-day-total WAPE |
|---|---:|---:|---:|---:|---:|
| **Selected per segment** | **49.8%** | **2.24** | **−0.8%** | 0.72 | 12.9% |
| LightGBM alone | 49.7% | 2.24 | −0.7% | 0.72 | 12.8% |
| Exponential smoothing | 56.6% | 2.55 | +2.5% | 0.77 | 17.2% |
| Croston (SBA) | 57.3% | 2.58 | +0.9% | 0.91 | 18.7% |
| TSB | 57.6% | 2.60 | +4.6% | 0.81 | 19.4% |
| 28-day moving average | 73.5% | 3.31 | +2.9% | 0.93 | 18.0% |
| Seasonal naive (baseline) | 74.9% | 3.38 | +1.2% | 0.96 | 28.0% |

Source: `reports/holdout_metrics_overall.csv`.

- **Against the baseline:** the selected models are 34% more accurate than seasonal naive overall, and better in every segment that has demand (`reports/holdout_selected_vs_naive_by_segment.csv`).
- **Where it does not win:**
  - No branch, category, ABC class or demand pattern is better served by a simple model; exponential smoothing is the closest overall, at 56.6%.
  - Series by series, the selected model has more error than seasonal naive on 19% of series with demand: 4% of A items, 34% of C items and 38% of intermittent series.
  - The detail is in `reports/holdout_where_selected_loses.csv` and on the dashboard's Forecast accuracy page.
- **Bias:** no branch, category, ABC class or pattern is biased by more than 5% (`reports/bias_check.csv`).
- **Ranges:** the P10–P90 range held 84% of held-out actuals against an 80% target (`reports/interval_coverage.csv`).
- **At 10× scale** (250 branches, 3,000 SKUs): held-out WAPE is 50.8%.

## Limitations and risks

| Limitation | Effect | Mitigation |
|---|---|---|
| Synthetic data only | Absolute accuracy will differ on real data | Shadow pilot on real extracts before any decision |
| Daily demand for slow movers is mostly noise | Daily WAPE stays high; for C-class intermittent items it is above 100% | Plan on lead-time and 28-day totals (12.9% error), not single days |
| A-class lumpy series are under-forecast by 7.2% (all folds) | Slightly low safety stock for these positions | Logged as deferred item D13 in [AUDIT.md](../AUDIT.md): extend the bias check to segments and correct per segment |
| Lost sales are excluded, not reconstructed | Demand at chronically short branches may be understated | Held-out bias is reported per branch (`reports/bias_check.csv`); live monitoring checks bias by ABC class |
| Normal error model for safety stock | Crude for very slow movers | Error σ comes from weekly backtest blocks, which capture autocorrelation |
| Ranges over-cover slightly (84% vs 80%) | Conservative | Per-segment scaling is logged as deferred item D7 in [AUDIT.md](../AUDIT.md) |
| New branches and SKUs | No history | Cold-start forecasts borrow from peers (same SKU elsewhere, or the category median) |

## Monitoring and retraining

`python -m src.monitoring --actuals new_sales.csv` compares live accuracy and bias with the held-out baseline and checks for data drift. It exits with code 1 when retraining is due.

Thresholds are in `config.yaml` → `monitoring`; the plan is in [MONITORING.md](MONITORING.md). Every run writes `outputs/run_manifest.json`, which records:
- the config hash;
- package versions;
- SHA-256 hashes of the outputs.

## Fairness and data privacy

- **Data:** the model uses no personal data, only aggregate sales by branch and SKU. On real data, branch-level performance is commercially sensitive. Keep extracts inside the company network, and turn on sign-in and financial masking in the dashboard (see [SECURITY.md](../SECURITY.md)).
- **Bias between branches:** accuracy and bias are reported per branch, so no branch is systematically under-served by the forecast.
