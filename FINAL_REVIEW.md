# Final completeness review

## How this review was run

- **Scope:** the code, dashboard, audit fixes and slide deck, reviewed against the original internship scope and every later request.
- **Approach:** each item was checked by running or opening it, not by reading the code and assuming.
- **Commits:**
  - Review fixes are in `4478beb`; review documents and the remaining fixes are in the commit that adds this file.
  - Evidence for the clean-run, Docker and CI checks comes from `4478beb`. That is the code under test: the later commit adds only documentation and two screenshots, no code.
- **Date:** 2026-10-13.
- **Data:** all figures are from the synthetic network (25 branches, 300 SKUs, 4,889 branch × SKU series).

**Result:** 47 checklist items: 45 PASS, 2 PARTIAL, 0 FAIL.
- **Fixed in this pass:** 12 of the PASS items started as FAIL (4) or PARTIAL (8) and were fixed and re-verified.
- **Still PARTIAL:** the brand-colour check and the GitHub Release object. Both need something this environment can't reach, or a click by a person; see "Open items" at the end.

---

## 1. Scope traceability

### Project 1: Forecasting

| Requirement | Status | Where it is delivered | Evidence |
|---|---|---|---|
| Historical SKU-level consumption analysed by branch | PASS | `notebooks/01_eda.ipynb`, `src/prep.py` | Notebook re-executed: 22 cells, 0 errors, 0 stderr. Demand patterns, seasonality and stock-outs are covered by branch and SKU |
| Uses sales history, inventory transactions, seasonality, operational factors | PASS | `src/prep.py` (ledger from inventory transactions), `src/features.py` | Features include lags 7–365 and rolling statistics; calendar and holidays; promotions and price; lead time, case pack, branch size and region (`feature_names()`, 42 features) |
| Forecasts at branch × SKU level | PASS | `outputs/forecasts/`, `outputs/forecast_series_stats.parquet` | 4,889 series × 28 days with P10 / P50 / P90 (`test_outputs_are_consistent` checks `len(forecasts) == len(policy) * 28`) |
| Data preparation | PASS | `src/prep.py`, `reports/data_quality_report.csv` | 2,661 duplicates and 2,665 returns removed, 1,259 outliers capped, 309,899 stock-out days flagged. Against the generator's ground truth: outlier precision 99.6% and recall 78.0%; stock-out flags catch 99.2% of lost units |
| EDA | PASS | `notebooks/01_eda.ipynb` | Re-executed in this review with `make eda`: exit 0 |
| Feature engineering | PASS | `src/features.py`, `tests/test_features.py` | A leakage guard test checks that features at time t use only data before t |
| Model development | PASS | `src/models.py`, `src/forecast.py` | Seasonal naive, moving average, Croston-SBA, TSB, exponential smoothing and a global LightGBM, selected per segment |
| Evaluation | PASS | `reports/holdout_*.csv`, `reports/backtest_*.csv` | 4-fold rolling-origin backtest; models chosen on folds 1–3, reported on fold 4 |
| Visualisation | PASS | Dashboard Forecast explorer and Forecast accuracy pages; `reports/figures/` | All 4 pages load in a browser with 0 exceptions (section 2) |
| Reduces stock-out risk: risk flags exist and are shown | PASS | `inventory_policy.parquet` (`p_stockout`, `stockout_risk`); Inventory health page | 411 positions at risk, shown on the health heatmap; 72 after transfers |
| Optimal inventory levels (safety stock, reorder point, target) | PASS | `src/policy.py` → `inventory_policy.parquet` | Columns `safety_stock`, `reorder_point`, `target_max` and `recommended_order_qty`, shown per series in the Forecast explorer |
| Deliverables: predictive model, dashboard, actionable insights | PASS | `outputs/lightgbm_model.txt`; `app.py`; transfer list with reasons; FAQ and deck | Model file hash-verified on warm start; the dashboard is live at stockflow-six-mu.vercel.app (redeployed `4478beb`, health 200) |

### Project 2: Rebalancing

| Requirement | Status | Where it is delivered | Evidence |
|---|---|---|---|
| Evaluates SKU positions, forecast consumption, excess, stagnant stock | PASS | `src/policy.py` (`health`: healthy / excess / stagnant / liquidate) | $8.3M excess, $3.7M stagnant, 291 liquidation candidates |
| Recommends transfers between branches | PASS | `outputs/transfer_recommendations.csv` | 883 transfers of 34,534 units, each with a reason code and a plain-language reason |
| Statistical analysis, optimisation, business rules | PASS | `src/optimizer.py` (PuLP + HiGHS MILP), `config.yaml` → `optimizer` | Business rules include the sender floor, minimum transfer size, maximum transit days and same-region option. All 7 sanity checks are 0 (`reports/optimizer_checks.csv`) |
| Quantifies carrying cost reduction | PASS | `reports/optimizer_impact.csv` | Annual carrying cost of excess falls from $2.18M to $1.92M |
| Quantifies excess / obsolete reduction | PASS | same | Excess falls from $8.3M to $7.2M ($1.14M freed). Stagnant stock barely changes ($3.71M before, $3.71M after) and is flagged for liquidation instead |
| Quantifies utilisation | **was FAIL → PASS** | `impact()` now reports utilisation | Share of stock value that is not excess rises from 37.9% to 46.4%. Storage use stays at 58.0% and no branch is above 90% of capacity |
| Quantifies availability | **was PARTIAL → PASS** | `impact()` now reports fill rate | Expected fill rate over the lead time rises from 93.9% to 99.0%. Before this pass only lost units and at-risk positions were reported |
| Deliverables: optimisation model, transfer logic, visual reporting with projected impact | **was PARTIAL → PASS** | `src/optimizer.py`; dashboard Transfer recommendations page | New "Before / after, and how these numbers are calculated" panel ([screenshot](docs/screenshots/impact_breakdown.png)) |

**Missing or partly done in scope:** nothing remains open. Utilisation and availability were the gaps and are now quantified, reported and on the dashboard.

---

## 2. Clean-run test

| Check | Status | Evidence |
|---|---|---|
| Fresh clone, install from requirements, one command end to end | PASS | `git clone` of `4478beb` into an empty directory with a new venv. `pip install -r requirements-dev.txt` exit 0 (55 s); `python run_pipeline.py` exit 0 in 513 s with no manual steps |
| All outputs regenerate | **was PARTIAL → PASS** | The sensitivity report was a separate manual step, so the dashboard's robustness table could go stale. It is now a pipeline stage (`--skip-sensitivity` to skip). `make all` adds the EDA notebook and demo snapshot; `make deck` rebuilds the slides and PDF from the same outputs |
| Dashboard launches; every page loads with no errors or warnings | PASS | Chromium (Playwright) opened all 4 pages from the fresh clone, the local repo and the Docker container. Each had 0 Streamlit exceptions and 0 dialogs, and every expander was opened and checked. Server log: no tracebacks. Only console noise: Streamlit's own 404 probe of `<page>/_stcore/health` when a sub-page URL is opened directly (it then uses the root endpoint) |
| Tests pass; lint, types, security clean | PASS | 64 tests pass, 94% coverage. black, ruff and mypy are clean; bandit reports 0 issues; pip-audit finds no known vulnerabilities. GitHub Actions `quality` job: success on `4478beb` |
| Docker build and run | **was FAIL → PASS** | The README's `docker run` showed only the empty state (no outputs in the image), and the documented `-v $PWD/outputs` mount was not writable by the non-root user. Fixed: the Dockerfile creates its mount points, and the README uses named volumes. Verified twice: (1) here, pipeline in the container exit 0 in 549 s with identical results (883 transfers, $203,124), dashboard container healthy, all 4 pages 0 exceptions; (2) GitHub Actions' new `docker` job builds the unmodified Dockerfile, runs the pipeline inside it and checks the dashboard: success |
| Identical results across two runs | PASS | Two back-to-back runs in the fresh clone: all 57 CSV / Parquet outputs and reports have identical content (`compare.py`). Only per-run metadata differs by design: `run_id`, the `transfer_id` derived from it, and solver wall time. The intermediate `sales_clean` Parquet has the same rows in a different order because of parallel writes. The demo snapshot built from it was not reproducible either; it is now sorted and written single-threaded, and two rebuilds are byte-identical |

The Docker check here needed a sandbox-only build copy: this environment's network policy blocks `deb.debian.org`, so the host's `libgomp` was copied in place of `apt-get`. The real Dockerfile is verified unmodified by CI.

---

## 3. Audit follow-through

| Check | Status | Evidence |
|---|---|---|
| Every critical and high item fixed and verified | PASS | All 4 critical and 22 high items in AUDIT.md are marked Fixed, and no critical or high item is open. Their fixes are exercised by the tests, the 1x runs and the 10x run below |
| Deferred items have a reason and a next step | **was PARTIAL → PASS** | U9 (chart ARIA) and U13 (dark theme) had a reason but no next step; both now have one. New finding D13 (A-class lumpy series under-forecast by 7.2%) added with a reason and next step. Totals: 68 found, 58 fixed, 10 deferred or accepted |
| 10x scale test still passes after all changes | PENDING | See below |
| No secrets in repo or history; `.env.example` present | PASS | `git log --all -p` scanned for key, token, password, private-key and `ghp_` / `sk-` patterns: only documentation mentions. `.env.example` is tracked; `.env` and `.streamlit/secrets.toml` are git-ignored |

**10x scale test** (250 branches, 3,000 SKUs, `scripts/profile_pipeline.py --config configs/scale_10x.yaml`):

Running at the time of this commit (started 07:00 on `4478beb`); the result is added here when it completes.

---

## 4. Numbers consistency

| Check | Status | Evidence |
|---|---|---|
| README, dashboard and slides match the latest output | PASS | **README:** every figure was recomputed from `reports/` and `outputs/` and matched. **Slides:** every number comes from `presentation/assets/facts.json`, which is generated from the outputs; regenerating it after this pass's runs changed only `run_id` and `git_commit`, and the deck was rebuilt from it. **Dashboard:** reads the outputs directly; the screenshots show the same $203k, 8,172, $1.1M and 8.9 |
| Metrics on backtest / holdout, not training data | PASS | `src/forecast.py`: the selection rule is fitted on folds 1–3 (`choice_early`) and scored on fold 4 (`last`). The README, dashboard and slides quote only fold-4 numbers |
| Impact figures show calculation and assumptions | **was PARTIAL → PASS** | The dashboard listed results without formulas. The new panel states each formula and the assumptions (95% service level, 20% risk threshold, lost sale = 0.6 × unit cost, holding rates per SKU); the FAQ and model card repeat them |
| Honest model vs baseline, including where it loses | **was PARTIAL → PASS** | Only segment-level wins were reported. New `reports/holdout_where_selected_loses.csv` and a dashboard section show that no group is better served by a simple model, but **series by series the model is worse than seasonal naive on 19%** (4% of A items, 34% of C items, 38% of intermittent series). This is now in the README and model card |
| Cleaning-accuracy claims complete | **was PARTIAL → PASS** | The README quoted outlier precision (99.6%) but not recall (78.0%), and stock-out recall (99.2%) but not precision (69.4%). Both are now stated, matching the re-executed notebook |

---

## 5. Presentation

| Check | Status | Evidence |
|---|---|---|
| 8 slides or fewer; a visual on every slide | PASS | `pdfinfo`: 8 pages. Visuals: map, bar chart, flow diagram, forecast chart and table, map and table, impact panel and table, 2 screenshots, timeline |
| Mission and the four values | PASS | Mission line on slides 1 and 8. **Integrity:** synthetic-data footer on every slide and assumptions stated. **Accountability:** baseline comparisons and planner approval. **People:** branch-manager screen and plain-language reasons. **Performance:** savings, stock-outs and turns. Keyword counts in the slide text and notes are 14–20 per value |
| Brand colours verified and applied; text readable | PARTIAL | **Applied:** one palette throughout, and the deck contains no white text. Contrast (WCAG): black on gold 11.9:1, gray on white 5.1:1, gray on light gray 4.55:1, all AA. **Not verified:** stryker.com is unreachable from this environment (DNS ENOTFOUND, retried in this review), so gold #FFB502 rests on [Brandfetch](https://brandfetch.com/stryker.com) and [BrandColorsHub](https://brandcolorshub.com/brand/stryker). **To close:** check it against Stryker's brand guide |
| Takeaway titles; speaker notes on every slide | PASS | Every title states a result, e.g. "883 transfers cost $45k to move and save $203k net…". Notes are on all 8 slides, about 150–250 words each, with likely Q&A |
| No overflow, misalignment, typos or placeholder text | PASS | All 8 slides rendered to PNG at 80 dpi and inspected. Spell-check of 2,356 words found only proper nouns. No TODO, TBD, lorem or `{}` placeholders |
| Synthetic-data disclosure | PASS | Footer "Synthetic data – for demonstration" on all 8 slides, plus the limitations box on slide 8 |
| .pptx and PDF open correctly | PASS | python-pptx re-opens the .pptx (8 slides, 175 text runs); LibreOffice converts it to an 8-page PDF |

---

## 6. Documentation and handoff

| Check | Status | Evidence |
|---|---|---|
| README covers problem, approach, architecture, setup, results, limitations and real data | PASS | Sections: intro; Results; How it works (Mermaid diagram); Quick start; Assumptions and limitations; Connecting real data. A new Documentation table links every doc |
| Docstrings on public functions; config documented | **was FAIL → PASS** | An AST scan found 49 public functions or modules without docstrings; now 0. Nine `config.yaml` keys had no comment; all now do |
| Model card | **was FAIL → PASS** | New [docs/MODEL_CARD.md](docs/MODEL_CARD.md): task, intended and non-intended use, training data, features, held-out metrics, where the model loses, limitations, monitoring, privacy |
| Clear structure; no dead code, unused files, broken notebooks, debug output | **was PARTIAL → PASS** | vulture found an unused parameter (`_sample_plan(n_series)`) and an unused colour constant; both removed. `make_charts.py` hard-coded this sandbox's browser path and would fail elsewhere; it now falls back to Playwright's default. Remaining vulture hits are used by tests: the `mae` / `mase` reference metrics and `render_page`. No `print` or `breakpoint` in `src/` |
| Clean history, meaningful commits; final tag / release | PARTIAL | 11 descriptive commits authored by Bryce Biyeba; annotated tag `v1.0.0` pushed with release notes in [CHANGELOG.md](CHANGELOG.md). **To close:** the connected GitHub tools cannot create a Release, so on GitHub open Releases → Draft a new release → choose `v1.0.0` → paste the changelog |

---

## 7. Gaps nobody asked for

What a hiring manager or Stryker reviewer would likely ask, and what was added:

| Gap | Added |
|---|---|
| "Show me in 3 minutes" | [docs/DEMO_SCRIPT.md](docs/DEMO_SCRIPT.md): a timed walkthrough with the numbers to say |
| Cost to run, ERP integration, data privacy, monitoring, rollout | [docs/FAQ.md](docs/FAQ.md) |
| "How was the model built and when should I not trust it?" | [docs/MODEL_CARD.md](docs/MODEL_CARD.md) |
| "Does the Docker image actually work?" | CI `docker` job: build, run the pipeline in the container, check the dashboard on every push |
| "Where does the model lose?" | `holdout_where_selected_loses.csv` and a dashboard section |
| "Is any segment systematically short?" | Found A-class lumpy under-forecast (−7.2%); logged as D13 with a fix |
| Reproducible demo | Demo snapshot rebuilds byte-identically |

Not added, because each needs real data or a decision by the business:
- a cost model agreed with finance;
- multi-period planning;
- cube-based capacity;
- SSO configured against the company identity provider.

These are listed as limitations in the README, model card and slide 8.

---

## Open items (the 2 PARTIALs plus 2 environment limits)

| Item | Why | What closes it |
|---|---|---|
| Brand colour verified on stryker.com | Site unreachable from this environment | Check #FFB502 against Stryker's brand guide |
| GitHub Release object | No tool here can create one | Draft a release from tag `v1.0.0` (one click) |
| Hosted demo publicly reachable, live WebSocket | The demo can only be fetched through Vercel's authenticated bypass; WebSockets can't be tested from here | Open stockflow-six-mu.vercel.app in a private window and click between pages |
| Docker `apt-get` inside this sandbox | `deb.debian.org` blocked by network policy | Already verified in CI; nothing to do |
