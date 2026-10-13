# Changelog

## v1.0.0 (2026-10-13)

First complete release of Stockflow: branch × SKU demand forecasting and inventory rebalancing, demonstrated on a synthetic 25-branch, 300-SKU network.

**Forecasting**
- **Models:** global LightGBM plus five simple models, chosen per segment on three backtest months and reported on a fourth.
- **Accuracy:** held-out daily WAPE is 49.8%, against 74.9% for seasonal naive (34% better). Bias is −0.8%, and 84% of actuals fall inside the P10–P90 range.
- **Where it loses:** reported both by group and series by series.

**Inventory policy and rebalancing**
- **Policy:** safety stock, reorder points, target levels, stock-out risk, and excess / stagnant / liquidation flags.
- **Optimiser:** a MILP (PuLP + HiGHS) with arc pruning, parallel SKU batches and a greedy fallback.
- **Result:** 883 transfers save a net $203k ($10k more than greedy). They free $1.1M of excess, raise the expected fill rate from 93.9% to 99.0%, and raise turns from 8.4 to 8.9.
- **Robustness:** sensitivity re-solves keep 66–92% of the plan, and savings stay between $171k and $235k.

**Dashboard**
- **Pages:** transfer recommendations with approve / reject and an audit log; inventory health; forecast explorer with plain-language drivers; forecast accuracy.
- **Transparency:** every impact figure has its formula and assumptions.
- **Security:** optional OIDC sign-in, roles and financial masking.
- **Hosted demo:** a view-only copy is deployed on Vercel.

**Engineering**
- **One command:** `python run_pipeline.py` regenerates everything, and runs are deterministic.
- **Checks:** 64 tests at 94% coverage; ruff, black, mypy, bandit and pip-audit are clean.
- **Docker:** the image is built and run end to end in CI.
- **Scale:** the 10× network (250 branches, 3,000 SKUs) runs on a 4-core, 15 GB machine.

**Documentation:** README, model card, FAQ, 3-minute demo script, data contract, monitoring plan, security notes, audit report and final review. The slide deck (.pptx and .pdf) is built from the pipeline outputs.
