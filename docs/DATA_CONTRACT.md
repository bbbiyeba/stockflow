# Connecting real ERP / WMS data

The pipeline reads plain files from `paths.raw_dir`. To run it on company data, produce
these extracts (CSV, or Parquet folders for the two large tables), point `config.yaml` at
them, and run `python run_pipeline.py --skip-generate`. Every file is validated on the way
in (`src/validation.py`); a contract violation stops the run with a readable message.

| File | Grain | Required columns | Typical source |
|---|---|---|---|
| `branches.csv` | one row per branch | `branch_id, branch_name, region, lat, lon, size_tier, storage_capacity_units` | Location master; capacity from WMS bin capacity |
| `skus.csv` | one row per SKU | `sku_id, category, unit_cost, list_price, holding_cost_rate, lead_time_days, case_pack, shelf_life_days, obsolescence_risk, abc_class` | Item master; holding rate from finance |
| `sales_daily.csv` or `sales_daily/*.parquet` | branch × SKU × day | `date, branch_id, sku_id, qty` | ERP invoice / pick lines aggregated per day. **Include explicit zero rows** for carried items; a missing row is treated as an outage and imputed. |
| `inventory_txn.csv` or `inventory_txn/*.parquet` | one row per movement | `txn_id, date, branch_id, sku_id, txn_type, qty, ref` with `txn_type` in receipt, transfer_in, transfer_out, adjustment, snapshot | WMS movement history plus periodic on-hand snapshots (at least weekly, and one on the last day) |
| `open_purchase_orders.csv` | one row per open PO line | `branch_id, sku_id, due_date, qty` | ERP open PO lines |
| `promotions.csv` | one row per promotion | `sku_id, start_date, end_date, discount_pct` | Merchandising calendar, including future promotions |
| `transfer_costs.csv` | one row per lane | `from_branch, to_branch, cost_per_unit, transit_days` (`distance_km` optional) | Freight rate card or a cost model |

## Mapping notes

- **Demand vs sales:** use shipped or invoiced quantity. Stock-out days are detected from the rebuilt ledger and excluded from training, so don't try to un-censor demand upstream.
- **Returns:** may stay in as negative quantities; they are split out during cleaning.
- **Units:** all quantities in the stocking unit. Convert cases to eaches before export.
- **Capacity:** `storage_capacity_units` is total units. If your WMS tracks cube or pallet positions, convert with an average units-per-position per category.
- **Costs:** `holding_cost_rate` and `obsolescence_risk` are annual rates. The optimiser's stock-out penalty and other economics are in `config.yaml` → `optimizer`; agree them with finance and planning before the first run.
- **Scale:** set `data.raw_format`-style Parquet folders for networks above roughly 50k series, and size `prep.bucket_series` so a bucket fits comfortably in memory (30k series ≈ 4 GB peak).
- **Credentials:** if you add a step that queries the ERP directly, read DSNs from the environment (`.env.example`) and use parameterised queries only.
