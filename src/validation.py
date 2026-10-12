"""Data contracts checked at pipeline boundaries.

Small tables (masters, promotions, costs, outputs) are validated with pandera schemas.
The two large extracts (sales, inventory transactions) are validated with DuckDB SQL so
they never have to fit in memory. Blocking problems raise ``DataValidationError`` with a
readable summary; known, tolerated data-quality issues (returns, duplicates, gaps) are
left to the cleaning step and counted in the data-quality report instead.
"""

from __future__ import annotations

import re
from typing import Any

import duckdb
import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors


class DataValidationError(ValueError):
    """Raised when an input or output breaks its data contract."""


ID = pa.Column(str, pa.Check.str_length(1, 64), nullable=False)

BRANCHES = pa.DataFrameSchema(
    {
        "branch_id": pa.Column(str, unique=True, nullable=False),
        "branch_name": pa.Column(str, nullable=True),
        "region": pa.Column(str, nullable=False),
        "lat": pa.Column(float, pa.Check.in_range(-90, 90)),
        "lon": pa.Column(float, pa.Check.in_range(-180, 180)),
        "size_tier": pa.Column(str, nullable=True),
        "storage_capacity_units": pa.Column(int, pa.Check.ge(0)),
    },
    strict=False,
    coerce=True,
    name="branches",
)

SKUS = pa.DataFrameSchema(
    {
        "sku_id": pa.Column(str, unique=True, nullable=False),
        "category": pa.Column(str, nullable=False),
        "unit_cost": pa.Column(float, pa.Check.gt(0)),
        "list_price": pa.Column(float, pa.Check.gt(0)),
        "holding_cost_rate": pa.Column(float, pa.Check.in_range(0, 1)),
        "lead_time_days": pa.Column(int, pa.Check.in_range(1, 180)),
        "case_pack": pa.Column(int, pa.Check.ge(1)),
        "shelf_life_days": pa.Column(float, pa.Check.gt(0), nullable=True),
        "obsolescence_risk": pa.Column(float, pa.Check.in_range(0, 1)),
        "abc_class": pa.Column(str, pa.Check.isin(["A", "B", "C"])),
    },
    strict=False,
    coerce=True,
    name="skus",
)

PROMOTIONS = pa.DataFrameSchema(
    {
        "sku_id": ID,
        "start_date": pa.Column("datetime64[ns]"),
        "end_date": pa.Column("datetime64[ns]"),
        "discount_pct": pa.Column(float, pa.Check.in_range(0, 0.9)),
    },
    checks=pa.Check(lambda d: d["end_date"] >= d["start_date"], error="end_date before start_date"),
    strict=False,
    coerce=True,
    name="promotions",
)

TRANSFER_COSTS = pa.DataFrameSchema(
    {
        "from_branch": ID,
        "to_branch": ID,
        "cost_per_unit": pa.Column(float, pa.Check.ge(0)),
        "transit_days": pa.Column(int, pa.Check.in_range(0, 60)),
    },
    checks=pa.Check(lambda d: d["from_branch"] != d["to_branch"], error="self-transfer lane"),
    unique=["from_branch", "to_branch"],
    strict=False,
    coerce=True,
    name="transfer_costs",
)

OPEN_PO = pa.DataFrameSchema(
    {
        "branch_id": ID,
        "sku_id": ID,
        "due_date": pa.Column("datetime64[ns]"),
        "qty": pa.Column(int, pa.Check.gt(0)),
    },
    strict=False,
    coerce=True,
    name="open_purchase_orders",
)

FORECASTS = pa.DataFrameSchema(
    {
        "date": pa.Column("datetime64[ns]"),
        "branch_id": ID,
        "sku_id": ID,
        "model": pa.Column(str),
        "forecast": pa.Column(float, pa.Check.ge(0)),
        "p10": pa.Column(float, pa.Check.ge(0)),
        "p50": pa.Column(float, pa.Check.ge(0)),
        "p90": pa.Column(float, pa.Check.ge(0)),
    },
    checks=[
        pa.Check(
            lambda d: (d["p10"] <= d["p50"] + 1e-9) & (d["p50"] <= d["p90"] + 1e-9), error="quantiles not monotone"
        )
    ],
    unique=["date", "branch_id", "sku_id"],
    strict=False,
    coerce=True,
    name="forecasts",
)

POLICY = pa.DataFrameSchema(
    {
        "branch_id": ID,
        "sku_id": ID,
        "on_hand": pa.Column(float, pa.Check.ge(0)),
        "safety_stock": pa.Column(float, pa.Check.ge(0)),
        "reorder_point": pa.Column(float, pa.Check.ge(0)),
        "target_max": pa.Column(float, pa.Check.ge(0)),
        "p_stockout": pa.Column(float, pa.Check.in_range(0, 1)),
        "days_of_supply": pa.Column(float, pa.Check.ge(0)),
    },
    checks=pa.Check(lambda d: d["target_max"] >= d["reorder_point"] - 1e-6, error="target_max < reorder point"),
    unique=["branch_id", "sku_id"],
    strict=False,
    coerce=True,
    name="inventory_policy",
)

TRANSFERS = pa.DataFrameSchema(
    {
        "transfer_id": pa.Column(str, unique=True),
        "from_branch": ID,
        "to_branch": ID,
        "sku_id": ID,
        "qty": pa.Column(float, pa.Check.gt(0)),
        "transfer_cost": pa.Column(float, pa.Check.ge(0)),
        "reason_code": pa.Column(str, pa.Check.isin(["excess->shortfall", "stagnant->demand", "excess->demand"])),
    },
    checks=pa.Check(lambda d: d["from_branch"] != d["to_branch"], error="self-transfer"),
    strict=False,
    coerce=True,
    name="transfer_recommendations",
)


def validate(df: pd.DataFrame, schema: pa.DataFrameSchema) -> pd.DataFrame:
    """Validate ``df`` lazily and raise one readable error listing every failed check."""
    missing = [c for c, col in schema.columns.items() if c not in df.columns and col.required]
    if missing:
        raise DataValidationError(f"{schema.name}: missing required column(s) {missing}")
    try:
        return schema.validate(df, lazy=True)
    except SchemaErrors as e:
        fc = e.failure_cases
        summary = (
            fc.groupby(["column", "check"], dropna=False)
            .size()
            .reset_index(name="rows")
            .sort_values("rows", ascending=False)
            .head(10)
        )
        lines = [f"  - {r.column or '<frame>'}: {r.check} ({r.rows} row(s))" for r in summary.itertuples()]
        raise DataValidationError(f"{schema.name}: {len(fc)} failed check(s)\n" + "\n".join(lines)) from None


SALES_COLUMNS = {"date", "branch_id", "sku_id", "qty"}
TXN_COLUMNS = {"txn_id", "date", "branch_id", "sku_id", "txn_type", "qty"}
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
TXN_TYPES = ("receipt", "transfer_in", "transfer_out", "adjustment", "snapshot")


def validate_large_extracts(
    con: duckdb.DuckDBPyConnection, sales_view: str, txn_view: str, branches: pd.DataFrame, skus: pd.DataFrame
) -> dict[str, Any]:
    """Blocking checks on the big extracts via SQL; returns soft-issue counts for the DQ report."""
    for view in (sales_view, txn_view):  # view names are interpolated below, so they must be plain identifiers
        if not _IDENT.match(view):
            raise ValueError(f"invalid view name {view!r}")
    for view, need in ((sales_view, SALES_COLUMNS), (txn_view, TXN_COLUMNS)):
        cols = {r[0] for r in con.execute(f"DESCRIBE {view}").fetchall()}  # nosec B608 - internal view name
        if not need <= cols:
            raise DataValidationError(f"{view}: missing required column(s) {sorted(need - cols)}")
    con.register("_branches", branches[["branch_id"]])
    con.register("_skus", skus[["sku_id"]])
    q = con.execute(f"""
        SELECT count(*) AS rows,
               count(*) FILTER (WHERE date IS NULL OR branch_id IS NULL OR sku_id IS NULL OR qty IS NULL) AS null_keys,
               count(*) FILTER (WHERE branch_id NOT IN (SELECT branch_id FROM _branches)) AS unknown_branch,
               count(*) FILTER (WHERE sku_id NOT IN (SELECT sku_id FROM _skus)) AS unknown_sku,
               count(*) FILTER (WHERE abs(qty) > 1e7) AS absurd_qty,
               min(date) AS first_date, max(date) AS last_date
        FROM {sales_view}""").fetchone()  # nosec B608 - internal view name, no user input
    if q is None:
        raise DataValidationError("sales extract could not be read")
    sales = dict(
        zip(
            ["rows", "null_keys", "unknown_branch", "unknown_sku", "absurd_qty", "first_date", "last_date"],
            q,
            strict=True,
        )
    )
    t = con.execute(f"""
        SELECT count(*) AS rows,
               count(*) FILTER (WHERE txn_type NOT IN {TXN_TYPES}) AS bad_type,
               count(*) FILTER (WHERE date IS NULL OR branch_id IS NULL OR sku_id IS NULL OR qty IS NULL) AS null_keys,
               count(*) FILTER (WHERE txn_type = 'snapshot' AND qty < 0) AS negative_snapshot
        FROM {txn_view}""").fetchone()  # nosec B608
    if t is None:
        raise DataValidationError("inventory transactions could not be read")
    txn = dict(zip(["rows", "bad_type", "null_keys", "negative_snapshot"], t, strict=True))
    problems = []
    if sales["rows"] == 0:
        problems.append("sales extract is empty")
    for k in ("null_keys", "unknown_branch", "unknown_sku", "absurd_qty"):
        if sales[k]:
            problems.append(f"sales: {sales[k]:,} row(s) with {k.replace('_', ' ')}")
    for k in ("bad_type", "null_keys", "negative_snapshot"):
        if txn[k]:
            problems.append(f"inventory_txn: {txn[k]:,} row(s) with {k.replace('_', ' ')}")
    if problems:
        raise DataValidationError("raw extracts failed validation:\n" + "\n".join(f"  - {p}" for p in problems))
    return {"sales_rows_validated": sales["rows"], "txn_rows_validated": txn["rows"]}
