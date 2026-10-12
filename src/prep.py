"""Data preparation and cleaning pipeline.

Documented assumptions
----------------------
A1. Series universe: a branch x SKU series exists if it appears in the sales extract.
    Days before its first appearance are "not active" (NaN) and are never imputed.
A2. Exact duplicate rows are extract artefacts and are dropped. Remaining duplicate keys
    (same date/branch/SKU, different qty) are summed, i.e. treated as split postings.
A3. Negative quantities are customer returns. Returns are not negative demand, so the
    demand signal is floored at 0 and the return is kept in ``qty_returned``.
A4. Missing (series, day) rows inside the active window are extract gaps (system outages),
    not zero sales: the extract carries explicit zero rows for carried items. Gaps are
    imputed with the mean of the same weekday over the previous ``impute_weeks`` weeks.
    Only *past* weeks are used, so an imputed value never contains information from after
    the day it fills (no look-ahead into backtest origins). Gaps at the very start of a
    series, where no past exists, fall back to the following weeks.
A5. Outliers: a value is a keying error if it exceeds ``outlier_q99_multiple`` x the
    series' 99th percentile of positive sales and ``outlier_min_qty``. Flagged values are
    treated like gaps (imputed), because the true value is unknown.
A6. Inventory ledger: daily end-of-day on-hand is rebuilt from the last weekly snapshot plus
    receipts, transfers and adjustments minus sales; every snapshot resets the ledger, so
    errors cannot accumulate beyond a week.
A7. A negative adjustment that is both larger than the stock on hand at the previous
    snapshot and larger than ``max_plausible_adjustment`` is treated as a keying error and
    excluded from the ledger (small shrink right after a receipt is legitimate).
A8. Days whose rebuilt end-of-day on-hand is <= 0 are stock-out days: sales on those days
    are censored demand. They stay in the history used for features but are excluded from
    model training targets and from forecast evaluation.
A9. The latest snapshot (last day of history) is the current on-hand. Open purchase orders
    are treated as firm inbound supply on their due dates.

Scale
-----
Raw extracts (CSV or Parquet) are ingested with DuckDB, validated with SQL, and written
to ``processed/ingest/<table>/bucket=k/`` where a bucket holds complete SKUs of about
``prep.bucket_series`` series. Each bucket is then cleaned independently in memory, so
peak memory depends on the bucket size, not on the network size.

Outputs (``paths.processed_dir``): ``series.parquet``, ``buckets/bucket_XXX.npz`` (dense
series x day matrices per bucket), ``sales_clean/`` (Parquet partitioned by branch),
``inventory_status.parquet`` and ``promotions_daily.npz``. A data-quality report is
written to ``reports/``.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import time
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import Paths, get_paths, load_config  # noqa: E402
from src.runlog import get_logger  # noqa: E402
from src.validation import (
    BRANCHES,
    OPEN_PO,
    PROMOTIONS,
    SKUS,
    TRANSFER_COSTS,  # noqa: E402
    DataValidationError,
    validate,
    validate_large_extracts,
)

log = get_logger("prep")


# --------------------------------------------------------------------------- loading
def load_masters(paths: Paths) -> dict[str, pd.DataFrame]:
    """Read and validate the small raw tables."""
    raw = paths.raw
    try:
        out = {
            "branches": validate(pd.read_csv(raw / "branches.csv"), BRANCHES),
            "skus": validate(pd.read_csv(raw / "skus.csv"), SKUS),
            "promotions": validate(pd.read_csv(raw / "promotions.csv"), PROMOTIONS),
            "transfer_costs": validate(pd.read_csv(raw / "transfer_costs.csv"), TRANSFER_COSTS),
            "open_po": validate(pd.read_csv(raw / "open_purchase_orders.csv"), OPEN_PO),
        }
    except FileNotFoundError as e:
        raise DataValidationError(f"missing input file: {Path(str(e.filename)).name}") from None
    except (pd.errors.ParserError, UnicodeDecodeError) as e:
        raise DataValidationError(f"could not parse an input CSV: {type(e).__name__}") from None
    return out


def _register(con: duckdb.DuckDBPyConnection, raw: Path, name: str, columns: dict[str, str], view: str) -> None:
    """Register ``name`` (a Parquet folder or a CSV file) as a typed DuckDB view.

    Uses the relational API, so file paths are passed as arguments and never spliced into SQL.
    """
    folder, csv = raw / name, raw / f"{name}.csv"
    casts = ", ".join(f"CAST({c} AS {t}) AS {c}" for c, t in columns.items())  # constants defined below
    if folder.is_dir() and any(folder.glob("*.parquet")):
        rel = con.read_parquet(str(folder / "*.parquet"))
    elif csv.exists():
        rel = con.read_csv(str(csv), header=True, dtype=columns)
    else:
        raise DataValidationError(f"missing input: {name}.csv or {name}/*.parquet")
    missing = set(columns) - set(rel.columns)
    if missing:
        raise DataValidationError(f"{name}: missing required column(s) {sorted(missing)}")
    rel.project(casts).create_view(view)


_MEMORY = re.compile(r"^\d+(\.\d+)?\s*(KB|MB|GB|TB|KiB|MiB|GiB|TiB)$")


SALES_TYPES = {"date": "DATE", "branch_id": "VARCHAR", "sku_id": "VARCHAR", "qty": "BIGINT"}
TXN_TYPES = {
    "txn_id": "VARCHAR",
    "date": "DATE",
    "branch_id": "VARCHAR",
    "sku_id": "VARCHAR",
    "txn_type": "VARCHAR",
    "qty": "BIGINT",
    "ref": "VARCHAR",
}


def ingest(
    paths: Paths, masters: dict[str, pd.DataFrame], cfg: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DatetimeIndex, dict[str, Any]]:
    """Validate the large extracts and write them as SKU-bucketed Parquet.

    Returns the SKU -> bucket map, the full history date range and validation counts.
    """
    pc = cfg["prep"]
    target = int(pc.get("bucket_series", 30_000))
    out = paths.processed / "ingest"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    memory = str(pc.get("duckdb_memory_limit", "4GB"))
    if not _MEMORY.match(memory):
        raise ValueError(f"prep.duckdb_memory_limit must look like '4GB', got {memory!r}")
    con = duckdb.connect(
        config={
            "memory_limit": memory,
            "preserve_insertion_order": False,
            "temp_directory": str(paths.processed / "_duckdb_tmp"),
        }
    )
    try:
        _register(con, paths.raw, "sales_daily", SALES_TYPES, "sales")
        _register(con, paths.raw, "inventory_txn", TXN_TYPES, "txn")
        stats = validate_large_extracts(con, "sales", "txn", masters["branches"], masters["skus"])
        first, last = con.execute("SELECT min(date), max(date) FROM sales").fetchone()
        per_sku = con.execute(
            "SELECT sku_id, count(DISTINCT branch_id) AS n FROM sales " "GROUP BY sku_id ORDER BY sku_id"
        ).df()
        bmap = per_sku.assign(bucket=(per_sku["n"].cumsum() - 1) // target)[["sku_id", "bucket"]]
        con.register("_bucket_map", bmap)
        for view in ("sales", "txn"):
            # no ORDER BY: rows are pivoted by key later, and a global sort would spill tens of GB at scale
            rel = con.table(view).join(con.table("_bucket_map"), "sku_id")
            rel.write_parquet(str(out / view), partition_by=["bucket"])
    except duckdb.Error as e:  # malformed CSVs (bad dates, non-numeric qty, ...) end up here
        raise DataValidationError(f"could not read raw extracts: {str(e).splitlines()[0][:200]}") from None
    finally:
        con.close()
        shutil.rmtree(paths.processed / "_duckdb_tmp", ignore_errors=True)
    dates = pd.date_range(pd.Timestamp(first), pd.Timestamp(last), freq="D")
    stats["buckets"] = int(bmap["bucket"].nunique())
    return bmap, dates, stats


def read_bucket(paths: Paths, view: str, bucket: int) -> pd.DataFrame:
    """Load one ingested bucket with dictionary-encoded ids (low memory)."""
    folder = paths.processed / "ingest" / view / f"bucket={bucket}"
    df = pq.read_table(folder, read_dictionary=["branch_id", "sku_id"]).to_pandas()
    df["date"] = pd.to_datetime(df["date"])
    return df


# --------------------------------------------------------------------------- helpers
class _quiet_nanmean:
    """Context manager silencing 'mean of empty slice' warnings."""

    def __enter__(self) -> None:
        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter("ignore", category=RuntimeWarning)

    def __exit__(self, *exc: Any) -> None:
        self._w.__exit__(*exc)


def _shift(Y: np.ndarray, s: int) -> np.ndarray:
    """Shift columns right by ``s`` (left if negative), filling with NaN."""
    out = np.full_like(Y, np.nan)
    if s > 0:
        out[:, s:] = Y[:, :-s]
    elif s < 0:
        out[:, :s] = Y[:, -s:]
    else:
        out[:] = Y
    return out


def same_weekday_impute(Y: np.ndarray, weeks: int) -> np.ndarray:
    """Fill NaNs with the mean of the same weekday over the previous ``weeks`` weeks (A4).

    Backward-looking only. Remaining gaps (start of history) use the following weeks,
    then the series' weekday mean, then 0.
    """
    with _quiet_nanmean():
        past = np.nanmean(np.stack([_shift(Y, 7 * k) for k in range(1, weeks + 1)]), axis=0)
        out = np.where(np.isnan(Y), past, Y)
        if np.isnan(out).any():
            future = np.nanmean(np.stack([_shift(Y, -7 * k) for k in range(1, weeks + 1)]), axis=0)
            out = np.where(np.isnan(out), future, out)
        if np.isnan(out).any():
            T = Y.shape[1]
            dow = np.arange(T) % 7
            for d in range(7):
                cols = dow == d
                m = np.nan_to_num(np.nanmean(Y[:, cols], axis=1))
                block = out[:, cols]
                out[:, cols] = np.where(np.isnan(block), m[:, None], block)
    return out


def flag_outliers(Y: np.ndarray, q99_multiple: float, min_qty: float) -> np.ndarray:
    """Boolean mask of keying-error outliers (A5)."""
    pos = np.where(Y > 0, Y, np.nan)
    with _quiet_nanmean():
        q99 = np.nanquantile(pos, 0.99, axis=1)
    thresh = np.maximum(np.nan_to_num(q99, nan=0.0) * q99_multiple, min_qty)
    return np.nan_to_num(Y, nan=0.0) > thresh[:, None]


@dataclass
class Index:
    """Maps (branch_id, sku_id, date) to (row, col) of the dense bucket matrices."""

    branch_cats: pd.Index
    sku_cats: pd.Index
    keys: np.ndarray  # sorted int64 series keys = branch_code * n_sku + sku_code
    d0: pd.Timestamp
    T: int

    @property
    def n(self) -> int:
        return len(self.keys)

    def key(self, df: pd.DataFrame) -> np.ndarray:
        b = pd.Categorical(df["branch_id"].astype(str), categories=self.branch_cats).codes.astype(np.int64)
        s = pd.Categorical(df["sku_id"].astype(str), categories=self.sku_cats).codes.astype(np.int64)
        return np.where((b < 0) | (s < 0), -1, b * len(self.sku_cats) + s)

    def locate(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Row, column and validity mask for each record."""
        k = self.key(df)
        if self.n == 0:
            z = np.zeros(len(df), dtype=np.int64)
            return z, z, np.zeros(len(df), dtype=bool)
        row = np.minimum(np.searchsorted(self.keys, k), self.n - 1)
        col = (df["date"] - self.d0).dt.days.to_numpy()
        ok = (k >= 0) & (self.keys[row] == k) & (col >= 0) & (col < self.T)
        return row, col, ok


def to_wide(idx: Index, df: pd.DataFrame, value: str, fill: float = np.nan, agg: str = "sum") -> np.ndarray:
    """Pivot a long (branch_id, sku_id, date, value) frame into a dense float64 matrix."""
    out = np.full((idx.n, idx.T), fill, dtype=np.float64)
    if df.empty:
        return out
    row, col, ok = idx.locate(df)
    rows, cols, vals = row[ok], col[ok], df[value].to_numpy()[ok].astype(np.float64)
    if agg == "sum":
        if np.isnan(fill):
            out[rows, cols] = 0.0
        np.add.at(out, (rows, cols), vals)
    else:  # last value wins
        out[rows, cols] = vals
    return out


# --------------------------------------------------------------------------- cleaning steps
def clean_sales(sales: pd.DataFrame, cfg: dict[str, Any] | None = None) -> tuple[pd.DataFrame, dict[str, int]]:
    """Dedupe and split returns (A2, A3). Returns the tidy frame and counts for the DQ report."""
    stats: dict[str, int] = {"raw_rows": len(sales)}
    dup = sales.duplicated(["date", "branch_id", "sku_id", "qty"])
    stats["exact_duplicates_dropped"] = int(dup.sum())
    s = sales.loc[~dup, ["date", "branch_id", "sku_id", "qty"]]
    stats["duplicate_keys_summed"] = int(s.duplicated(["date", "branch_id", "sku_id"]).sum())
    if stats["duplicate_keys_summed"]:
        s = s.groupby(["date", "branch_id", "sku_id"], as_index=False, observed=True)["qty"].sum()
    s = s.reset_index(drop=True)
    stats["negative_qty_rows"] = int((s["qty"] < 0).sum())
    s["qty_returned"] = (-s["qty"]).clip(lower=0)
    s["qty"] = s["qty"].clip(lower=0)
    return s, stats


def build_ledger(
    idx: Index,
    txn: pd.DataFrame,
    sales_used: np.ndarray,
    max_plausible_adjustment: float = 10.0,
) -> tuple[np.ndarray, dict[str, int]]:
    """Rebuild daily end-of-day on-hand from snapshots + movements (A6, A7)."""
    stats: dict[str, int] = {}
    snaps = txn[txn["txn_type"] == "snapshot"]
    moves = txn[txn["txn_type"] != "snapshot"]

    adj = moves[moves["txn_type"] == "adjustment"].sort_values("date")
    snap_sorted = snaps.sort_values("date")[["date", "branch_id", "sku_id", "qty"]].rename(
        columns={"qty": "prior_snapshot"}
    )
    for c in ("branch_id", "sku_id"):  # merge_asof needs identical key dtypes
        adj = adj.assign(**{c: adj[c].astype(str)})
        snap_sorted = snap_sorted.assign(**{c: snap_sorted[c].astype(str)})
    prior = pd.merge_asof(adj, snap_sorted, on="date", by=["branch_id", "sku_id"], allow_exact_matches=False)
    bad = (
        (prior["qty"] < 0)
        & (-prior["qty"] > prior["prior_snapshot"].fillna(np.inf))
        & (-prior["qty"] > max_plausible_adjustment)
    )
    bad_ids = set(prior.loc[bad, "txn_id"])
    stats["bad_adjustments_excluded"] = len(bad_ids)
    stats["bad_adjustment_units"] = int(-prior.loc[bad, "qty"].sum())
    moves = moves[~moves["txn_id"].isin(bad_ids)]

    net = to_wide(idx, moves, "qty", fill=0.0)
    net -= sales_used
    snap = to_wide(idx, snaps, "qty", agg="last")
    stats["snapshots"] = int((~np.isnan(snap)).sum())

    C = np.cumsum(net, axis=1)
    del net
    base = snap - C
    eod = pd.DataFrame(base).ffill(axis=1).bfill(axis=1).to_numpy() + C  # A6
    pred = pd.DataFrame(base).ffill(axis=1).shift(1, axis=1).to_numpy() + C
    has = ~np.isnan(snap) & ~np.isnan(pred)
    var = (pred - snap)[has]
    stats["snapshot_reconciliation_checks"] = int(has.sum())
    stats["snapshots_with_variance"] = int((np.abs(var) > 0.5).sum())
    stats["abs_variance_units"] = int(np.abs(var).sum())
    return eod, stats


def process_bucket(
    sales_raw: pd.DataFrame,
    txn: pd.DataFrame,
    branches: pd.DataFrame,
    skus: pd.DataFrame,
    dates: pd.DatetimeIndex,
    pc: dict[str, Any],
) -> dict[str, Any]:
    """Clean one bucket of complete SKUs. Returns matrices, series keys and DQ counts."""
    sales, dq = clean_sales(sales_raw)
    del sales_raw
    branch_cats = pd.Index(branches["branch_id"].astype(str))
    sku_cats = pd.Index(skus["sku_id"].astype(str))
    tmp = Index(branch_cats, sku_cats, np.array([], dtype=np.int64), dates[0], len(dates))
    keys = np.unique(tmp.key(sales))
    keys = keys[keys >= 0]
    idx = Index(branch_cats, sku_cats, keys, dates[0], len(dates))
    shape = (idx.n, idx.T)

    raw_qty = to_wide(idx, sales, "qty", agg="last")
    returned = to_wide(idx, sales, "qty_returned", fill=0.0)
    first_col = np.where(np.isnan(raw_qty), shape[1], np.arange(shape[1])[None, :]).min(axis=1)
    del sales
    active = np.arange(shape[1])[None, :] >= first_col[:, None]

    is_missing = np.isnan(raw_qty) & active
    is_outlier = flag_outliers(raw_qty, pc["outlier_q99_multiple"], pc["outlier_min_qty"])
    dq["missing_series_days"] = int(is_missing.sum())
    dq["outliers_flagged"] = int(is_outlier.sum())
    dq["outlier_units_removed"] = int(np.nansum(np.where(is_outlier, raw_qty, 0)))

    y = np.where(is_outlier, np.nan, raw_qty)
    qty_clean = np.where(active, same_weekday_impute(y, pc["impute_weeks"]), np.nan)
    del y

    txn_known = idx.locate(txn)[2]
    dq["txn_rows_without_sales_series"] = int((~txn_known).sum())
    eod, led = build_ledger(idx, txn[txn_known], np.nan_to_num(qty_clean), pc["max_plausible_adjustment"])
    dq.update(led)
    is_stockout = (eod <= 0) & active
    dq["stockout_series_days"] = int(is_stockout.sum())
    dq["negative_ledger_days_clipped"] = int((eod < 0).sum())
    eod = np.clip(eod, 0, None)

    nb = len(sku_cats)
    series = pd.DataFrame(
        {
            "branch_id": branch_cats[keys // nb],
            "sku_id": sku_cats[keys % nb],
            "first_date": dates[np.minimum(first_col, shape[1] - 1)],
        }
    )
    snaps = txn[(txn["txn_type"] == "snapshot") & txn_known].sort_values("date")
    last_snap = snaps.groupby(["branch_id", "sku_id"], observed=True).tail(1)
    return {
        "series": series,
        "dq": dq,
        "last_snap": last_snap,
        "qty_clean": qty_clean,
        "raw_qty": raw_qty,
        "is_missing": is_missing,
        "is_outlier": is_outlier,
        "is_stockout": is_stockout,
        "returned": returned,
        "eod": eod,
        "active": active,
    }


def inventory_status(
    series: pd.DataFrame, b: dict[str, Any], dates: pd.DatetimeIndex, open_po: pd.DataFrame, stagnant_days: int
) -> tuple[pd.DataFrame, int]:
    """Current on-hand, last sale and stagnation per series (A9)."""
    T = len(dates)
    snap = b["last_snap"][["branch_id", "sku_id", "date", "qty"]].rename(
        columns={"date": "snapshot_date", "qty": "on_hand"}
    )
    snap = snap.assign(branch_id=snap["branch_id"].astype(str), sku_id=snap["sku_id"].astype(str))
    status = series[["branch_id", "sku_id"]].merge(snap, how="left", on=["branch_id", "sku_id"])
    stale = (status["snapshot_date"] < dates[-1]) | status["on_hand"].isna()
    status["on_hand"] = np.where(stale, b["eod"][:, -1], status["on_hand"]).astype(float)
    sold = np.nan_to_num(b["qty_clean"]) > 0
    last_sale_col = np.where(sold.any(axis=1), T - 1 - np.argmax(sold[:, ::-1], axis=1), -1)
    status["last_sale_date"] = pd.Series(dates[np.maximum(last_sale_col, 0)]).where(last_sale_col >= 0)
    status["days_since_last_sale"] = np.where(last_sale_col >= 0, T - 1 - last_sale_col, T)
    status["sales_last_90d"] = np.nansum(b["qty_clean"][:, -90:], axis=1)
    status["history_days"] = b["active"].sum(axis=1)
    status["is_stagnant"] = (status["days_since_last_sale"] >= stagnant_days) & (status["on_hand"] > 0)
    po = (
        open_po.groupby(["branch_id", "sku_id"])
        .agg(open_po_qty=("qty", "sum"), next_po_date=("due_date", "min"))
        .reset_index()
    )
    status = status.merge(po, how="left", on=["branch_id", "sku_id"])
    status["open_po_qty"] = status["open_po_qty"].fillna(0)
    return status, int(stale.sum())


def promotions_daily(
    promos: pd.DataFrame, skus: pd.DataFrame, all_dates: pd.DatetimeIndex
) -> tuple[np.ndarray, np.ndarray]:
    """SKU x day discount and days-into-promo matrices (future calendar included)."""
    sku_pos = {s: i for i, s in enumerate(skus["sku_id"].astype(str))}
    discount = np.zeros((len(sku_pos), len(all_dates)), dtype=np.float32)
    days_into = np.full_like(discount, -1)
    for r in promos.itertuples():
        if r.sku_id not in sku_pos:
            continue
        a = (r.start_date - all_dates[0]).days
        b = (r.end_date - all_dates[0]).days
        if b < 0 or a >= len(all_dates):
            continue
        lo, hi = max(a, 0), min(b + 1, len(all_dates))
        discount[sku_pos[r.sku_id], lo:hi] = r.discount_pct
        days_into[sku_pos[r.sku_id], lo:hi] = np.arange(lo - a, hi - a)
    return discount, days_into


def bucket_path(paths: Paths, k: int) -> Path:
    return paths.processed / "buckets" / f"bucket_{k:03d}.npz"


def run(config_path: str | None = None) -> None:
    """Execute the full cleaning pipeline and write processed outputs."""
    t0 = time.time()
    cfg = load_config(config_path)
    pc = cfg["prep"]
    paths = get_paths(cfg)
    masters = load_masters(paths)
    branches, skus = masters["branches"], masters["skus"]
    bmap, dates, vstats = ingest(paths, masters, cfg)
    log.info(
        "ingested and validated %s sales rows into %d bucket(s) (%.0fs)",
        f"{vstats['sales_rows_validated']:,}",
        vstats["buckets"],
        time.time() - t0,
    )

    for sub in ("buckets", "sales_clean"):
        shutil.rmtree(paths.processed / sub, ignore_errors=True)
        (paths.processed / sub).mkdir(parents=True)
    dq: Counter[str] = Counter()
    series_parts, status_parts = [], []
    T = len(dates)
    for k in sorted(bmap["bucket"].unique()):
        b = process_bucket(read_bucket(paths, "sales", k), read_bucket(paths, "txn", k), branches, skus, dates, pc)
        dq.update(b["dq"])
        series = b["series"].assign(bucket=k, row=np.arange(len(b["series"])))
        status, stale = inventory_status(series, b, dates, masters["open_po"], pc["stagnant_days"])
        dq["series_without_final_snapshot"] += stale
        series_parts.append(series)
        status_parts.append(status)
        np.savez(
            bucket_path(paths, k),  # only what modelling needs; full detail lives in sales_clean/
            qty_clean=b["qty_clean"].astype(np.float32),
            is_missing=b["is_missing"],
            is_outlier=b["is_outlier"],
            is_stockout=b["is_stockout"],
            active=b["active"],
            dates=dates.strftime("%Y-%m-%d").to_numpy().astype("U10"),
            branch_id=series["branch_id"].to_numpy().astype("U16"),
            sku_id=series["sku_id"].to_numpy().astype("U16"),
        )
        act = b["active"].ravel()
        b_cat = pd.Categorical(series["branch_id"])
        s_cat = pd.Categorical(series["sku_id"])
        long = pa.table(
            {
                "date": pa.array(np.tile(dates.to_numpy(), len(series))[act]),
                "branch_id": pa.DictionaryArray.from_arrays(
                    pa.array(np.repeat(b_cat.codes.astype(np.int32), T)[act]), pa.array(b_cat.categories.to_numpy())
                ),
                "sku_id": pa.DictionaryArray.from_arrays(
                    pa.array(np.repeat(s_cat.codes.astype(np.int32), T)[act]), pa.array(s_cat.categories.to_numpy())
                ),
                "qty_raw": b["raw_qty"].ravel()[act].astype(np.float32),
                "qty": b["qty_clean"].ravel()[act].astype(np.float32),
                "is_missing": b["is_missing"].ravel()[act],
                "is_outlier": b["is_outlier"].ravel()[act],
                "is_stockout": b["is_stockout"].ravel()[act],
                "on_hand_eod": b["eod"].ravel()[act].astype(np.float32),
            }
        )
        ds.write_dataset(
            long,
            paths.processed / "sales_clean",
            format="parquet",
            partitioning=ds.partitioning(
                pa.schema([("branch_id", pa.dictionary(pa.int32(), pa.string()))]), flavor="hive"
            ),
            basename_template=f"bucket{k:03d}-{{i}}.parquet",
            existing_data_behavior="overwrite_or_ignore",
        )
        log.info("bucket %d: %s series cleaned (%.0fs)", k, f"{len(series):,}", time.time() - t0)
        del b, long

    series = pd.concat(series_parts, ignore_index=True)
    series = series.merge(
        branches[["branch_id", "region", "size_tier", "storage_capacity_units"]], on="branch_id", how="left"
    ).merge(skus, on="sku_id", how="left")
    series.to_parquet(paths.processed / "series.parquet", index=False)
    pd.concat(status_parts, ignore_index=True).to_parquet(paths.processed / "inventory_status.parquet", index=False)

    horizon = cfg["forecast"]["horizon"]
    all_dates = pd.date_range(dates[0], dates[-1] + pd.Timedelta(days=horizon + 7), freq="D")
    discount, days_into = promotions_daily(masters["promotions"], skus, all_dates)
    np.savez_compressed(
        paths.processed / "promotions_daily.npz",
        discount=discount,
        days_into=days_into,
        sku_ids=skus["sku_id"].to_numpy().astype("U16"),
        dates=all_dates.strftime("%Y-%m-%d").to_numpy().astype("U10"),
    )
    shutil.rmtree(paths.processed / "ingest", ignore_errors=True)

    dq.update(vstats)
    report = pd.DataFrame({"check": list(dq), "value": [int(v) for v in dq.values()]})
    report.to_csv(paths.reports / "data_quality_report.csv", index=False)
    log.info("data quality: %s", ", ".join(f"{k}={v:,}" for k, v in dq.items()))
    log.info("%s series x %d days -> %s (%.0fs)", f"{len(series):,}", T, paths.processed.name, time.time() - t0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Clean raw extracts into model-ready data.")
    ap.add_argument("--config", default=None)
    run(ap.parse_args().config)
