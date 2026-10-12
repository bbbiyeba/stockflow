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
    imputed with the mean of the same weekday over the surrounding ``impute_weeks`` weeks.
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

Outputs (``paths.processed_dir``): ``series.parquet``, ``sales_clean.parquet``,
``wide.npz`` (dense series x day matrices), ``inventory_status.parquet`` and
``promotions_daily.npz``. A data-quality report is written to ``reports/``.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import Paths, get_paths, load_config  # noqa: E402


# --------------------------------------------------------------------------- loading
def load_raw(paths: Paths) -> dict[str, pd.DataFrame]:
    """Read all raw CSVs from ``paths.raw``."""
    raw = paths.raw
    return {
        "branches": pd.read_csv(raw / "branches.csv"),
        "skus": pd.read_csv(raw / "skus.csv"),
        "promotions": pd.read_csv(raw / "promotions.csv", parse_dates=["start_date", "end_date"]),
        "sales": pd.read_csv(raw / "sales_daily.csv", parse_dates=["date"],
                             dtype={"branch_id": "string", "sku_id": "string"}),
        "txn": pd.read_csv(raw / "inventory_txn.csv", parse_dates=["date"]),
        "transfer_costs": pd.read_csv(raw / "transfer_costs.csv"),
        "open_po": pd.read_csv(raw / "open_purchase_orders.csv", parse_dates=["due_date"]),
    }


# --------------------------------------------------------------------------- helpers
def same_weekday_impute(Y: np.ndarray, weeks: int) -> np.ndarray:
    """Fill NaNs with the mean of the same weekday +/- 1..``weeks`` weeks (A4).

    Falls back to the series' overall weekday mean, then to 0.
    """
    n, T = Y.shape
    stack = []
    for k in range(1, weeks + 1):
        for sign in (-1, 1):
            shifted = np.full_like(Y, np.nan)
            s = 7 * k * sign
            if s > 0:
                shifted[:, s:] = Y[:, :-s]
            else:
                shifted[:, :s] = Y[:, -s:]
            stack.append(shifted)
    with np.errstate(invalid="ignore"), _quiet_nanmean():
        local = np.nanmean(np.stack(stack), axis=0)
    out = np.where(np.isnan(Y), local, Y)
    # fallback: series x weekday mean
    if np.isnan(out).any():
        dow = np.arange(T) % 7
        for d in range(7):
            cols = dow == d
            with _quiet_nanmean():
                m = np.nanmean(Y[:, cols], axis=1)
            block = out[:, cols]
            fill = np.broadcast_to(np.nan_to_num(m)[:, None], block.shape)
            out[:, cols] = np.where(np.isnan(block), fill, block)
    return out


class _quiet_nanmean:
    """Context manager silencing 'mean of empty slice' warnings."""

    def __enter__(self) -> None:
        import warnings
        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter("ignore", category=RuntimeWarning)

    def __exit__(self, *exc: Any) -> None:
        self._w.__exit__(*exc)


def flag_outliers(Y: np.ndarray, q99_multiple: float, min_qty: float) -> np.ndarray:
    """Boolean mask of keying-error outliers (A5)."""
    pos = np.where(Y > 0, Y, np.nan)
    with _quiet_nanmean():
        q99 = np.nanquantile(pos, 0.99, axis=1)
    q99 = np.nan_to_num(q99, nan=0.0)
    thresh = np.maximum(q99 * q99_multiple, min_qty)
    return np.nan_to_num(Y, nan=0.0) > thresh[:, None]


def to_wide(df: pd.DataFrame, value: str, series_pos: pd.Series, date_pos: pd.Series,
            shape: tuple[int, int], fill: float = np.nan, agg: str = "sum") -> np.ndarray:
    """Pivot a long (branch_id, sku_id, date, value) frame into a dense matrix."""
    out = np.full(shape, fill, dtype=np.float64)
    if df.empty:
        return out
    key = pd.MultiIndex.from_frame(df[["branch_id", "sku_id"]])
    rows = series_pos.reindex(key).to_numpy()
    cols = date_pos.reindex(df["date"]).to_numpy()
    ok = ~(pd.isna(rows) | pd.isna(cols))
    rows, cols, vals = rows[ok].astype(int), cols[ok].astype(int), df[value].to_numpy()[ok]
    if agg == "sum":
        if np.isnan(fill):
            out[rows, cols] = 0.0
        np.add.at(out, (rows, cols), vals)
    else:  # last value wins
        out[rows, cols] = vals
    return out


# --------------------------------------------------------------------------- main steps
def clean_sales(sales: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, int]]:
    """Dedupe and split returns (A2, A3). Returns the tidy frame and counts for the DQ report."""
    stats: dict[str, int] = {"raw_rows": len(sales)}
    dup = sales.duplicated()
    stats["exact_duplicates_dropped"] = int(dup.sum())
    s = sales.loc[~dup]
    stats["duplicate_keys_summed"] = int(s.duplicated(["date", "branch_id", "sku_id"]).sum())
    s = s.groupby(["date", "branch_id", "sku_id"], as_index=False, observed=True)["qty"].sum()
    stats["negative_qty_rows"] = int((s["qty"] < 0).sum())
    s["qty_returned"] = (-s["qty"]).clip(lower=0)
    s["qty"] = s["qty"].clip(lower=0)
    return s, stats


def build_ledger(txn: pd.DataFrame, sales_used: np.ndarray, series_pos: pd.Series,
                 date_pos: pd.Series, max_plausible_adjustment: float = 10.0,
                 ) -> tuple[np.ndarray, pd.DataFrame, dict[str, int]]:
    """Rebuild daily end-of-day on-hand from snapshots + movements (A6, A7)."""
    shape = sales_used.shape
    stats: dict[str, int] = {}
    snaps = txn[txn["txn_type"] == "snapshot"]
    moves = txn[txn["txn_type"] != "snapshot"].copy()

    # A7: impossible negative adjustments
    adj = moves[moves["txn_type"] == "adjustment"].sort_values("date")
    prior = pd.merge_asof(adj, snaps.sort_values("date")[["date", "branch_id", "sku_id", "qty"]]
                          .rename(columns={"qty": "prior_snapshot"}),
                          on="date", by=["branch_id", "sku_id"], allow_exact_matches=False)
    bad = ((prior["qty"] < 0) & (-prior["qty"] > prior["prior_snapshot"].fillna(np.inf))
           & (-prior["qty"] > max_plausible_adjustment))
    bad_ids = set(prior.loc[bad, "txn_id"])
    stats["bad_adjustments_excluded"] = len(bad_ids)
    stats["bad_adjustment_units"] = int(-prior.loc[bad, "qty"].sum())
    moves = moves[~moves["txn_id"].isin(bad_ids)]

    net = to_wide(moves, "qty", series_pos, date_pos, shape, fill=0.0) - sales_used
    snap = to_wide(snaps, "qty", series_pos, date_pos, shape, agg="last")
    stats["snapshots"] = int((~np.isnan(snap)).sum())

    C = np.cumsum(net, axis=1)
    anchor = pd.DataFrame(snap - C).ffill(axis=1).bfill(axis=1).to_numpy()  # A6
    eod = anchor + C

    # reconciliation: ledger prediction at each snapshot vs. the count itself
    pred_anchor = pd.DataFrame(snap - C).ffill(axis=1).shift(1, axis=1).to_numpy()
    pred = pred_anchor + C
    has = ~np.isnan(snap) & ~np.isnan(pred)
    var = (pred - snap)[has]
    stats["snapshot_reconciliation_checks"] = int(has.sum())
    stats["snapshots_with_variance"] = int((np.abs(var) > 0.5).sum())
    stats["abs_variance_units"] = int(np.abs(var).sum())
    return eod, moves, stats


def run(config_path: str | None = None) -> None:
    """Execute the full cleaning pipeline and write processed outputs."""
    t0 = time.time()
    cfg = load_config(config_path)
    pc = cfg["prep"]
    paths = get_paths(cfg)
    raw = load_raw(paths)

    sales, dq = clean_sales(raw["sales"], cfg)
    dates = pd.date_range(sales["date"].min(), sales["date"].max(), freq="D")
    series = (sales.groupby(["branch_id", "sku_id"], observed=True)["date"].min()
              .rename("first_date").reset_index().sort_values(["branch_id", "sku_id"])
              .reset_index(drop=True))
    series_pos = pd.Series(np.arange(len(series)),
                           index=pd.MultiIndex.from_frame(series[["branch_id", "sku_id"]]))
    date_pos = pd.Series(np.arange(len(dates)), index=dates)
    shape = (len(series), len(dates))

    raw_qty = to_wide(sales, "qty", series_pos, date_pos, shape, agg="last")
    returned = to_wide(sales, "qty_returned", series_pos, date_pos, shape, fill=0.0)
    first_col = date_pos.reindex(series["first_date"]).to_numpy()
    active = np.arange(shape[1])[None, :] >= first_col[:, None]

    is_missing = np.isnan(raw_qty) & active
    is_outlier = flag_outliers(raw_qty, pc["outlier_q99_multiple"], pc["outlier_min_qty"])
    dq["missing_series_days"] = int(is_missing.sum())
    dq["missing_branch_days_full_outage"] = int(
        (pd.DataFrame(is_missing).groupby(series["branch_id"].to_numpy()).mean() == 1.0).to_numpy().sum())
    dq["outliers_flagged"] = int(is_outlier.sum())
    dq["outlier_units_removed"] = int(np.nansum(np.where(is_outlier, raw_qty, 0)))

    y = np.where(is_outlier, np.nan, raw_qty)
    y_imp = same_weekday_impute(y, pc["impute_weeks"])
    qty_clean = np.where(active, y_imp, np.nan)

    eod, moves, led_stats = build_ledger(raw["txn"], np.nan_to_num(qty_clean), series_pos, date_pos,
                                    pc["max_plausible_adjustment"])
    dq.update(led_stats)
    is_stockout = (eod <= 0) & active
    dq["stockout_series_days"] = int(is_stockout.sum())
    dq["negative_ledger_days_clipped"] = int((eod < 0).sum())
    eod = np.clip(eod, 0, None)

    # ---- series attributes -------------------------------------------------
    branches, skus = raw["branches"], raw["skus"]
    series = series.merge(branches[["branch_id", "region", "size_tier", "storage_capacity_units"]],
                          on="branch_id", how="left")
    series = series.merge(skus, on="sku_id", how="left")

    # ---- current inventory status (A9) -------------------------------------
    T = shape[1]
    last_snap = (raw["txn"][raw["txn"]["txn_type"] == "snapshot"]
                 .sort_values("date").groupby(["branch_id", "sku_id"]).tail(1))
    status = series[["branch_id", "sku_id"]].merge(
        last_snap[["branch_id", "sku_id", "date", "qty"]].rename(
            columns={"date": "snapshot_date", "qty": "on_hand"}), how="left")
    stale = status["snapshot_date"] < dates[-1]
    dq["series_without_final_snapshot"] = int(stale.sum() + status["on_hand"].isna().sum())
    status["on_hand"] = np.where(stale | status["on_hand"].isna(), eod[:, -1], status["on_hand"])
    sold = np.nan_to_num(qty_clean) > 0
    last_sale_col = np.where(sold.any(axis=1), T - 1 - np.argmax(sold[:, ::-1], axis=1), -1)
    status["last_sale_date"] = [dates[c] if c >= 0 else pd.NaT for c in last_sale_col]
    status["days_since_last_sale"] = np.where(last_sale_col >= 0, T - 1 - last_sale_col, T)
    status["sales_last_90d"] = np.nansum(qty_clean[:, -90:], axis=1)
    status["is_stagnant"] = (status["days_since_last_sale"] >= pc["stagnant_days"]) & (status["on_hand"] > 0)
    po = raw["open_po"]
    status = status.merge(po.groupby(["branch_id", "sku_id"]).agg(
        open_po_qty=("qty", "sum"), next_po_date=("due_date", "min")).reset_index(), how="left")
    status["open_po_qty"] = status["open_po_qty"].fillna(0)

    # ---- daily promo matrix per SKU (future calendar included) --------------
    horizon = cfg["forecast"]["horizon"]
    all_dates = pd.date_range(dates[0], dates[-1] + pd.Timedelta(days=horizon + 7), freq="D")
    sku_ids = skus["sku_id"].to_numpy()
    sku_pos = {s: i for i, s in enumerate(sku_ids)}
    discount = np.zeros((len(sku_ids), len(all_dates)), dtype=np.float32)
    days_into = np.full_like(discount, -1)
    for r in raw["promotions"].itertuples():
        a = (r.start_date - all_dates[0]).days
        b = (r.end_date - all_dates[0]).days
        if b < 0 or a >= len(all_dates):
            continue
        lo, hi = max(a, 0), min(b + 1, len(all_dates))
        discount[sku_pos[r.sku_id], lo:hi] = r.discount_pct
        days_into[sku_pos[r.sku_id], lo:hi] = np.arange(lo - a, hi - a)

    # ---- write ---------------------------------------------------------------
    out = paths.processed
    series.to_parquet(out / "series.parquet", index=False)
    status.to_parquet(out / "inventory_status.parquet", index=False)
    np.savez_compressed(
        out / "wide.npz",
        qty_clean=qty_clean.astype(np.float32), qty_raw=raw_qty.astype(np.float32),
        is_missing=is_missing, is_outlier=is_outlier, is_stockout=is_stockout,
        returned=returned.astype(np.float32), on_hand_eod=eod.astype(np.float32), active=active,
        dates=dates.strftime("%Y-%m-%d").to_numpy().astype("U10"))
    np.savez_compressed(out / "promotions_daily.npz", discount=discount, days_into=days_into,
                        sku_ids=sku_ids.astype("U16"), dates=all_dates.strftime("%Y-%m-%d").to_numpy().astype("U10"))

    long = pd.DataFrame({
        "date": np.tile(dates.to_numpy(), shape[0]),
        "branch_id": np.repeat(series["branch_id"].to_numpy(), T),
        "sku_id": np.repeat(series["sku_id"].to_numpy(), T),
        "qty_raw": raw_qty.ravel().astype(np.float32),
        "qty": qty_clean.ravel().astype(np.float32),
        "is_missing": is_missing.ravel(), "is_outlier": is_outlier.ravel(),
        "is_stockout": is_stockout.ravel(), "on_hand_eod": eod.ravel().astype(np.float32),
    })
    long[active.ravel()].to_parquet(out / "sales_clean.parquet", index=False)

    report = pd.DataFrame({"check": list(dq), "value": list(dq.values())})
    report.to_csv(paths.reports / "data_quality_report.csv", index=False)
    print(report.to_string(index=False))
    print(f"[prep] {shape[0]:,} series x {T} days -> {out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Clean raw extracts into model-ready data.")
    ap.add_argument("--config", default=None)
    run(ap.parse_args().config)
