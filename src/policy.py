"""Turn forecasts into inventory policy and current health metrics per branch x SKU.

Definitions (all quantities in units unless stated):
- ``fc_lead``      forecast demand over the replenishment lead time L
- ``sigma_lead``   forecast-error std over L, scaled from the weekly backtest error:
                   sigma_week * sqrt(L / 7) (weekly blocks capture error autocorrelation
                   better than sqrt(L) x daily error)
- ``safety_stock`` z(service_level) * sigma_lead
- ``reorder_point`` fc_lead + safety_stock
- ``target_max``   order-up-to level: forecast over L + R (review period) + safety_stock
- ``inv_position`` on hand + open purchase orders due within L
- ``excess_units`` inventory position above target_max
- ``shortfall_units`` units needed to bring the inventory position back to the reorder point
- ``p_stockout``   P(lead-time demand > inventory position) under a normal error model
- ``expected_lost_units`` E[(D_L - IP)^+] via the standard normal loss function
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import get_paths, load_config  # noqa: E402

DOS_CAP = 999.0


def normal_loss(k: np.ndarray) -> np.ndarray:
    """Standard normal loss function G(k) = phi(k) - k * (1 - Phi(k))."""
    return norm.pdf(k) - k * norm.sf(k)


def expected_lost_units(position: np.ndarray, mean: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Expected units short over the lead time, E[(D - position)^+], D ~ N(mean, sigma)."""
    position, mean, sigma = map(lambda a: np.asarray(a, dtype=float), (position, mean, sigma))
    with np.errstate(divide="ignore", invalid="ignore"):
        k = (position - mean) / sigma
        loss = sigma * normal_loss(k)
    return np.where(sigma > 0, loss, np.maximum(mean - position, 0.0))


def stockout_probability(position: np.ndarray, mean: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """P(D > position), D ~ N(mean, sigma); degenerate case when sigma == 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        p = norm.sf((position - mean) / sigma)
    return np.where(sigma > 0, p, (mean > position).astype(float))


def horizon_sum(daily: np.ndarray, days: np.ndarray) -> np.ndarray:
    """Sum the first ``days[i]`` columns of each row; beyond the horizon extrapolate the row mean."""
    n, H = daily.shape
    cum = np.concatenate([np.zeros((n, 1)), np.cumsum(daily, axis=1)], axis=1)
    within = cum[np.arange(n), np.minimum(days, H)]
    extra = np.maximum(days - H, 0) * daily.mean(axis=1)
    return within + extra


def compute_policy(stats: pd.DataFrame, daily: np.ndarray, status: pd.DataFrame, skus: pd.DataFrame,
                   open_po: pd.DataFrame, today: pd.Timestamp, cfg: dict[str, Any]) -> pd.DataFrame:
    """Compute policy parameters and health metrics.

    Args:
        stats: per-series forecast statistics (row order defines output order).
        daily: (n_series, horizon) daily point forecasts aligned with ``stats``.
        status: current on-hand / stagnation per series.
        skus: SKU master.
        open_po: open purchase orders (branch_id, sku_id, due_date, qty).
        today: first day of the forecast horizon.
    """
    pc = cfg["policy"]
    df = stats.merge(status, on=["branch_id", "sku_id"], how="left", validate="1:1")
    df = df.merge(skus[["sku_id", "category", "unit_cost", "holding_cost_rate", "lead_time_days",
                        "case_pack", "obsolescence_risk"]], on="sku_id", how="left")
    L = df["lead_time_days"].to_numpy().astype(int)
    R = int(pc["review_period_days"])
    z = norm.ppf(pc["service_level"])

    df["fc_lead"] = horizon_sum(daily, L)
    df["fc_lead_review"] = horizon_sum(daily, L + R)
    df["sigma_lead"] = df["sigma_week"] * np.sqrt(L / 7.0)
    df["safety_stock"] = z * df["sigma_lead"]
    df["reorder_point"] = df["fc_lead"] + df["safety_stock"]
    df["target_max"] = df["fc_lead_review"] + df["safety_stock"]

    po = open_po.copy()
    po = po.merge(df[["branch_id", "sku_id", "lead_time_days"]], on=["branch_id", "sku_id"], how="inner")
    po = po[po["due_date"] < today + pd.to_timedelta(po["lead_time_days"], unit="D")]
    inbound = po.groupby(["branch_id", "sku_id"])["qty"].sum().rename("inbound_within_lead")
    df = df.merge(inbound.reset_index(), on=["branch_id", "sku_id"], how="left")
    df["inbound_within_lead"] = df["inbound_within_lead"].fillna(0.0)
    df["inv_position"] = df["on_hand"] + df["inbound_within_lead"]

    mean_daily = df["forecast_mean_daily"].to_numpy()
    with np.errstate(divide="ignore"):
        dos = np.where(mean_daily > 1e-3, df["on_hand"] / np.maximum(mean_daily, 1e-9), DOS_CAP)
    df["days_of_supply"] = np.minimum(dos, DOS_CAP)
    df.loc[df["on_hand"] <= 0, "days_of_supply"] = 0.0
    df["excess_units"] = np.maximum(df["inv_position"] - df["target_max"], 0.0)
    df["below_safety_stock_units"] = np.maximum(df["safety_stock"] - df["on_hand"], 0.0)
    df["shortfall_units"] = np.maximum(df["reorder_point"] - df["inv_position"], 0.0)
    df["p_stockout"] = stockout_probability(df["inv_position"], df["fc_lead"], df["sigma_lead"])
    df["expected_lost_units"] = expected_lost_units(df["inv_position"], df["fc_lead"], df["sigma_lead"])
    df["stockout_risk"] = df["p_stockout"] > pc["stockout_risk_threshold"]
    order = np.where(df["inv_position"] <= df["reorder_point"],
                     np.maximum(df["target_max"] - df["inv_position"], 0.0), 0.0)
    df["recommended_order_qty"] = np.ceil(order / df["case_pack"]) * df["case_pack"]

    df["on_hand_value"] = df["on_hand"] * df["unit_cost"]
    df["excess_value"] = df["excess_units"] * df["unit_cost"]
    df["stagnant_value"] = np.where(df["is_stagnant"], df["on_hand_value"], 0.0)
    df["health"] = np.select(
        [df["is_stagnant"], df["stockout_risk"], df["excess_units"] > 0],
        ["stagnant", "stockout_risk", "excess"], default="healthy")
    return df


def run(config_path: str | None = None) -> pd.DataFrame:
    """Compute and write ``inventory_policy.csv``."""
    t0 = time.time()
    cfg = load_config(config_path)
    paths = get_paths(cfg)
    stats = pd.read_parquet(paths.output / "forecast_series_stats.parquet")
    fc = pd.read_csv(paths.output / "forecasts.csv", parse_dates=["date"])
    H = fc["date"].nunique()
    daily = fc["forecast"].to_numpy().reshape(len(stats), H)  # forecasts.csv is series-major
    assert (fc["branch_id"].to_numpy()[::H] == stats["branch_id"].to_numpy()).all()
    status = pd.read_parquet(paths.processed / "inventory_status.parquet")
    skus = pd.read_csv(paths.raw / "skus.csv")
    open_po = pd.read_csv(paths.raw / "open_purchase_orders.csv", parse_dates=["due_date"])
    today = fc["date"].min()

    df = compute_policy(stats, daily, status, skus, open_po, today, cfg)
    df.to_csv(paths.output / "inventory_policy.csv", index=False)
    summ = df.groupby("health").agg(series=("sku_id", "size"), on_hand_value=("on_hand_value", "sum"))
    print(summ.round(0).to_string())
    print(f"[policy] stockout-risk series: {int(df['stockout_risk'].sum()):,}  "
          f"excess ${df['excess_value'].sum():,.0f}  stagnant ${df['stagnant_value'].sum():,.0f}  "
          f"({time.time() - t0:.0f}s)")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Compute inventory policy from forecasts.")
    ap.add_argument("--config", default=None)
    run(ap.parse_args().config)
