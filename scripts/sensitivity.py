"""Sensitivity of the transfer plan to service level, transfer cost and forecast error.

Each scenario changes one assumption, re-computes the inventory policy and re-solves the
optimiser; everything else stays at the base case. ``plan_overlap`` is the share of the
base plan's units that the scenario still moves on the same (SKU, from, to) lanes: a
high overlap means the recommendations are robust to that assumption.

Usage::

    python scripts/sensitivity.py [--config config.yaml]
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import get_paths, load_config  # noqa: E402
from src.optimizer import build_problem, impact, solve_decomposed  # noqa: E402
from src.policy import compute_policy, load_daily_forecasts  # noqa: E402
from src.runlog import get_logger  # noqa: E402

log = get_logger("sensitivity")

SCENARIOS: list[tuple[str, str, dict[str, Any]]] = [
    ("Base case", "as configured", {}),
    ("Service level", "90%", {"service_level": 0.90}),
    ("Service level", "98%", {"service_level": 0.98}),
    ("Transfer cost", "x0.5", {"cost_mult": 0.5}),
    ("Transfer cost", "x2", {"cost_mult": 2.0}),
    ("Forecast error", "sigma x0.75", {"sigma_mult": 0.75}),
    ("Forecast error", "sigma x1.5", {"sigma_mult": 1.5}),
    ("Forecast level", "-10%", {"level_mult": 0.9}),
    ("Forecast level", "+10%", {"level_mult": 1.1}),
]


def run_scenario(
    base: dict[str, Any], cfg: dict[str, Any], change: dict[str, Any]
) -> tuple[dict[str, float], pd.DataFrame]:
    c = copy.deepcopy(cfg)
    if "service_level" in change:
        c["policy"]["service_level"] = change["service_level"]
    stats = base["stats"].copy()
    daily = base["daily"] * change.get("level_mult", 1.0)
    stats["forecast_mean_daily"] *= change.get("level_mult", 1.0)
    stats["sigma_week"] *= change.get("sigma_mult", 1.0)
    costs = base["costs"].copy()
    costs["cost_per_unit"] *= change.get("cost_mult", 1.0)
    pol = compute_policy(stats, daily, base["status"], base["skus"], base["open_po"], base["today"], c)
    prob = build_problem(pol, base["branches"], costs, c)
    plan, _ = solve_decomposed(prob, c)
    return impact(prob.nodes, plan, c["policy"]["stockout_risk_threshold"]), plan


def main(config_path: str | None = None) -> pd.DataFrame:
    cfg = load_config(config_path)
    paths = get_paths(cfg)
    stats = pd.read_parquet(paths.output / "forecast_series_stats.parquet")
    daily, today = load_daily_forecasts(paths, stats)
    base = {
        "stats": stats,
        "daily": daily,
        "today": today,
        "status": pd.read_parquet(paths.processed / "inventory_status.parquet"),
        "skus": pd.read_csv(paths.raw / "skus.csv"),
        "open_po": pd.read_csv(paths.raw / "open_purchase_orders.csv", parse_dates=["due_date"]),
        "branches": pd.read_csv(paths.raw / "branches.csv"),
        "costs": pd.read_csv(paths.raw / "transfer_costs.csv"),
    }
    rows, base_lanes = [], None
    for name, setting, change in SCENARIOS:
        imp, plan = run_scenario(base, cfg, change)
        lanes = plan.groupby(["sku_id", "branch_id_s", "branch_id_r"])["qty"].sum()
        if base_lanes is None:
            base_lanes = lanes
        overlap = (
            pd.concat([base_lanes, lanes], axis=1).fillna(0).min(axis=1).sum() / base_lanes.sum()
            if base_lanes.sum()
            else float("nan")
        )
        rows.append(
            {
                "scenario": name,
                "setting": setting,
                **{
                    k: imp[k]
                    for k in (
                        "transfers",
                        "units_moved",
                        "transfer_cost",
                        "net_savings",
                        "excess_value_freed",
                        "stockout_units_avoided",
                        "series_at_risk_before",
                        "series_at_risk_after",
                    )
                },
                "plan_overlap": overlap,
            }
        )
        log.info(
            "%s %s: %d transfers, net savings $%s, overlap with base %.0f%%",
            name,
            setting,
            imp["transfers"],
            f"{imp['net_savings']:,.0f}",
            100 * overlap,
        )
    out = pd.DataFrame(rows)
    out.to_csv(paths.reports / "sensitivity.csv", index=False)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--config", default=None)
    main(ap.parse_args().config)
