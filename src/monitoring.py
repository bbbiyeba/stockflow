"""Production monitoring: forecast-error drift, data drift and the retrain decision.

Run after actual sales for (part of) a published forecast horizon have arrived::

    python -m src.monitoring --actuals new_sales.csv [--config config.yaml]

``new_sales.csv`` has the same columns as the sales extract (date, branch_id, sku_id, qty).
The report compares live accuracy with the held-out backtest baseline and checks whether
recent demand still looks like the training data. Exit code 1 means "retrain recommended".
See docs/MONITORING.md for thresholds and the operating procedure.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import get_paths, load_config  # noqa: E402
from src.runlog import get_logger  # noqa: E402

log = get_logger("monitoring")


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population stability index between two samples (quantile bins of ``expected``).

    < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 significant shift.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected, actual = expected[np.isfinite(expected)], actual[np.isfinite(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return float("nan")
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))


def accuracy_vs_baseline(
    forecasts: pd.DataFrame, actuals: pd.DataFrame, baseline: pd.DataFrame, stats: pd.DataFrame, cfg: dict[str, Any]
) -> pd.DataFrame:
    """Live WAPE / bias by ABC class vs the held-out backtest baseline."""
    mc = cfg.get("monitoring", {})
    j = forecasts.merge(actuals, on=["date", "branch_id", "sku_id"], how="inner")
    j = j.merge(stats[["branch_id", "sku_id", "abc_class"]], on=["branch_id", "sku_id"], how="left")
    j = j.assign(err=j["forecast"] - j["qty"], inside=(j["qty"] >= j["p10"]) & (j["qty"] <= j["p90"]))
    j["abs_err"] = j["err"].abs()
    live = (
        j.groupby("abc_class")
        .agg(
            days_scored=("date", "nunique"),
            actual_units=("qty", "sum"),
            abs_err=("abs_err", "sum"),
            err=("err", "sum"),
            p10_p90_coverage=("inside", "mean"),
        )
        .reset_index()
    )
    denom = live["actual_units"].clip(lower=1e-9)
    live["wape"] = live.pop("abs_err") / denom
    live["bias"] = live.pop("err") / denom
    base = baseline.set_index("abc_class")
    live["baseline_wape"] = live["abc_class"].map(base["wape"])
    live["wape_ratio"] = live["wape"] / live["baseline_wape"]
    live["alert"] = np.select(
        [
            live["wape_ratio"] > 1 + mc.get("wape_degradation", 0.20),
            live["bias"].abs() > mc.get("bias_limit", 0.10),
            (live["p10_p90_coverage"] - 0.8).abs() > mc.get("coverage_tolerance", 0.10),
        ],
        ["accuracy degraded", "biased", "intervals miscalibrated"],
        default="ok",
    )
    return live


def demand_drift(history: pd.DataFrame, actuals: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """PSI of daily demand per ABC class: recent actuals vs the last 365 days of training history."""
    rows = []
    h = history.merge(stats[["branch_id", "sku_id", "abc_class"]], on=["branch_id", "sku_id"])
    a = actuals.merge(stats[["branch_id", "sku_id", "abc_class"]], on=["branch_id", "sku_id"])
    for cls, hg in h.groupby("abc_class"):
        ag = a[a["abc_class"] == cls]
        rows.append(
            {
                "abc_class": cls,
                "psi_daily_qty": psi(np.log1p(hg["qty"]), np.log1p(ag["qty"])),
                "share_zero_train": float((hg["qty"] == 0).mean()),
                "share_zero_live": float((ag["qty"] == 0).mean()) if len(ag) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--actuals", required=True, help="CSV of actual sales for the forecast period")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    mc = cfg.get("monitoring", {})
    paths = get_paths(cfg)
    actuals = pd.read_csv(args.actuals, parse_dates=["date"], usecols=["date", "branch_id", "sku_id", "qty"])
    actuals["qty"] = actuals["qty"].clip(lower=0)
    forecasts = pd.read_parquet(paths.output / "forecasts")
    forecasts["branch_id"] = forecasts["branch_id"].astype(str)
    stats = pd.read_parquet(paths.output / "forecast_series_stats.parquet")
    baseline = pd.read_csv(paths.reports / "holdout_selected_by_abc.csv")
    start = forecasts["date"].min()
    history = pd.read_parquet(
        paths.processed / "sales_clean",
        columns=["date", "branch_id", "sku_id", "qty"],
        filters=[("date", ">=", start - pd.Timedelta(days=365))],
    )
    history["branch_id"] = history["branch_id"].astype(str)
    history["sku_id"] = history["sku_id"].astype(str)

    acc = accuracy_vs_baseline(forecasts, actuals, baseline, stats, cfg)
    drift = demand_drift(history, actuals, stats)
    acc.to_csv(paths.reports / "monitoring_accuracy.csv", index=False)
    drift.to_csv(paths.reports / "monitoring_drift.csv", index=False)
    retrain = bool((acc["alert"] != "ok").any() or (drift["psi_daily_qty"] > mc.get("psi_limit", 0.25)).any())
    log.info(
        "accuracy: %s",
        "; ".join(
            f"{r.abc_class}: WAPE {r.wape:.2f} (baseline {r.baseline_wape:.2f}) {r.alert}" for r in acc.itertuples()
        ),
    )
    log.info("drift: %s", "; ".join(f"{r.abc_class}: PSI {r.psi_daily_qty:.3f}" for r in drift.itertuples()))
    log.info("retrain recommended: %s", "YES" if retrain else "no")
    return 1 if retrain else 0


if __name__ == "__main__":
    sys.exit(main())
