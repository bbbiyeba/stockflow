"""Rolling-origin backtest, per-segment model selection and final forecast with quantiles.

Pipeline:
1. For each of ``backtest_folds`` origins (28 days apart, ending at the last history day),
   fit every model on data strictly before the origin and forecast the next ``horizon`` days.
2. Score each model on uncensored days (not missing, not an outlier, not a stock-out day);
   report WAPE / MAE / bias / MASE by ABC class, demand pattern and branch.
3. Choose the best model per segment (ABC class x Syntetos-Boylan demand pattern).
4. Build residual-based prediction intervals: per-series error scale x per-segment
   quantiles of standardised residuals; verify coverage on the last fold out-of-sample.
5. Refit on all history and write ``forecasts.csv`` plus per-series error statistics
   used by the inventory policy.
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import Paths, get_paths, load_config  # noqa: E402
from src.features import (STATIC_CATEGORICAL, Cumsums, calendar_features,  # noqa: E402
                          feature_frame, feature_names)
from src.metrics import aggregate_metrics, demand_pattern, mase_scale, series_error_sums  # noqa: E402
from src.models import (forecast_moving_average, forecast_seasonal_naive,  # noqa: E402
                        forecast_statistical)

warnings.filterwarnings("ignore", category=RuntimeWarning)

MODELS: list[str] = ["seasonal_naive", "moving_average", "croston_sba", "tsb", "ses", "lightgbm"]


class ForecastData:
    """Model-ready matrices loaded from ``data/processed``."""

    def __init__(self, paths: Paths, horizon: int) -> None:
        w = np.load(paths.processed / "wide.npz")
        self.dates = pd.DatetimeIndex(w["dates"].astype(str))
        self.Y = w["qty_clean"].astype(np.float64)
        self.valid = w["active"] & ~w["is_missing"] & ~w["is_outlier"] & ~w["is_stockout"]
        self.series = pd.read_parquet(paths.processed / "series.parquet")
        for c in STATIC_CATEGORICAL:
            self.series[c] = self.series[c].astype("category")
        promo = np.load(paths.processed / "promotions_daily.npz")
        self.all_dates = pd.DatetimeIndex(promo["dates"].astype(str))
        sku_pos = {s: i for i, s in enumerate(promo["sku_ids"].astype(str))}
        rows = self.series["sku_id"].astype(str).map(sku_pos).to_numpy()
        self.discount = promo["discount"][rows]
        self.days_into = promo["days_into"][rows]
        self.cal = calendar_features(self.all_dates)
        self.T = len(self.dates)
        assert len(self.all_dates) >= self.T + horizon, "promotion calendar must cover the horizon"

    @property
    def n(self) -> int:
        return self.Y.shape[0]


def fit_lightgbm(data: ForecastData, origin: int, cfg: dict[str, Any]) -> lgb.Booster:
    """Train the global model on days [origin - train_days, origin)."""
    fc = cfg["forecast"]
    lo = max(origin - fc["lgbm_train_days"], 0)
    t_cols = np.arange(lo, origin)
    Y = data.Y[:, :origin]
    X = feature_frame(Y, t_cols, data.series, data.cal, data.discount, data.days_into)
    y = data.Y[:, t_cols].ravel()
    keep = data.valid[:, t_cols].ravel()
    X, y = X.loc[keep, feature_names()], y[keep]
    params = dict(fc["lgbm_params"], seed=cfg["data"]["seed"], num_threads=0,
                  deterministic=True, force_row_wise=True)
    ds = lgb.Dataset(X, y, categorical_feature=STATIC_CATEGORICAL, free_raw_data=True)
    return lgb.train(params, ds, num_boost_round=fc["lgbm_num_boost_round"])


def predict_lightgbm(model: lgb.Booster, data: ForecastData, origin: int, horizon: int) -> np.ndarray:
    """Recursive multi-step forecast: each day's prediction feeds the next day's short lags."""
    Yw = np.full((data.n, origin + horizon), np.nan)
    Yw[:, :origin] = data.Y[:, :origin]
    for h in range(horizon):
        t = origin + h
        X = feature_frame(Yw, np.array([t]), data.series, data.cal, data.discount,
                          data.days_into, Cumsums.of(Yw[:, :t]))
        Yw[:, t] = np.clip(model.predict(X[feature_names()]), 0, None)
    return Yw[:, origin:]


def forecast_all(data: ForecastData, origin: int, cfg: dict[str, Any],
                 models: list[str] | None = None) -> tuple[dict[str, np.ndarray], lgb.Booster | None]:
    """Run every model from ``origin`` (first forecast day index)."""
    fc = cfg["forecast"]
    H = fc["horizon"]
    hist = data.Y[:, :origin]
    last_dow = int(data.all_dates[origin - 1].dayofweek)
    out: dict[str, np.ndarray] = {}
    booster = None
    for m in models or MODELS:
        t0 = time.time()
        if m == "seasonal_naive":
            out[m] = forecast_seasonal_naive(hist, H, fc["seasonal_period"])
        elif m == "moving_average":
            out[m] = forecast_moving_average(hist, H, fc["moving_average_window"])
        elif m in ("croston_sba", "tsb", "ses"):
            out[m] = forecast_statistical(hist, H, last_dow, m)
        elif m == "lightgbm":
            booster = fit_lightgbm(data, origin, cfg)
            out[m] = predict_lightgbm(booster, data, origin, H)
        print(f"    {m:<15} {time.time() - t0:5.1f}s")
    return out, booster


def interval_params(resid: np.ndarray, segments: np.ndarray, quantiles: list[float]
                    ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Per-series daily / weekly error scale and per-segment standardised residual quantiles.

    Args:
        resid: (n_series, n_folds, horizon) residuals actual - forecast (NaN = not scored).
        segments: (n_series,) segment labels.
    Returns:
        sigma_daily, sigma_week (per series) and {segment: quantile values}.
    """
    n, k, H = resid.shape
    sigma_d = np.sqrt(np.nanmean(resid.reshape(n, -1) ** 2, axis=1))
    weeks = H // 7
    blocks = resid[:, :, : weeks * 7].reshape(n, k, weeks, 7)
    cnt = (~np.isnan(blocks)).sum(axis=3)
    bsum = np.nansum(blocks, axis=3) * 7 / np.maximum(cnt, 1)
    bsum = np.where(cnt >= 4, bsum, np.nan)
    sigma_w = np.sqrt(np.nanmean(bsum.reshape(n, -1) ** 2, axis=1))
    sigma_d = np.nan_to_num(sigma_d, nan=0.0)
    sigma_w = np.where(np.isnan(sigma_w), sigma_d * np.sqrt(7), sigma_w)
    z = resid / np.maximum(sigma_d, 1e-6)[:, None, None]
    qs: dict[str, np.ndarray] = {}
    for seg in np.unique(segments):
        vals = z[segments == seg].ravel()
        vals = vals[np.isfinite(vals)]
        qs[seg] = np.quantile(vals, quantiles) if len(vals) else np.zeros(len(quantiles))
    return sigma_d, sigma_w, qs


def apply_intervals(point: np.ndarray, sigma_d: np.ndarray, segments: np.ndarray,
                    qs: dict[str, np.ndarray]) -> np.ndarray:
    """Return (n_series, horizon, n_quantiles) non-negative, monotone quantile forecasts."""
    q = np.stack([qs.get(s, np.zeros(len(next(iter(qs.values()))))) for s in segments])
    out = point[:, :, None] + q[:, None, :] * sigma_d[:, None, None]
    return np.maximum.accumulate(np.clip(out, 0, None), axis=2)


def run(config_path: str | None = None) -> None:
    """Backtest, select models, and write the final forecast."""
    t_start = time.time()
    cfg = load_config(config_path)
    fc = cfg["forecast"]
    H, K = fc["horizon"], fc["backtest_folds"]
    paths = get_paths(cfg)
    data = ForecastData(paths, H)
    s = data.series
    print(f"[forecast] {data.n:,} series, history {data.dates[0].date()} .. {data.dates[-1].date()}")

    # ---------------------------------------------------------------- backtest
    origins = [data.T - H * k for k in range(K, 0, -1)]
    rows, fold_fc, fold_actual, fold_seg = [], [], [], []
    for f, origin in enumerate(origins):
        print(f"[forecast] fold {f + 1}/{K}: origin {data.all_dates[origin].date()}")
        preds, _ = forecast_all(data, origin, cfg)
        actual = np.where(data.valid[:, origin:origin + H], data.Y[:, origin:origin + H], np.nan)
        pattern = demand_pattern(data.Y[:, origin - 365:origin])
        scale = mase_scale(data.Y[:, origin - 365:origin], fc["seasonal_period"])
        for m, p in preds.items():
            d = series_error_sums(actual, p)
            d["fold"], d["model"], d["series_idx"] = f + 1, m, np.arange(data.n)
            d["scale"], d["pattern"] = scale, pattern
            rows.append(d)
        fold_fc.append(preds)
        fold_actual.append(actual)
        fold_seg.append(pattern)
    bt = pd.concat(rows, ignore_index=True)
    bt = bt.merge(s[["branch_id", "sku_id", "abc_class", "category"]].assign(series_idx=np.arange(data.n)),
                  on="series_idx")

    rep = paths.reports
    overall = aggregate_metrics(bt, ["model"]).sort_values("wape")
    by_abc = aggregate_metrics(bt, ["abc_class", "model"])
    by_seg = aggregate_metrics(bt, ["abc_class", "pattern", "model"])
    by_branch = aggregate_metrics(bt, ["branch_id", "model"])
    by_cat = aggregate_metrics(bt, ["category", "model"])
    by_fold = aggregate_metrics(bt, ["fold", "model"])
    for name, df in [("overall", overall), ("by_abc", by_abc), ("by_segment", by_seg),
                     ("by_branch", by_branch), ("by_category", by_cat), ("by_fold", by_fold)]:
        df.to_csv(rep / f"backtest_metrics_{name}.csv", index=False)

    # ------------------------------------------------- model selection per segment
    metric = fc["selection_metric"]
    seg_key = by_seg["abc_class"] + "|" + by_seg["pattern"]
    by_seg["segment"] = seg_key
    best = by_seg.loc[by_seg.groupby("segment")[metric].idxmin()].set_index("segment")
    naive = by_seg[by_seg["model"] == "seasonal_naive"].set_index("segment")[metric]
    selection = best[["abc_class", "pattern", "model", "series", metric, "bias", "mase"]].copy()
    selection["seasonal_naive_" + metric] = naive.reindex(selection.index)
    selection["improvement_vs_naive"] = 1 - selection[metric] / selection["seasonal_naive_" + metric]
    selection.reset_index().to_csv(rep / "model_selection.csv", index=False)
    print("[forecast] backtest WAPE by model:\n" + overall[["model", "wape", "mae", "bias", "mase"]]
          .round(3).to_string(index=False))
    print("[forecast] selected model per segment:\n" + selection.reset_index()[
        ["segment", "model", "series", metric, "seasonal_naive_" + metric]].round(3).to_string(index=False))

    # chosen-model backtest (segment of each series taken per fold, selection pooled)
    choice = selection["model"].to_dict()
    sel_rows = []
    resid = np.full((data.n, K, H), np.nan)
    for f in range(K):
        seg = s["abc_class"].astype(str).to_numpy() + "|" + fold_seg[f]
        chosen = np.stack([fold_fc[f][choice.get(g, "lightgbm")][i] for i, g in enumerate(seg)])
        resid[:, f, :] = fold_actual[f] - chosen
        d = series_error_sums(fold_actual[f], chosen)
        d["fold"], d["model"], d["series_idx"] = f + 1, "selected", np.arange(data.n)
        d["scale"], d["pattern"] = bt.loc[(bt["fold"] == f + 1) & (bt["model"] == "seasonal_naive"),
                                          "scale"].to_numpy(), fold_seg[f]
        sel_rows.append(d)
    sel = pd.concat(sel_rows).merge(s[["branch_id", "abc_class", "category"]]
                                    .assign(series_idx=np.arange(data.n)), on="series_idx")
    sel_overall = aggregate_metrics(sel, ["model"])
    sel_overall.to_csv(rep / "backtest_metrics_selected.csv", index=False)
    aggregate_metrics(sel, ["abc_class"]).to_csv(rep / "backtest_metrics_selected_by_abc.csv", index=False)
    aggregate_metrics(sel, ["branch_id"]).to_csv(rep / "backtest_metrics_selected_by_branch.csv", index=False)

    # ---------------------------------------------------- interval calibration
    final_pattern = demand_pattern(data.Y[:, data.T - 365:])
    final_seg = s["abc_class"].astype(str).to_numpy() + "|" + final_pattern
    qlev = fc["quantiles"]
    seg_last = s["abc_class"].astype(str).to_numpy() + "|" + fold_seg[-1]
    sd_cal, _, qs_cal = interval_params(resid[:, :-1, :], seg_last, qlev)
    last_point = fold_actual[-1] - resid[:, -1, :]
    qf = apply_intervals(last_point, sd_cal, seg_last, qs_cal)
    a = fold_actual[-1]
    ok = ~np.isnan(a)
    inside = (a >= qf[:, :, 0]) & (a <= qf[:, :, -1])
    cov = pd.DataFrame({"segment": np.repeat(seg_last, H)[ok.ravel()], "inside": inside[ok]})
    coverage = cov.groupby("segment")["inside"].agg(["mean", "size"]).rename(
        columns={"mean": "p10_p90_coverage", "size": "obs"}).reset_index()
    coverage.loc[len(coverage)] = ["ALL", inside[ok].mean(), ok.sum()]
    coverage.to_csv(rep / "interval_coverage.csv", index=False)
    print(f"[forecast] P10-P90 coverage on held-out fold: {inside[ok].mean():.3f} (target 0.80)")
    sigma_d, sigma_w, qs = interval_params(resid, seg_last, qlev)

    # ---------------------------------------------------------- final forecast
    print(f"[forecast] final fit from origin {data.all_dates[data.T].date()}")
    needed = sorted(set(choice.values()) | {"lightgbm"})
    final, booster = forecast_all(data, data.T, cfg, needed)
    point = np.stack([final[choice.get(g, "lightgbm")][i] for i, g in enumerate(final_seg)])
    qf = apply_intervals(point, sigma_d, final_seg, qs)
    fdates = data.all_dates[data.T:data.T + H]
    out = pd.DataFrame({
        "date": np.tile(fdates.strftime("%Y-%m-%d"), data.n),
        "branch_id": np.repeat(s["branch_id"].astype(str).to_numpy(), H),
        "sku_id": np.repeat(s["sku_id"].astype(str).to_numpy(), H),
        "model": np.repeat([choice.get(g, "lightgbm") for g in final_seg], H),
        "forecast": point.ravel().round(3),
        **{f"p{int(q * 100)}": qf[:, :, i].ravel().round(3) for i, q in enumerate(qlev)},
    })
    out.to_csv(paths.output / "forecasts.csv", index=False)

    stats = pd.DataFrame({
        "branch_id": s["branch_id"].astype(str), "sku_id": s["sku_id"].astype(str),
        "abc_class": s["abc_class"].astype(str), "pattern": final_pattern,
        "segment": final_seg, "model": [choice.get(g, "lightgbm") for g in final_seg],
        "sigma_daily": sigma_d, "sigma_week": sigma_w,
        "forecast_mean_daily": point.mean(axis=1), "forecast_28d": point.sum(axis=1),
        "p10_28d_approx": qf[:, :, 0].sum(axis=1), "p90_28d_approx": qf[:, :, -1].sum(axis=1),
    })
    stats.to_parquet(paths.output / "forecast_series_stats.parquet", index=False)

    if booster is not None:
        booster.save_model(str(paths.output / "lightgbm_model.txt"))
        imp = pd.DataFrame({"feature": booster.feature_name(),
                            "gain": booster.feature_importance("gain"),
                            "splits": booster.feature_importance("split")}).sort_values("gain", ascending=False)
        imp["gain_share"] = imp["gain"] / imp["gain"].sum()
        imp.to_csv(rep / "feature_importance.csv", index=False)
    print(f"[forecast] wrote {len(out):,} forecast rows ({time.time() - t_start:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Backtest and forecast demand.")
    ap.add_argument("--config", default=None)
    run(ap.parse_args().config)
