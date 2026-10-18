"""Rolling-origin backtest, per-segment model selection and final forecast with quantiles.

Pipeline:
1. For each of ``backtest_folds`` origins (28 days apart, ending at the last history day),
   fit every model on data strictly before the origin and forecast the next ``horizon`` days.
   The global LightGBM model is trained once per origin on a row sample drawn across all
   buckets (capped at ``lgbm_max_train_rows``) and then predicts bucket by bucket.
2. Score each model on uncensored days (not missing, not an outlier, not a stock-out day);
   report WAPE / MAE / bias / MASE by ABC class, demand pattern, category and branch.
3. Nested selection: the best model per segment (ABC class x Syntetos-Boylan pattern) is
   chosen on folds 1..K-1 and its accuracy is reported on the untouched fold K. The
   production choice is then re-made on all K folds.
4. Residual-based prediction intervals: per-series error scale x per-segment quantiles of
   standardised residuals, calibrated on folds 1..K-1 and checked for coverage on fold K.
5. Refit on all history (optionally warm-starting from the previous model), forecast,
   apply the cold-start rule to series with too little history, and write forecasts,
   per-series error statistics and per-series feature contributions (SHAP).
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import Paths, get_paths, load_config  # noqa: E402
from src.features import STATIC_CATEGORICAL, calendar_features, feature_frame, feature_names  # noqa: E402
from src.metrics import aggregate_metrics, demand_pattern, mase_scale, series_error_sums  # noqa: E402
from src.models import forecast_moving_average, forecast_seasonal_naive, forecast_statistical  # noqa: E402
from src.runlog import get_logger, sha256_file  # noqa: E402
from src.validation import FORECASTS, validate  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
log = get_logger("forecast")

MODELS: list[str] = ["seasonal_naive", "moving_average", "croston_sba", "tsb", "ses", "lightgbm"]
COLD_START = "cold_start"


# --------------------------------------------------------------------------- data access
class Network:
    """Network-wide, small inputs shared by all buckets."""

    def __init__(self, paths: Paths, horizon: int) -> None:
        self.paths = paths
        self.series = pd.read_parquet(paths.processed / "series.parquet")
        for c in STATIC_CATEGORICAL:  # global categories => identical codes in every bucket
            self.series[c] = self.series[c].astype("category")
        promo = np.load(paths.processed / "promotions_daily.npz")
        self.all_dates = pd.DatetimeIndex(promo["dates"].astype(str))
        self.sku_pos = {s: i for i, s in enumerate(promo["sku_ids"].astype(str))}
        self.discount_sku, self.days_into_sku = promo["discount"], promo["days_into"]
        self.cal = calendar_features(self.all_dates)
        self.buckets = sorted(self.series["bucket"].unique())
        first = np.load(self.bucket_file(self.buckets[0]))
        self.dates = pd.DatetimeIndex(first["dates"].astype(str))
        self.T = len(self.dates)
        if len(self.all_dates) < self.T + horizon:
            raise ValueError("promotion calendar must cover the forecast horizon")

    def bucket_file(self, k: int) -> Path:
        return self.paths.processed / "buckets" / f"bucket_{k:03d}.npz"

    @property
    def n(self) -> int:
        return len(self.series)


@dataclass
class Bucket:
    """Dense matrices for one bucket of complete SKUs."""

    k: int
    rows: np.ndarray  # positions of this bucket's series in Network.series
    static: pd.DataFrame
    Y: np.ndarray
    valid: np.ndarray
    discount: np.ndarray
    days_into: np.ndarray
    history_days: np.ndarray

    @classmethod
    def load(cls, net: Network, k: int) -> Bucket:
        w = np.load(net.bucket_file(k))
        rows = np.nonzero((net.series["bucket"] == k).to_numpy())[0]
        rows = rows[np.argsort(net.series["row"].to_numpy()[rows])]
        static = net.series.iloc[rows].reset_index(drop=True)
        sk = static["sku_id"].astype(str).map(net.sku_pos).to_numpy()
        valid = w["active"] & ~w["is_missing"] & ~w["is_outlier"] & ~w["is_stockout"]
        return cls(
            k,
            rows,
            static,
            w["qty_clean"].astype(np.float64),
            valid,
            net.discount_sku[sk],
            net.days_into_sku[sk],
            w["active"].sum(axis=1),
        )

    @property
    def n(self) -> int:
        return self.Y.shape[0]


# --------------------------------------------------------------------------- LightGBM
def _sample_plan(n_days: int, n_total: int, budget: int) -> tuple[float, int]:
    """Share of series and number of days to sample so the network stays within ``budget`` rows."""
    total = n_total * n_days
    if total <= budget:
        return 1.0, n_days
    per_series_days = budget / n_total
    days = int(min(n_days, max(56, np.ceil(per_series_days * 4))))
    return float(per_series_days / days), days


def training_rows(net: Network, b: Bucket, origin: int, cfg: dict[str, Any]) -> tuple[pd.DataFrame, np.ndarray]:
    """Feature rows and targets from one bucket for a model trained at ``origin``."""
    fc = cfg["forecast"]
    lo = max(origin - fc["lgbm_train_days"], 0)
    share, n_days = _sample_plan(origin - lo, net.n, int(fc["lgbm_max_train_rows"]))
    rng = np.random.default_rng([cfg["data"]["seed"], origin, b.k])
    t_cols = (
        np.arange(lo, origin)
        if n_days >= origin - lo
        else np.sort(rng.choice(np.arange(lo, origin), n_days, replace=False))
    )
    sel = np.arange(b.n) if share >= 1 else np.sort(rng.choice(b.n, max(1, round(b.n * share)), replace=False))
    X = feature_frame(
        b.Y[sel][:, :origin],
        t_cols,
        b.static.iloc[sel].reset_index(drop=True),
        net.cal,
        b.discount[sel],
        b.days_into[sel],
    )
    y = b.Y[sel][:, t_cols].ravel()
    keep = b.valid[sel][:, t_cols].ravel()
    return X.loc[keep, feature_names()].reset_index(drop=True), y[keep]


def fit_lightgbm(
    net: Network,
    buckets: dict[int, Bucket] | None,
    origin: int,
    cfg: dict[str, Any],
    init_model: lgb.Booster | None = None,
    rounds: int | None = None,
) -> lgb.Booster:
    """Train the global model on a capped sample across all buckets."""
    fc = cfg["forecast"]
    parts, ys = [], []
    for k in net.buckets:
        b = buckets[k] if buckets and k in buckets else Bucket.load(net, k)
        X, y = training_rows(net, b, origin, cfg)
        parts.append(X)
        ys.append(y)
    X = pd.concat(parts, ignore_index=True)
    y = np.concatenate(ys)
    params = dict(fc["lgbm_params"], seed=cfg["data"]["seed"], num_threads=0, deterministic=True, force_row_wise=True)
    data = lgb.Dataset(X, y, categorical_feature=STATIC_CATEGORICAL, free_raw_data=True)
    log.info("lightgbm: training on %s rows (origin %s)", f"{len(y):,}", net.all_dates[origin].date())
    return lgb.train(params, data, num_boost_round=rounds or fc["lgbm_num_boost_round"], init_model=init_model)


def predict_lightgbm(
    model: lgb.Booster,
    net: Network,
    b: Bucket,
    origin: int,
    horizon: int,
    explain_steps: tuple[int, ...] = (),
    explain_rows: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Recursive multi-step forecast; optionally average SHAP contributions over ``explain_steps``.

    SHAP (TreeSHAP) costs far more than prediction, so it is computed only for ``explain_rows``
    (default: all rows); unexplained rows come back as NaN.
    """
    rows = np.arange(b.n) if explain_rows is None else explain_rows
    Yw = np.full((b.n, origin + horizon), np.nan)
    Yw[:, :origin] = b.Y[:, :origin]
    contrib = None
    for h in range(horizon):
        t = origin + h
        X = feature_frame(Yw, np.array([t]), b.static, net.cal, b.discount, b.days_into)[feature_names()]
        Yw[:, t] = np.clip(np.asarray(model.predict(X)), 0, None)
        if h in explain_steps and len(rows):
            c = np.full((b.n, X.shape[1] + 1), np.nan)
            c[rows] = np.asarray(model.predict(X.iloc[rows], pred_contrib=True))  # SHAP in log-link space
            contrib = c if contrib is None else contrib + c
    if contrib is not None:
        contrib /= len(explain_steps)
    return Yw[:, origin:], contrib


def forecast_bucket(
    net: Network,
    b: Bucket,
    origin: int,
    cfg: dict[str, Any],
    booster: lgb.Booster | None,
    models: list[str],
    explain_steps: tuple[int, ...] = (),
    explain_rows: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray | None]:
    """Run ``models`` for one bucket from ``origin`` (first forecast day index)."""
    fc = cfg["forecast"]
    H = fc["horizon"]
    hist = b.Y[:, :origin]
    last_dow = int(net.all_dates[origin - 1].dayofweek)
    out: dict[str, np.ndarray] = {}
    contrib = None
    for m in models:
        if m == "seasonal_naive":
            out[m] = forecast_seasonal_naive(hist, H, fc["seasonal_period"])
        elif m == "moving_average":
            out[m] = forecast_moving_average(hist, H, fc["moving_average_window"])
        elif m in ("croston_sba", "tsb", "ses"):
            out[m] = forecast_statistical(hist, H, last_dow, m)
        elif m == "lightgbm" and booster is not None:
            out[m], contrib = predict_lightgbm(booster, net, b, origin, H, explain_steps, explain_rows)
    return {k: v.astype(np.float32) for k, v in out.items()}, contrib


# --------------------------------------------------------------------------- cold start
def cold_start_forecast(
    static: pd.DataFrame,
    recent_mean: np.ndarray,
    history_days: np.ndarray,
    min_days: int,
    weekday_index: np.ndarray,
    first_dow: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Forecast series with fewer than ``min_days`` of history from their peers.

    rate = SKU's mean daily demand across established branches x the branch's typical
    volume relative to network peers; falls back to the category median for a brand-new
    SKU. The weekday index distributes the rate over the horizon.

    Args:
        static: series attributes (branch_id, sku_id, category) for the network.
        recent_mean: mean daily demand over the last ``min_days`` days per series.
        history_days: active days of history per series.
        weekday_index: (7,) multipliers with mean 1 (Mon=0).
    Returns:
        (mask of cold-start series, (n, horizon) forecasts; rows outside the mask are 0).
    """
    cold = history_days < min_days
    fc = np.zeros((len(static), horizon))
    if not cold.any():
        return cold, fc
    d = static[["branch_id", "sku_id", "category"]].astype(str).assign(rate=recent_mean, est=~cold)
    est = d[d["est"]]
    sku_rate = est.groupby("sku_id")["rate"].mean()
    rel = (est["rate"] / est["sku_id"].map(sku_rate)).replace([np.inf, -np.inf], np.nan)
    branch_factor = rel.groupby(est["branch_id"]).median().clip(0.2, 5.0)
    cat_rate = sku_rate.groupby(est.drop_duplicates("sku_id").set_index("sku_id")["category"]).median()
    c = d[cold]
    rate = c["sku_id"].map(sku_rate)
    rate = rate.fillna(c["category"].map(cat_rate)).fillna(0.0)
    rate *= c["branch_id"].map(branch_factor).fillna(1.0)
    dows = (first_dow + np.arange(horizon)) % 7
    fc[cold] = rate.to_numpy()[:, None] * weekday_index[dows][None, :]
    return cold, fc


# --------------------------------------------------------------------------- intervals
def interval_params(resid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-series daily and weekly error scale from (n_series, n_folds, horizon) residuals."""
    n, k, H = resid.shape
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sigma_d = np.sqrt(np.nanmean(resid.reshape(n, -1) ** 2, axis=1))
        weeks = H // 7
        blocks = resid[:, :, : weeks * 7].reshape(n, k, weeks, 7)
        cnt = (~np.isnan(blocks)).sum(axis=3)
        bsum = np.where(cnt >= 4, np.nansum(blocks, axis=3) * 7 / np.maximum(cnt, 1), np.nan)
        sigma_w = np.sqrt(np.nanmean(bsum.reshape(n, -1) ** 2, axis=1))
    sigma_d = np.nan_to_num(sigma_d, nan=0.0)
    sigma_w = np.where(np.isnan(sigma_w), sigma_d * np.sqrt(7), sigma_w)
    return sigma_d, sigma_w


def segment_quantiles(z: pd.DataFrame, quantiles: list[float]) -> dict[str, np.ndarray]:
    """Quantiles of standardised residuals per segment (``z`` has columns segment, z)."""
    out = {str(seg): np.quantile(g["z"].to_numpy(), quantiles) for seg, g in z.groupby("segment")}
    out["__all__"] = np.quantile(z["z"].to_numpy(), quantiles) if len(z) else np.zeros(len(quantiles))
    return out


def apply_intervals(
    point: np.ndarray, sigma_d: np.ndarray, segments: np.ndarray, qs: dict[str, np.ndarray]
) -> np.ndarray:
    """Return (n_series, horizon, n_quantiles) non-negative, monotone quantile forecasts."""
    q = np.stack([qs.get(s, qs["__all__"]) for s in segments])
    out = point[:, :, None] + q[:, None, :] * sigma_d[:, None, None]
    return np.maximum.accumulate(np.clip(out, 0, None), axis=2)


def _standardised(
    resid: np.ndarray, sigma_d: np.ndarray, seg: np.ndarray, rng: np.random.Generator, cap: int = 400_000
) -> pd.DataFrame:
    z = resid / np.maximum(sigma_d, 1e-6)[:, None, None]
    segs = np.broadcast_to(seg[:, None, None], z.shape).ravel()
    z = z.ravel()
    ok = np.isfinite(z)
    idx = np.nonzero(ok)[0]
    if len(idx) > cap:
        idx = rng.choice(idx, cap, replace=False)
    return pd.DataFrame({"segment": segs[idx], "z": z[idx]})


# --------------------------------------------------------------------------- run
def calibration_factors(sums: pd.DataFrame, prior: float, lo: float = 0.8, hi: float = 1.25) -> dict[str, float]:
    """Per-segment multiplicative bias correction from earlier folds: (actual + prior) / (forecast + prior).

    ``sums`` has one row per series and fold with ``segment``, ``act`` and ``fc`` (scored days only).
    ``prior`` pseudo-units shrink thin segments toward 1; the result is clipped to [lo, hi].
    """
    if sums.empty:
        return {}
    g = sums.groupby("segment")[["act", "fc"]].sum()
    f = ((g["act"] + prior) / (g["fc"] + prior)).clip(lo, hi)
    return {str(k): float(v) for k, v in f.items()}


def segment_factor(factors: dict[str, float], segments: np.ndarray) -> np.ndarray:
    """Factor per row for an array of segment labels (1.0 where a segment has no factor yet)."""
    return np.array([factors.get(str(g), 1.0) for g in segments], dtype=np.float64)


def write_loss_report(held: pd.DataFrame, rep: Path) -> pd.DataFrame:
    """Where the selected model loses to a simpler one on the held-out fold.

    Writes ``holdout_where_selected_loses.csv``: for the whole network (``all``) and per ABC class,
    pattern, branch and category, the selected WAPE against seasonal naive and the best simple
    model, and the share of individual series (with demand) on which the selected model has more
    absolute error than seasonal naive. Returns the group table.
    """
    simple = [m for m in MODELS if m != "lightgbm"]
    held = held.assign(all="all")
    parts = []
    for dim in ("all", "abc_class", "pattern", "branch_id", "category"):
        g = aggregate_metrics(held, [dim, "model"]).pivot(index=dim, columns="model", values="wape")
        wide = held.pivot_table(index="series_idx", columns="model", values="abs_err", aggfunc="sum")
        grp = held.drop_duplicates("series_idx").set_index("series_idx")[dim]
        has_demand = held[held["model"] == "selected"].set_index("series_idx")["actual_sum"] > 0
        loses = (wide["selected"] > wide["seasonal_naive"])[has_demand]
        parts.append(
            pd.DataFrame(
                {
                    "dimension": dim,
                    "group": g.index.astype(str),
                    "selected_wape": g["selected"].to_numpy(),
                    "seasonal_naive_wape": g["seasonal_naive"].to_numpy(),
                    "best_simple_model": g[simple].idxmin(axis=1).to_numpy(),
                    "best_simple_wape": g[simple].min(axis=1).to_numpy(),
                    "share_series_worse_than_naive": loses.groupby(grp[loses.index]).mean().reindex(g.index).to_numpy(),
                }
            )
        )
    out = pd.concat(parts, ignore_index=True)
    out["selected_loses_to_simple"] = out["selected_wape"] > out["best_simple_wape"]
    out.to_csv(rep / "holdout_where_selected_loses.csv", index=False)
    return out


def run(config_path: str | None = None) -> None:
    """Backtest, select models, and write the final forecast."""
    t_start = time.time()
    cfg = load_config(config_path)
    fc = cfg["forecast"]
    H, K = fc["horizon"], fc["backtest_folds"]
    qlev = fc["quantiles"]
    paths = get_paths(cfg)
    net = Network(paths, H)
    s = net.series
    abc_all = s["abc_class"].astype(str).to_numpy()
    rng = np.random.default_rng(cfg["data"]["seed"])
    log.info(
        "%s series in %d bucket(s), history %s .. %s",
        f"{net.n:,}",
        len(net.buckets),
        net.dates[0].date(),
        net.dates[-1].date(),
    )
    keep_in_memory = len(net.buckets) == 1
    cache = {net.buckets[0]: Bucket.load(net, net.buckets[0])} if keep_in_memory else None
    get = (lambda k: cache[k]) if cache else (lambda k: Bucket.load(net, k))  # noqa: E731

    bt_dir = paths.processed / "backtest"
    shutil.rmtree(bt_dir, ignore_errors=True)
    bt_dir.mkdir(parents=True)

    # ---------------------------------------------------------------- backtest
    origins = [net.T - H * k for k in range(K, 0, -1)]
    rows = []
    # LightGBM bias calibration: fold f is corrected with factors learned on folds < f only
    calibrate, prior = bool(fc.get("bias_calibration", True)), float(fc.get("bias_calibration_prior_units", 1000))
    cal_sums: list[pd.DataFrame] = []
    cal_log = []
    for f, origin in enumerate(origins):
        t0 = time.time()
        booster = fit_lightgbm(net, cache, origin, cfg)
        factors = calibration_factors(pd.concat(cal_sums, ignore_index=True), prior) if cal_sums else {}
        cal_log += [{"fold": f + 1, "segment": g, "factor": v} for g, v in factors.items()]
        fold_sums = []
        for k in net.buckets:
            b = get(k)
            preds, _ = forecast_bucket(net, b, origin, cfg, booster, MODELS)
            actual = np.where(b.valid[:, origin : origin + H], b.Y[:, origin : origin + H], np.nan).astype(np.float32)
            pattern = demand_pattern(b.Y[:, origin - 365 : origin])
            scale = mase_scale(b.Y[:, origin - 365 : origin], fc["seasonal_period"])
            if calibrate and "lightgbm" in preds:
                seg_k = np.char.add(np.char.add(abc_all[b.rows].astype("U1"), "|"), pattern.astype("U12"))
                ok = ~np.isnan(actual)
                raw = preds["lightgbm"]
                fold_sums.append(
                    pd.DataFrame(
                        {"segment": seg_k, "act": np.where(ok, actual, 0).sum(1), "fc": np.where(ok, raw, 0).sum(1)}
                    )
                )
                preds["lightgbm"] = (raw * segment_factor(factors, seg_k)[:, None]).astype(np.float32)
            np.savez(
                bt_dir / f"fold{f}_bucket{k:03d}.npz",
                actual=actual,
                pattern=pattern.astype("U12"),
                scale=scale,
                **preds,  # type: ignore[arg-type]  # numpy's stub types **kwds too narrowly
            )
            for m, p in preds.items():
                d = series_error_sums(actual, p)
                d["fold"], d["model"], d["series_idx"] = f + 1, m, b.rows
                d["scale"], d["pattern"] = scale, pattern
                rows.append(d)
        cal_sums += fold_sums
        log.info("fold %d/%d (origin %s) done in %.0fs", f + 1, K, net.all_dates[origin].date(), time.time() - t0)
    final_factors = calibration_factors(pd.concat(cal_sums, ignore_index=True), prior) if cal_sums else {}
    cal_log += [{"fold": "final", "segment": g, "factor": v} for g, v in final_factors.items()]
    pd.DataFrame(cal_log, columns=["fold", "segment", "factor"]).to_csv(
        paths.reports / "bias_calibration.csv", index=False
    )
    if final_factors:
        log.info("bias calibration factors: %s", {g: round(v, 3) for g, v in sorted(final_factors.items())})
    bt = pd.concat(rows, ignore_index=True)
    attrs = s[["branch_id", "sku_id", "abc_class", "category"]].assign(series_idx=np.arange(net.n))
    bt = bt.merge(attrs, on="series_idx")
    bt["segment"] = bt["abc_class"].astype(str) + "|" + bt["pattern"].astype(str)

    rep = paths.reports
    for name, by in [
        ("overall", ["model"]),
        ("by_abc", ["abc_class", "model"]),
        ("by_segment", ["abc_class", "pattern", "model"]),
        ("by_branch", ["branch_id", "model"]),
        ("by_category", ["category", "model"]),
        ("by_fold", ["fold", "model"]),
    ]:
        aggregate_metrics(bt, by).to_csv(rep / f"backtest_metrics_{name}.csv", index=False)

    # ------------------------------------------------ nested model selection
    metric = fc["selection_metric"]

    def choose(frame: pd.DataFrame) -> pd.DataFrame:
        seg = aggregate_metrics(frame, ["segment", "abc_class", "pattern", "model"])
        best = seg.loc[seg.groupby("segment")[metric].idxmin()].set_index("segment")
        naive = seg[seg["model"] == "seasonal_naive"].set_index("segment")[metric]
        out = best[["abc_class", "pattern", "model", "series", metric, "bias", "mase"]].copy()
        out[f"seasonal_naive_{metric}"] = naive.reindex(out.index)
        out["improvement_vs_naive"] = 1 - out[metric] / out[f"seasonal_naive_{metric}"]
        return out

    early = bt[bt["fold"] < K]
    choice_early = choose(early)["model"].to_dict()
    final_sel = choose(bt)
    choice = final_sel["model"].to_dict()
    final_sel.reset_index().to_csv(rep / "model_selection.csv", index=False)

    # held-out evaluation of the selection rule on fold K
    last = bt[bt["fold"] == K]
    picked = last[last["model"] == last["segment"].map(choice_early).fillna("lightgbm")].assign(model="selected")
    held = pd.concat([last, picked], ignore_index=True)
    held_overall = aggregate_metrics(held, ["model"]).sort_values("wape")
    held_overall.to_csv(rep / "holdout_metrics_overall.csv", index=False)
    held_seg = aggregate_metrics(held[held["model"].isin(["selected", "seasonal_naive"])], ["segment", "model"]).pivot(
        index="segment", columns="model", values=metric
    )
    held_seg["selected_beats_naive"] = held_seg["selected"] < held_seg["seasonal_naive"]
    held_seg.reset_index().to_csv(rep / "holdout_selected_vs_naive_by_segment.csv", index=False)
    write_loss_report(held, rep)
    for name, by in [
        ("by_abc", ["abc_class"]),
        ("by_branch", ["branch_id"]),
        ("by_category", ["category"]),
        ("by_pattern", ["pattern"]),
    ]:
        aggregate_metrics(picked, by).to_csv(rep / f"holdout_selected_{name}.csv", index=False)
    bias_rows = []
    for dim in ("abc_class", "branch_id", "category", "pattern"):
        a = aggregate_metrics(picked, [dim]).rename(columns={dim: "group"}).assign(dimension=dim)
        bias_rows.append(a[["dimension", "group", "actual_units", "bias", "wape"]])
    bias = pd.concat(bias_rows, ignore_index=True)
    # bias on a sliver of volume is noise: groups under 0.5% of units are reported but not flagged
    share = bias["actual_units"] / bias.groupby("dimension")["actual_units"].transform("sum")
    bias["flag"] = np.select(
        [share < 0.005, bias["bias"].abs() > fc.get("bias_flag_threshold", 0.05)], ["too_small", "check"], "ok"
    )
    bias.to_csv(rep / "bias_check.csv", index=False)
    sel_row = held_overall.set_index("model").loc["selected"]
    log.info(
        "held-out fold %d: selected WAPE %.3f bias %+.3f MASE %.3f | seasonal naive WAPE %.3f",
        K,
        sel_row["wape"],
        sel_row["bias"],
        sel_row["mase"],
        held_overall.set_index("model").loc["seasonal_naive", "wape"],
    )

    # ------------------------------------- residuals of the chosen model, per bucket
    sig_cal, sig_d, sig_w = (np.zeros(net.n) for _ in range(3))
    seg_last = np.empty(net.n, dtype=object)
    z_cal, z_all, last_fold = [], [], []
    for k in net.buckets:
        files = [np.load(bt_dir / f"fold{f}_bucket{k:03d}.npz") for f in range(K)]
        rows_k = s.index[s["bucket"] == k].to_numpy()
        rows_k = rows_k[np.argsort(s["row"].to_numpy()[rows_k])]
        abc = abc_all[rows_k]
        resid_e = np.full((len(rows_k), K, H), np.nan, dtype=np.float32)
        resid_f = np.full_like(resid_e, np.nan)
        for f, w in enumerate(files):
            seg = np.char.add(np.char.add(abc.astype("U1"), "|"), w["pattern"])
            for m in set(choice.values()) | set(choice_early.values()) | {"lightgbm"}:
                me = np.array([choice_early.get(g, "lightgbm") == m for g in seg])
                mf = np.array([choice.get(g, "lightgbm") == m for g in seg])
                resid_e[me, f] = w["actual"][me] - w[m][me]
                resid_f[mf, f] = w["actual"][mf] - w[m][mf]
        seg_k = np.char.add(np.char.add(abc.astype("U1"), "|"), files[-1]["pattern"]).astype(object)
        sd_cal, _ = interval_params(resid_e[:, :-1])
        sd, sw = interval_params(resid_f)
        sig_cal[rows_k], sig_d[rows_k], sig_w[rows_k], seg_last[rows_k] = sd_cal, sd, sw, seg_k
        z_cal.append(_standardised(resid_e[:, :-1], sd_cal, seg_k, rng))
        z_all.append(_standardised(resid_f, sd, seg_k, rng))
        last_fold.append((rows_k, files[-1]["actual"], files[-1]["actual"] - resid_e[:, -1]))
    qs_cal = segment_quantiles(pd.concat(z_cal, ignore_index=True), qlev)
    qs = segment_quantiles(pd.concat(z_all, ignore_index=True), qlev)

    cov_parts, held_parts = [], []
    held_dates = net.all_dates[origins[-1] : origins[-1] + H]
    for rows_k, a, point in last_fold:
        qf = apply_intervals(point, sig_cal[rows_k], seg_last[rows_k], qs_cal)
        ok = ~np.isnan(a)
        inside = (a >= qf[:, :, 0]) & (a <= qf[:, :, -1])
        cov_parts.append(pd.DataFrame({"segment": np.repeat(seg_last[rows_k], H)[ok.ravel()], "inside": inside[ok]}))
        held_parts.append(
            pd.DataFrame(
                {
                    "branch_id": np.repeat(s["branch_id"].astype(str).to_numpy()[rows_k], H),
                    "sku_id": np.repeat(s["sku_id"].astype(str).to_numpy()[rows_k], H),
                    "date": np.tile(held_dates, len(rows_k)),
                    "forecast": point.ravel().astype(np.float32),
                    **{f"p{int(q * 100)}": qf[:, :, i].ravel().astype(np.float32) for i, q in enumerate(qlev)},
                    "actual": a.ravel().astype(np.float32),  # NaN = not scored (missing / outlier / stock-out)
                }
            )
        )
    # held-out fold forecasts with calibrated intervals, for forecast-vs-actual charts
    pd.concat(held_parts, ignore_index=True).to_parquet(rep / "holdout_forecasts.parquet", index=False)
    cov_df = pd.concat(cov_parts, ignore_index=True)
    coverage = (
        cov_df.groupby("segment")["inside"]
        .agg(["mean", "size"])
        .rename(columns={"mean": "p10_p90_coverage", "size": "obs"})
        .reset_index()
    )
    total = pd.DataFrame({"segment": ["ALL"], "p10_p90_coverage": [cov_df["inside"].mean()], "obs": [len(cov_df)]})
    coverage = pd.concat([coverage, total], ignore_index=True)
    coverage.to_csv(rep / "interval_coverage.csv", index=False)
    log.info("P10-P90 coverage on held-out fold: %.3f (target 0.80)", cov_df["inside"].mean())

    # ---------------------------------------------------------- final forecast
    origin = net.T
    model_path = paths.output / "lightgbm_model.txt"
    init, rounds = None, None
    if fc.get("warm_start") and model_path.exists():
        init, rounds = _load_verified_model(paths, model_path), int(fc.get("warm_start_rounds", 100))
        log.info("warm start: continuing from previous model for %d rounds", rounds)
    booster = fit_lightgbm(net, cache, origin, cfg, init_model=init, rounds=rounds)
    booster.save_model(str(model_path))

    weekday = _network_weekday_index(net, get)
    first_dow = int(net.all_dates[origin].dayofweek)
    recent = np.zeros(net.n)
    hist_days = np.zeros(net.n, dtype=int)
    for k in net.buckets:
        b = get(k)
        recent[b.rows] = np.nanmean(b.Y[:, -fc.get("cold_start_days", 56) :], axis=1)
        hist_days[b.rows] = b.history_days
    cold, cold_fc = cold_start_forecast(
        s, np.nan_to_num(recent), hist_days, int(fc.get("cold_start_days", 56)), weekday, first_dow, H
    )
    final_pattern = np.empty(net.n, dtype=object)
    point = np.zeros((net.n, H), dtype=np.float64)
    contrib_all = np.full((net.n, len(feature_names()) + 1), np.nan, dtype=np.float32)
    needed = sorted(set(choice.values()) | {"lightgbm"})
    explain_steps = tuple(int(x) for x in fc.get("explain_steps", [0, 7, 14, 21]))
    explain = explain_selection(abc_all, int(fc.get("explain_max_series", 60_000)), rng)
    for k in net.buckets:
        b = get(k)
        preds, contrib = forecast_bucket(
            net, b, origin, cfg, booster, needed, explain_steps, np.nonzero(explain[b.rows])[0]
        )
        pat = demand_pattern(b.Y[:, origin - 365 : origin])
        seg = np.char.add(np.char.add(abc_all[b.rows].astype("U1"), "|"), pat.astype("U12"))
        final_pattern[b.rows] = pat
        if calibrate and "lightgbm" in preds:
            fac = segment_factor(final_factors, seg)
            preds["lightgbm"] = (preds["lightgbm"] * fac[:, None]).astype(np.float32)
            if contrib is not None:  # keep SHAP additive in log space: the factor moves the base value
                contrib[:, -1] += np.log(fac)
        point[b.rows] = np.stack([preds[choice.get(g, "lightgbm")][i] for i, g in enumerate(seg)])
        if contrib is not None:
            contrib_all[b.rows] = contrib
    final_pattern[cold] = COLD_START
    point[cold] = cold_fc[cold]
    final_seg = (pd.Series(abc_all) + "|" + pd.Series(final_pattern).astype(str)).to_numpy()
    model_used = np.array([choice.get(g, "lightgbm") for g in final_seg], dtype=object)
    model_used[cold] = COLD_START
    # cold-start error scale: typical sigma/mean ratio of established series of the same ABC class
    ratio = pd.Series(sig_d / np.maximum(point.mean(axis=1), 1e-3)).where(~cold)
    ratio_by_abc = ratio.groupby(abc_all).median()
    cs_ratio = pd.Series(abc_all).map(ratio_by_abc).to_numpy()
    sig_d = np.where(cold, cs_ratio * point.mean(axis=1), sig_d)
    sig_w = np.where(cold, sig_d * np.sqrt(7), sig_w)
    qf = apply_intervals(point, sig_d, final_seg, qs)

    fdates = net.all_dates[origin : origin + H]
    out = pd.DataFrame(
        {
            "date": np.tile(fdates, net.n),
            "branch_id": np.repeat(s["branch_id"].astype(str).to_numpy(), H),
            "sku_id": np.repeat(s["sku_id"].astype(str).to_numpy(), H),
            "model": np.repeat(model_used, H),
            "forecast": point.ravel().round(3),
            **{f"p{int(q * 100)}": qf[:, :, i].ravel().round(3) for i, q in enumerate(qlev)},
        }
    ).sort_values(["branch_id", "sku_id", "date"], kind="stable")
    validate(out, FORECASTS)
    shutil.rmtree(paths.output / "forecasts", ignore_errors=True)
    ds.write_dataset(
        pa.Table.from_pandas(out, preserve_index=False),
        paths.output / "forecasts",
        format="parquet",
        partitioning=ds.partitioning(pa.schema([("branch_id", pa.string())]), flavor="hive"),
    )
    if fc.get("write_csv", True):
        out.assign(date=out["date"].dt.strftime("%Y-%m-%d")).to_csv(paths.output / "forecasts.csv", index=False)

    stats = (
        pd.DataFrame(
            {
                "branch_id": s["branch_id"].astype(str),
                "sku_id": s["sku_id"].astype(str),
                "abc_class": abc_all,
                "pattern": final_pattern.astype(str),
                "segment": final_seg,
                "model": model_used.astype(str),
                "is_cold_start": cold,
                "history_days": hist_days,
                "sigma_daily": sig_d,
                "sigma_week": sig_w,
                "forecast_mean_daily": point.mean(axis=1),
                "forecast_28d": point.sum(axis=1),
                "p10_28d_approx": qf[:, :, 0].sum(axis=1),
                "p90_28d_approx": qf[:, :, -1].sum(axis=1),
            }
        )
        .sort_values(["branch_id", "sku_id"])
        .reset_index(drop=True)
    )
    stats.to_parquet(paths.output / "forecast_series_stats.parquet", index=False)

    _write_explanations(paths, s, contrib_all, model_used)
    imp = pd.DataFrame(
        {
            "feature": booster.feature_name(),
            "gain": booster.feature_importance("gain"),
            "splits": booster.feature_importance("split"),
        }
    ).sort_values("gain", ascending=False)
    imp["gain_share"] = imp["gain"] / imp["gain"].sum()
    imp.to_csv(rep / "feature_importance.csv", index=False)
    shutil.rmtree(bt_dir, ignore_errors=True)
    log.info(
        "wrote %s forecast rows; %d cold-start series (%.0fs)", f"{len(out):,}", int(cold.sum()), time.time() - t_start
    )


def _network_weekday_index(net: Network, get: Any) -> np.ndarray:
    """Network weekday multipliers (mean 1) from the last 26 weeks."""
    tot = np.zeros(7)
    for k in net.buckets:
        y = np.nan_to_num(get(k).Y[:, -182:]).sum(axis=0)
        dows = net.dates[-182:].dayofweek.to_numpy()
        tot += np.bincount(dows, weights=y, minlength=7)
    return tot / tot.mean() if tot.sum() > 0 else np.ones(7)


def explain_selection(abc: np.ndarray, cap: int, rng: np.random.Generator) -> np.ndarray:
    """Which series get SHAP explanations: all if they fit under ``cap``, else A-class first, then a sample."""
    n = len(abc)
    if n <= cap:
        return np.ones(n, dtype=bool)
    order = np.lexsort((rng.random(n), abc != "A"))  # A-class first, random within class
    mask = np.zeros(n, dtype=bool)
    mask[order[:cap]] = True
    return mask


def _write_explanations(
    paths: Paths, s: pd.DataFrame, contrib: np.ndarray, model_used: np.ndarray, top: int = 5
) -> None:
    """Top contributing features per series (mean SHAP over the explained horizon days)."""
    names = np.array(feature_names())
    keep = ~np.isnan(contrib[:, 0])  # series that were explained
    c = contrib[keep, :-1]
    order = np.argsort(-np.abs(c), axis=1)[:, :top]
    n = int(keep.sum())
    df = pd.DataFrame(
        {
            "branch_id": np.repeat(s["branch_id"].astype(str).to_numpy()[keep], top),
            "sku_id": np.repeat(s["sku_id"].astype(str).to_numpy()[keep], top),
            "rank": np.tile(np.arange(1, top + 1), n),
            "feature": names[order].ravel(),
            "shap_log": np.take_along_axis(c, order, axis=1).ravel(),
            "model_used": np.repeat(model_used.astype(str)[keep], top),
        }
    )
    df.to_parquet(paths.output / "forecast_explanations.parquet", index=False)


def _load_verified_model(paths: Paths, model_path: Path) -> lgb.Booster:
    """Load the previous model only if its hash matches the one recorded in the last run manifest."""
    import json

    expected = set()
    for name in ("run_manifest.json", "run_manifest.previous.json"):
        man_path = paths.output / name
        if man_path.exists():
            expected.add(json.loads(man_path.read_text()).get("outputs", {}).get(model_path.name, {}).get("sha256"))
    if sha256_file(model_path) not in expected - {None}:
        raise ValueError("previous model file does not match the run manifest; refusing to warm-start from it")
    return lgb.Booster(model_file=str(model_path))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Backtest and forecast demand.")
    ap.add_argument("--config", default=None)
    run(ap.parse_args().config)
