"""Forecast accuracy metrics.

Point metrics work on arrays where NaN in ``actual`` marks a day that must not be scored
(missing, outlier or stock-out day). Aggregation is done from additive per-series sums so
that any segment-level metric can be recomputed exactly with ``aggregate_metrics``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _mask(actual: np.ndarray, forecast: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a, f = np.asarray(actual, dtype=float), np.asarray(forecast, dtype=float)
    ok = ~np.isnan(a) & ~np.isnan(f)
    return a[ok], f[ok]


def wape(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Weighted absolute percentage error: sum|F - A| / sum|A|."""
    a, f = _mask(actual, forecast)
    denom = np.abs(a).sum()
    return float(np.abs(f - a).sum() / denom) if denom > 0 else np.nan


def mae(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Mean absolute error."""
    a, f = _mask(actual, forecast)
    return float(np.abs(f - a).mean()) if len(a) else np.nan


def bias(actual: np.ndarray, forecast: np.ndarray) -> float:
    """Relative bias: sum(F - A) / sum(A). Positive = over-forecasting."""
    a, f = _mask(actual, forecast)
    denom = np.abs(a).sum()
    return float((f - a).sum() / denom) if denom > 0 else np.nan


def mase_scale(history: np.ndarray, season: int = 7) -> np.ndarray:
    """In-sample MAE of the seasonal naive forecast, per series (rows of ``history``)."""
    h = np.asarray(history, dtype=float)
    diff = np.abs(h[:, season:] - h[:, :-season])
    with np.errstate(invalid="ignore"):
        cnt = (~np.isnan(diff)).sum(axis=1)
        scale = np.where(cnt > 0, np.nansum(diff, axis=1) / np.maximum(cnt, 1), np.nan)
    return scale


def mase(actual: np.ndarray, forecast: np.ndarray, scale: np.ndarray) -> float:
    """Mean absolute scaled error, averaged over series with a positive scale.

    ``actual`` and ``forecast`` are (n_series, horizon); ``scale`` is (n_series,).
    """
    a = np.asarray(actual, dtype=float)
    f = np.asarray(forecast, dtype=float)
    err = np.abs(f - a)
    with np.errstate(invalid="ignore"):
        cnt = (~np.isnan(err)).sum(axis=1)
        series_mae = np.where(cnt > 0, np.nansum(err, axis=1) / np.maximum(cnt, 1), np.nan)
        ratio = series_mae / scale
    ok = np.isfinite(ratio) & (scale > 0)
    return float(ratio[ok].mean()) if ok.any() else np.nan


def series_error_sums(actual: np.ndarray, forecast: np.ndarray) -> pd.DataFrame:
    """Additive per-series error components for (n_series, horizon) arrays."""
    a = np.asarray(actual, dtype=float)
    f = np.asarray(forecast, dtype=float)
    ok = ~np.isnan(a) & ~np.isnan(f)
    e = np.where(ok, f - a, 0.0)
    return pd.DataFrame(
        {
            "n_obs": ok.sum(axis=1),
            "actual_sum": np.where(ok, a, 0.0).sum(axis=1),
            "forecast_sum": np.where(ok, f, 0.0).sum(axis=1),
            "abs_err": np.abs(e).sum(axis=1),
            "err": e.sum(axis=1),
            "sq_err": (e**2).sum(axis=1),
        }
    )


def aggregate_metrics(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Aggregate per-series sums into WAPE / MAE / bias / MASE / horizon-total WAPE.

    Expects columns produced by ``series_error_sums`` plus ``scale`` (MASE denominator).
    """
    d = df.copy()
    d["series_mae"] = d["abs_err"] / d["n_obs"].where(d["n_obs"] > 0)
    d["scaled"] = d["series_mae"] / d["scale"].where(d["scale"] > 0)
    d["total_abs_err"] = (d["forecast_sum"] - d["actual_sum"]).abs()
    g = d.groupby(by, observed=True)
    out = pd.DataFrame(
        {
            "series": g.size(),
            "actual_units": g["actual_sum"].sum(),
            "wape": g["abs_err"].sum() / g["actual_sum"].sum(),
            "mae": g["abs_err"].sum() / g["n_obs"].sum(),
            "bias": g["err"].sum() / g["actual_sum"].sum(),
            "mase": g["scaled"].mean(),
            "wape_28d_total": g["total_abs_err"].sum() / g["actual_sum"].sum(),
        }
    )
    return out.reset_index()


def demand_pattern(history: np.ndarray, adi_cut: float = 1.32, cv2_cut: float = 0.49) -> np.ndarray:
    """Syntetos-Boylan classification (smooth / erratic / intermittent / lumpy) per series.

    ADI = average inter-demand interval (days per non-zero day);
    CV^2 = squared coefficient of variation of the non-zero demand sizes.
    """
    h = np.nan_to_num(np.asarray(history, dtype=float))
    nz = h > 0
    n_nz = nz.sum(axis=1)
    adi = np.where(n_nz > 0, h.shape[1] / np.maximum(n_nz, 1), np.inf)
    sizes = np.where(nz, h, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean = np.nanmean(sizes, axis=1)
            std = np.nanstd(sizes, axis=1)
    cv2 = np.where(n_nz > 1, (std / mean) ** 2, 0.0)
    out = np.where(
        adi < adi_cut, np.where(cv2 < cv2_cut, "smooth", "erratic"), np.where(cv2 < cv2_cut, "intermittent", "lumpy")
    )
    out = np.where(n_nz == 0, "no_demand", out)
    return out.astype(object)
