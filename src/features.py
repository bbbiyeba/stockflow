"""Feature engineering for the global demand model.

All dynamic features are computed from a dense ``Y`` matrix (series x day). A feature
for target day ``t`` only reads ``Y[:, :t]`` (strictly the past), which is what makes
recursive multi-step forecasting valid: during the forecast horizon the predictions
written into ``Y`` feed the short lags (7/14 days), while features lagged by 28+ days
never touch predictions inside a 28-day horizon.

Calendar, holiday and promotion features are known in advance (the promotion calendar
is planned by merchandising), so using their values at ``t`` is not leakage.
"""

from __future__ import annotations

from dataclasses import dataclass

import holidays as hol
import numpy as np
import pandas as pd

LAGS: tuple[int, ...] = (7, 14, 28, 364, 365)
# (name, lag, window): mean of Y[t-lag-window+1 .. t-lag]
ROLLING_MEANS: tuple[tuple[str, int, int], ...] = (
    ("rmean_7", 1, 7),
    ("rmean_28", 1, 28),
    ("rmean_28_lag28", 28, 28),
    ("rmean_91_lag28", 28, 91),
    ("rmean_365_lag28", 28, 365),
    ("yoy_mean_14", 358, 14),  # two weeks centred on the same date last year
)
ROLLING_STDS: tuple[tuple[str, int, int], ...] = (("rstd_28", 1, 28), ("rstd_91_lag28", 28, 91))
NONZERO_SHARES: tuple[tuple[str, int, int], ...] = (("nz_share_91_lag28", 28, 91),)

STATIC_CATEGORICAL: list[str] = ["branch_id", "sku_id", "region", "size_tier", "category"]
STATIC_NUMERIC: list[str] = ["unit_cost", "list_price", "lead_time_days", "case_pack", "obsolescence_risk"]
CALENDAR: list[str] = ["dow", "dom", "month", "weekofyear", "doy_sin", "doy_cos", "is_closed_holiday",
                       "is_minor_holiday", "is_yearend_break", "days_to_closed_holiday",
                       "days_after_closed_holiday"]
PROMO: list[str] = ["discount", "on_promo", "days_into_promo", "post_promo_7d", "net_price", "price_index"]


@dataclass
class Cumsums:
    """Prefix sums of Y used to compute any window statistic in O(1)."""

    s: np.ndarray    # sum of values
    s2: np.ndarray   # sum of squares
    cnt: np.ndarray  # count of non-NaN values
    nz: np.ndarray   # count of positive values

    @classmethod
    def of(cls, Y: np.ndarray) -> "Cumsums":
        valid = ~np.isnan(Y)
        Yz = np.where(valid, Y, 0.0).astype(np.float64)
        pad = lambda a: np.concatenate([np.zeros((a.shape[0], 1)), np.cumsum(a, axis=1)], axis=1)  # noqa: E731
        return cls(pad(Yz), pad(Yz ** 2), pad(valid.astype(np.float64)), pad((Yz > 0).astype(np.float64)))

    def window(self, arr: np.ndarray, t: np.ndarray, lag: int, w: int) -> np.ndarray:
        """Sum of ``arr``-statistic over days [t-lag-w+1, t-lag] for each column t (NaN if out of range)."""
        hi = t - lag + 1
        lo = hi - w
        ok = lo >= 0
        out = np.full((arr.shape[0], len(t)), np.nan)
        out[:, ok] = arr[:, hi[ok]] - arr[:, lo[ok]]
        return out


def dynamic_features(Y: np.ndarray, t_cols: np.ndarray, cs: Cumsums | None = None) -> dict[str, np.ndarray]:
    """Lag / rolling features for target days ``t_cols``; each array is (n_series, len(t_cols))."""
    t_cols = np.asarray(t_cols)
    cs = cs or Cumsums.of(Y)
    out: dict[str, np.ndarray] = {}
    for k in LAGS:
        idx = t_cols - k
        arr = np.full((Y.shape[0], len(t_cols)), np.nan)
        ok = idx >= 0
        arr[:, ok] = Y[:, idx[ok]]
        out[f"lag_{k}"] = arr
    with np.errstate(invalid="ignore", divide="ignore"):
        for name, lag, w in ROLLING_MEANS:
            out[name] = cs.window(cs.s, t_cols, lag, w) / cs.window(cs.cnt, t_cols, lag, w)
        for name, lag, w in ROLLING_STDS:
            n = cs.window(cs.cnt, t_cols, lag, w)
            m = cs.window(cs.s, t_cols, lag, w) / n
            var = cs.window(cs.s2, t_cols, lag, w) / n - m ** 2
            out[name] = np.sqrt(np.clip(var, 0, None))
        for name, lag, w in NONZERO_SHARES:
            out[name] = cs.window(cs.nz, t_cols, lag, w) / cs.window(cs.cnt, t_cols, lag, w)
        same_dow = np.stack([out["lag_7"], out["lag_14"], out["lag_28"],
                             _shift(Y, t_cols, 21)])
        out["dow_mean_4w"] = np.nanmean(same_dow, axis=0) if not np.all(np.isnan(same_dow)) else same_dow[0]
    return out


def _shift(Y: np.ndarray, t_cols: np.ndarray, k: int) -> np.ndarray:
    idx = t_cols - k
    arr = np.full((Y.shape[0], len(t_cols)), np.nan)
    ok = idx >= 0
    arr[:, ok] = Y[:, idx[ok]]
    return arr


def calendar_features(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Calendar and holiday features (known in advance)."""
    us = hol.US(years=range(dates.year.min() - 1, dates.year.max() + 2))
    closed_names = ("New Year's Day", "Memorial Day", "Independence Day", "Labor Day",
                    "Thanksgiving", "Christmas Day")
    closed = np.array([bool(us.get(d.date())) and any(c in us.get(d.date()) for c in closed_names)
                       and "observed" not in us.get(d.date()).lower() for d in dates])
    minor = np.array([bool(us.get(d.date())) for d in dates]) & ~closed
    # distance to the nearest closed holiday (looking +/- 14 days, includes days outside the range)
    ext = pd.date_range(dates[0] - pd.Timedelta(days=14), dates[-1] + pd.Timedelta(days=14))
    ext_closed = np.array([bool(us.get(d.date())) and any(c in us.get(d.date()) for c in closed_names)
                           and "observed" not in us.get(d.date()).lower() for d in ext])
    pos = np.nonzero(ext_closed)[0]
    idx = np.arange(len(dates)) + 14
    to_next = np.full(len(dates), 15)
    since = np.full(len(dates), 15)
    for p in pos:
        d = p - idx
        to_next = np.where((d >= 0) & (d < to_next), d, to_next)
        since = np.where((-d >= 0) & (-d < since), -d, since)
    doy = dates.dayofyear.to_numpy()
    return pd.DataFrame({
        "dow": dates.dayofweek, "dom": dates.day, "month": dates.month,
        "weekofyear": dates.isocalendar().week.to_numpy().astype(int),
        "doy_sin": np.sin(2 * np.pi * doy / 365.25), "doy_cos": np.cos(2 * np.pi * doy / 365.25),
        "is_closed_holiday": closed.astype(int), "is_minor_holiday": minor.astype(int),
        "is_yearend_break": (((dates.month == 12) & (dates.day >= 24)) | ((dates.month == 1) & (dates.day <= 2))).astype(int),
        "days_to_closed_holiday": to_next, "days_after_closed_holiday": since,
    }, index=dates)


def promo_features(discount: np.ndarray, days_into: np.ndarray, t_cols: np.ndarray,
                   list_price: np.ndarray) -> dict[str, np.ndarray]:
    """Promotion features for each series (rows already mapped from SKU to series)."""
    disc = discount[:, t_cols]
    post = np.zeros_like(disc, dtype=bool)
    for k in range(1, 8):
        idx = t_cols - k
        ok = idx >= 0
        post[:, ok] |= discount[:, idx[ok]] > 0
    post &= disc == 0
    net = list_price[:, None] * (1 - disc)
    return {"discount": disc, "on_promo": (disc > 0).astype(np.float32),
            "days_into_promo": days_into[:, t_cols], "post_promo_7d": post.astype(np.float32),
            "net_price": net, "price_index": 1 - disc}


def feature_frame(Y: np.ndarray, t_cols: np.ndarray, static: pd.DataFrame, cal: pd.DataFrame,
                  discount: np.ndarray, days_into: np.ndarray, cs: Cumsums | None = None) -> pd.DataFrame:
    """Assemble the model matrix for every series and every target day in ``t_cols``.

    Rows are ordered series-major: row = series_index * len(t_cols) + j.

    Args:
        Y: (n_series, n_days) demand history; future days may hold predictions or NaN.
        t_cols: target day indices into ``Y`` / ``cal`` / ``discount``.
        static: per-series attributes aligned with the rows of ``Y``.
        cal: calendar features indexed by day (positional, same day axis as ``Y``).
        discount, days_into: per-series promo matrices on the same day axis.
    """
    t_cols = np.asarray(t_cols)
    n, k = Y.shape[0], len(t_cols)
    cols: dict[str, np.ndarray] = {}
    for name, arr in dynamic_features(Y, t_cols, cs).items():
        cols[name] = arr.astype(np.float32).ravel()
    for name, arr in promo_features(discount, days_into, t_cols,
                                    static["list_price"].to_numpy()).items():
        cols[name] = np.asarray(arr, dtype=np.float32).ravel()
    calv = cal.iloc[t_cols]
    for c in CALENDAR:
        cols[c] = np.tile(calv[c].to_numpy(), n)
    for c in STATIC_NUMERIC:
        cols[c] = np.repeat(static[c].to_numpy(dtype=np.float32), k)
    df = pd.DataFrame(cols)
    for c in STATIC_CATEGORICAL:
        df[c] = pd.Categorical.from_codes(np.repeat(static[c].cat.codes.to_numpy(), k),
                                          categories=static[c].cat.categories)
    return df


def feature_names() -> list[str]:
    """Ordered list of model features."""
    dyn = [f"lag_{k}" for k in LAGS] + [r[0] for r in ROLLING_MEANS] + [r[0] for r in ROLLING_STDS] \
        + [r[0] for r in NONZERO_SHARES] + ["dow_mean_4w"]
    return dyn + PROMO + CALENDAR + STATIC_NUMERIC + STATIC_CATEGORICAL
