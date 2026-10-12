"""Forecasting models, vectorised across all series.

Every ``forecast_*`` function takes the history matrix ``Y`` (series x days, no NaN in the
active window) and returns an (n_series, horizon) array of daily point forecasts for the
days immediately after the history. ``first_dow`` is the weekday (Mon=0) of the first
forecast day, used to align weekday profiles.
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np

ArrayF: TypeAlias = np.ndarray


def _fill(Y: ArrayF) -> ArrayF:
    return np.nan_to_num(np.asarray(Y, dtype=float))


def forecast_seasonal_naive(Y: ArrayF, horizon: int, season: int = 7) -> ArrayF:
    """Repeat the last observed season (last week) across the horizon."""
    last = _fill(Y[:, -season:])
    reps = int(np.ceil(horizon / season))
    return np.tile(last, reps)[:, :horizon]


def forecast_moving_average(Y: ArrayF, horizon: int, window: int = 28) -> ArrayF:
    """Flat forecast equal to the mean of the last ``window`` days."""
    m = _fill(Y[:, -window:]).mean(axis=1)
    return np.repeat(m[:, None], horizon, axis=1)


def weekday_profile(Y: ArrayF, last_dow: int, weeks: int = 26, prior_strength: float = 20.0) -> ArrayF:
    """Per-series weekday multipliers (mean 1), shrunk toward the network profile.

    Args:
        Y: history matrix.
        last_dow: weekday of the last history column.
        weeks: number of trailing weeks used.
        prior_strength: pseudo-units of the network profile added to each series.
    Returns:
        (n_series, 7) multipliers indexed by weekday (Mon=0).
    """
    h = _fill(Y[:, -7 * weeks :])
    T = h.shape[1]
    dows = (last_dow - (T - 1 - np.arange(T))) % 7
    tot = np.stack([h[:, dows == d].sum(axis=1) for d in range(7)], axis=1)
    net = tot.sum(axis=0)
    net = net / net.sum() if net.sum() > 0 else np.full(7, 1 / 7)
    shrunk = (tot + prior_strength * net[None, :]) / (tot.sum(axis=1, keepdims=True) + prior_strength)
    return shrunk * 7


def _apply_profile(rate: ArrayF, profile: ArrayF, horizon: int, first_dow: int) -> ArrayF:
    dows = (first_dow + np.arange(horizon)) % 7
    return rate[:, None] * profile[:, dows]


def _deseasonalise(Y: ArrayF, profile: ArrayF, last_dow: int) -> ArrayF:
    h = _fill(Y)
    T = h.shape[1]
    dows = (last_dow - (T - 1 - np.arange(T))) % 7
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(profile[:, dows] > 0, h / profile[:, dows], 0.0)


def croston_sba(Y: ArrayF, alpha: float = 0.1, fit_days: int = 365) -> ArrayF:
    """Croston with the Syntetos-Boylan bias correction; returns the per-day demand rate."""
    h = _fill(Y[:, -fit_days:])
    n = h.shape[0]
    z = np.full(n, np.nan)  # demand size level
    p = np.full(n, np.nan)  # interval level
    q = np.ones(n)  # periods since last demand
    for t in range(h.shape[1]):
        d = h[:, t]
        hit = d > 0
        first = hit & np.isnan(z)
        z = np.where(first, d, z)
        p = np.where(first, q, p)
        upd = hit & ~first
        z = np.where(upd, z + alpha * (d - z), z)
        p = np.where(upd, p + alpha * (q - p), p)
        q = np.where(hit, 1.0, q + 1.0)
    rate = (1 - alpha / 2) * z / p
    return np.nan_to_num(rate)


def tsb(Y: ArrayF, alpha_d: float = 0.1, alpha_p: float = 0.05, fit_days: int = 365) -> ArrayF:
    """Teunter-Syntetos-Babai: smooths demand probability every period (handles obsolescence)."""
    h = _fill(Y[:, -fit_days:])
    nz = h > 0
    first = np.argmax(nz, axis=1)
    has = nz.any(axis=1)
    z = np.where(has, h[np.arange(len(h)), first], 0.0)
    p = nz[:, : max(28, 1)].mean(axis=1)
    for t in range(h.shape[1]):
        d = h[:, t]
        hit = d > 0
        p = p + alpha_p * (hit.astype(float) - p)
        z = np.where(hit, z + alpha_d * (d - z), z)
    return p * z


def ses_rate(
    Y: ArrayF, profile: ArrayF, last_dow: int, alphas: tuple[float, ...] = (0.03, 0.08, 0.15, 0.3), fit_days: int = 365
) -> ArrayF:
    """Simple exponential smoothing (ETS(A,N,N)) on weekday-adjusted demand.

    The smoothing constant is chosen per series from ``alphas`` by in-sample one-step SSE.
    Returns the per-day deseasonalised level.
    """
    x = _deseasonalise(Y[:, -fit_days:], profile, last_dow)
    n, T = x.shape
    init = x[:, :28].mean(axis=1)
    best_sse = np.full(n, np.inf)
    best_level = init.copy()
    for a in alphas:
        lvl = init.copy()
        sse = np.zeros(n)
        for t in range(T):
            e = x[:, t] - lvl
            if t >= 28:
                sse += e**2
            lvl = lvl + a * e
        better = sse < best_sse
        best_sse = np.where(better, sse, best_sse)
        best_level = np.where(better, lvl, best_level)
    return np.clip(best_level, 0, None)


def forecast_statistical(Y: ArrayF, horizon: int, last_dow: int, method: str) -> ArrayF:
    """Weekday-profiled statistical forecast: method in {'croston_sba', 'tsb', 'ses'}."""
    profile = weekday_profile(Y, last_dow)
    first_dow = (last_dow + 1) % 7
    if method == "croston_sba":
        rate = croston_sba(Y)
    elif method == "tsb":
        rate = tsb(Y)
    elif method == "ses":
        rate = ses_rate(Y, profile, last_dow)
    else:
        raise ValueError(f"unknown method {method!r}")
    return _apply_profile(rate, profile, horizon, first_dow)
