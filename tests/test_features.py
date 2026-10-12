import numpy as np
import pandas as pd

from src.features import Cumsums, calendar_features, dynamic_features


def _ramp(n: int = 3, T: int = 400) -> np.ndarray:
    return np.arange(T, dtype=float)[None, :].repeat(n, axis=0) + np.arange(n)[:, None] * 1000


def test_lags_read_the_right_day_and_are_nan_before_history():
    Y = _ramp()
    t = np.array([5, 30, 399])
    f = dynamic_features(Y, t)
    assert np.isnan(f["lag_7"][0, 0])
    np.testing.assert_allclose(f["lag_7"][:, 1], Y[:, 23])
    np.testing.assert_allclose(f["lag_28"][:, 2], Y[:, 371])
    np.testing.assert_allclose(f["lag_365"][:, 2], Y[:, 34])
    assert np.isnan(f["lag_365"][0, 1])


def test_rolling_stats_match_numpy():
    rng = np.random.default_rng(0)
    Y = rng.poisson(3, (4, 200)).astype(float)
    t = np.array([150])
    f = dynamic_features(Y, t)
    np.testing.assert_allclose(f["rmean_7"][:, 0], Y[:, 143:150].mean(axis=1))
    np.testing.assert_allclose(f["rmean_28_lag28"][:, 0], Y[:, 95:123].mean(axis=1))
    np.testing.assert_allclose(f["rstd_28"][:, 0], Y[:, 122:150].std(axis=1), atol=1e-9)
    np.testing.assert_allclose(f["nz_share_91_lag28"][:, 0], (Y[:, 32:123] > 0).mean(axis=1))


def test_rolling_mean_ignores_nan_days():
    Y = np.ones((1, 50))
    Y[0, 45] = np.nan
    f = dynamic_features(Y, np.array([49]))
    assert f["rmean_7"][0, 0] == 1.0


def test_no_feature_reads_the_target_day_or_later():
    """Leakage guard: overwriting Y[:, t:] must not change any feature at t."""
    rng = np.random.default_rng(1)
    Y = rng.poisson(2, (5, 500)).astype(float)
    t = np.array([420])
    base = dynamic_features(Y, t)
    Y2 = Y.copy()
    Y2[:, 420:] = 9999.0
    leak = dynamic_features(Y2, t, Cumsums.of(Y2))
    for name, arr in base.items():
        np.testing.assert_allclose(arr, leak[name], err_msg=name)


def test_calendar_flags_holidays_and_distances():
    cal = calendar_features(pd.date_range("2025-12-20", "2026-01-05"))
    assert cal.loc["2025-12-25", "is_closed_holiday"] == 1
    assert cal.loc["2025-12-24", "days_to_closed_holiday"] == 1
    assert cal.loc["2025-12-26", "days_after_closed_holiday"] == 1
    assert cal.loc["2026-01-01", "is_closed_holiday"] == 1
    assert cal.loc["2025-12-22", "dow"] == 0
