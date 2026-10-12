import numpy as np
import pytest

from src.models import (
    croston_sba,
    forecast_moving_average,
    forecast_seasonal_naive,
    forecast_statistical,
    tsb,
    weekday_profile,
)


def test_seasonal_naive_repeats_last_week():
    Y = np.arange(21, dtype=float)[None, :]
    f = forecast_seasonal_naive(Y, 10)
    np.testing.assert_allclose(f[0], [14, 15, 16, 17, 18, 19, 20, 14, 15, 16])


def test_moving_average_flat():
    Y = np.concatenate([np.zeros(10), np.full(28, 2.0)])[None, :]
    np.testing.assert_allclose(forecast_moving_average(Y, 5, 28), 2.0)


def test_croston_constant_intermittent_rate():
    # 4 units every 4th day -> true rate 1/day; SBA applies (1 - alpha/2)
    Y = np.where(np.arange(400) % 4 == 0, 4.0, 0.0)[None, :]
    assert croston_sba(Y, alpha=0.1)[0] == pytest.approx(0.95, rel=0.02)


def test_tsb_decays_for_dead_series():
    Y = np.concatenate([np.full(200, 3.0), np.zeros(165)])[None, :]
    assert tsb(Y)[0] < 0.01


def test_weekday_profile_mean_one_and_shape():
    rng = np.random.default_rng(0)
    Y = rng.poisson(5, (3, 182)).astype(float)
    p = weekday_profile(Y, last_dow=6)
    assert p.shape == (3, 7)
    np.testing.assert_allclose(p.mean(axis=1), 1.0)


def test_statistical_forecasts_are_non_negative():
    rng = np.random.default_rng(0)
    Y = rng.poisson(0.3, (20, 400)).astype(float)
    for m in ("croston_sba", "tsb", "ses"):
        f = forecast_statistical(Y, 28, last_dow=2, method=m)
        assert f.shape == (20, 28) and (f >= 0).all()
