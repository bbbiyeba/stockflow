import numpy as np
import pytest

from src.metrics import aggregate_metrics, bias, demand_pattern, mae, mase, mase_scale, series_error_sums, wape


def test_point_metrics_simple_values():
    a = np.array([10.0, 0.0, 5.0, 5.0])
    f = np.array([8.0, 1.0, 5.0, 7.0])
    assert wape(a, f) == pytest.approx(5 / 20)
    assert mae(a, f) == pytest.approx(5 / 4)
    assert bias(a, f) == pytest.approx(1 / 20)


def test_nan_actuals_are_not_scored():
    a = np.array([10.0, np.nan])
    f = np.array([12.0, 1000.0])
    assert wape(a, f) == pytest.approx(0.2)
    assert mae(a, f) == pytest.approx(2.0)


def test_wape_undefined_on_zero_actuals():
    assert np.isnan(wape(np.zeros(3), np.ones(3)))


def test_mase_scale_and_value():
    hist = np.tile(np.array([1.0, 2, 3, 4, 5, 6, 7]), 3)[None, :]
    hist[0, 7:] += 1  # each week-over-week change is 1 for days 7..13, 0 afterwards
    scale = mase_scale(hist, 7)
    assert scale[0] == pytest.approx(7 / 14)
    a = np.array([[2.0, 2.0]])
    f = np.array([[3.0, 3.0]])
    assert mase(a, f, scale) == pytest.approx(1.0 / 0.5)


def test_aggregate_matches_direct_wape():
    rng = np.random.default_rng(0)
    a = rng.poisson(4, (6, 28)).astype(float)
    f = a + rng.normal(0, 1, a.shape)
    d = series_error_sums(a, f)
    d["scale"] = 1.0
    d["grp"] = "x"
    agg = aggregate_metrics(d, ["grp"])
    assert agg["wape"].iloc[0] == pytest.approx(wape(a, f))
    assert agg["bias"].iloc[0] == pytest.approx(bias(a, f))


def test_demand_pattern_classes():
    smooth = np.full(100, 5.0)
    intermittent = np.where(np.arange(100) % 10 == 0, 3.0, 0.0)
    lumpy = np.where(np.arange(100) % 10 == 0, np.where(np.arange(100) % 20 == 0, 1.0, 30.0), 0.0)
    erratic = np.tile([1.0, 20.0], 50)
    none = np.zeros(100)
    got = demand_pattern(np.stack([smooth, intermittent, lumpy, erratic, none]))
    assert list(got) == ["smooth", "intermittent", "lumpy", "erratic", "no_demand"]
