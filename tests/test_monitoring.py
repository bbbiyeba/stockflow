import numpy as np
import pandas as pd

from src.monitoring import accuracy_vs_baseline, psi


def test_psi_zero_for_same_distribution_and_large_for_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 20_000)
    assert psi(a, rng.normal(0, 1, 20_000)) < 0.01
    assert psi(a, rng.normal(1.5, 1, 20_000)) > 0.25


def test_accuracy_alerts_on_degradation():
    d = pd.date_range("2026-10-01", periods=7)
    fc = pd.DataFrame(
        {
            "date": np.tile(d, 2),
            "branch_id": "B1",
            "sku_id": np.repeat(["S1", "S2"], 7),
            "forecast": 10.0,
            "p10": 5.0,
            "p90": 15.0,
        }
    )
    # A: unbiased, 5 of 7 days inside the P10-P90 band; B: demand tripled
    act = fc[["date", "branch_id", "sku_id"]].assign(qty=np.r_[[4.0, 16.0, 10, 10, 10, 10, 10], np.full(7, 30.0)])
    stats = pd.DataFrame({"branch_id": ["B1", "B1"], "sku_id": ["S1", "S2"], "abc_class": ["A", "B"]})
    base = pd.DataFrame({"abc_class": ["A", "B"], "wape": [0.3, 0.3]})
    out = accuracy_vs_baseline(fc, act, base, stats, {}).set_index("abc_class")
    assert out.loc["A", "alert"] == "ok"
    assert out.loc["B", "alert"] != "ok"
