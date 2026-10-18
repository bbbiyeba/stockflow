import numpy as np
import pandas as pd
import pytest

from src.forecast import (
    _sample_plan,
    apply_intervals,
    calibration_factors,
    cold_start_forecast,
    interval_params,
    segment_factor,
    segment_quantiles,
    write_loss_report,
)


def test_cold_start_borrows_from_peers():
    static = pd.DataFrame(
        {
            "branch_id": ["B1", "B2", "B3", "B1", "B3"],
            "sku_id": ["S1", "S1", "S1", "S2", "S2"],
            "category": ["Tools"] * 5,
        }
    )
    recent = np.array([2.0, 4.0, 0.0, 1.0, 2.0])
    hist = np.array([400, 400, 10, 400, 400])  # B3/S1 is new
    weekday = np.ones(7)
    cold, fc = cold_start_forecast(static, recent, hist, 56, weekday, 0, 7)
    assert cold.tolist() == [False, False, True, False, False]
    # S1 rate (mean of B1, B2) = 3; on S2, B3 sells 2 vs the SKU mean 1.5 -> branch factor 4/3 -> 4/day
    np.testing.assert_allclose(fc[2], 4.0)
    assert (fc[~cold] == 0).all()


def test_cold_start_new_sku_uses_category_median():
    static = pd.DataFrame({"branch_id": ["B1", "B1"], "sku_id": ["S1", "S9"], "category": ["Tools", "Tools"]})
    cold, fc = cold_start_forecast(static, np.array([3.0, 0.0]), np.array([400, 3]), 56, np.ones(7), 0, 3)
    np.testing.assert_allclose(fc[1], 3.0)


def test_intervals_monotone_non_negative_and_scaled():
    rng = np.random.default_rng(0)
    resid = rng.normal(0, 2, (50, 4, 28))
    sd, sw = interval_params(resid)
    assert abs(sd.mean() - 2) < 0.1 and np.all(sw > sd)
    z = pd.DataFrame({"segment": "A|smooth", "z": (resid / sd[:, None, None]).ravel()})
    qs = segment_quantiles(z, [0.1, 0.5, 0.9])
    q = apply_intervals(np.full((50, 28), 1.0), sd, np.array(["A|smooth"] * 50), qs)
    assert (q >= 0).all() and (np.diff(q, axis=2) >= 0).all()


def test_sample_plan_caps_rows():
    assert _sample_plan(540, 4000, 10_000_000) == (1.0, 540)
    share, days = _sample_plan(540, 490_000, 4_000_000)
    assert 490_000 * share * days <= 4_000_000 * 1.01 and days >= 56


def test_explain_selection_prefers_a_class_under_cap():
    from src.forecast import explain_selection

    abc = np.array(["C", "A", "B", "A", "C", "A"])
    rng = np.random.default_rng(0)
    assert explain_selection(abc, 10, rng).all()
    m = explain_selection(abc, 3, rng)
    assert m.sum() == 3 and m[abc == "A"].all()


def test_loss_report_flags_groups_and_series_where_selected_is_worse(tmp_path):
    # two series in branch B1; selected beats naive on s0 and loses on s1
    rows = []
    for idx, (sel, naive, ma) in enumerate([(2.0, 6.0, 5.0), (8.0, 4.0, 3.0)]):
        for model, err in [
            ("selected", sel),
            ("seasonal_naive", naive),
            ("moving_average", ma),
            ("croston_sba", 9.0),
            ("tsb", 9.0),
            ("ses", 9.0),
            ("lightgbm", sel),
        ]:
            rows.append(
                dict(
                    series_idx=idx,
                    model=model,
                    abs_err=err,
                    actual_sum=10.0,
                    n_obs=28,
                    err=0.0,
                    forecast_sum=10.0,
                    sq_err=0.0,
                    scale=1.0,
                    abc_class="A",
                    pattern="smooth",
                    branch_id="B1",
                    category="Tools",
                )
            )
    out = write_loss_report(pd.DataFrame(rows), tmp_path).set_index("dimension").loc["branch_id"]
    assert out["selected_wape"] == 0.5 and out["best_simple_model"] == "moving_average"
    assert out["best_simple_wape"] == 0.4 and bool(out["selected_loses_to_simple"])
    assert out["share_series_worse_than_naive"] == 0.5
    assert (tmp_path / "holdout_where_selected_loses.csv").exists()


def test_calibration_factors_shrink_and_clip():
    sums = pd.DataFrame(
        {
            "segment": ["A|smooth", "A|smooth", "C|lumpy", "B|erratic"],
            "act": [10_000.0, 10_000.0, 10.0, 5_000.0],
            "fc": [9_000.0, 9_000.0, 1.0, 50_000.0],
        }
    )
    f = calibration_factors(sums, prior=1000)
    assert f["A|smooth"] == pytest.approx(21_000 / 19_000)  # under-forecast segment is scaled up
    assert f["C|lumpy"] == pytest.approx(1010 / 1001)  # thin segment shrunk almost to 1
    assert f["B|erratic"] == 0.8  # clipped
    assert calibration_factors(sums.iloc[0:0], prior=1000) == {}
    assert segment_factor(f, np.array(["A|smooth", "new|seg"])).tolist() == [pytest.approx(21 / 19), 1.0]
