import numpy as np
import pytest
from scipy.stats import norm

from src.policy import expected_lost_units, horizon_sum, normal_loss, stockout_probability


def test_normal_loss_known_value():
    assert normal_loss(np.array([0.0]))[0] == pytest.approx(norm.pdf(0))


def test_expected_lost_units_degenerate_sigma():
    out = expected_lost_units(np.array([5.0, 12.0]), np.array([10.0, 10.0]), np.array([0.0, 0.0]))
    np.testing.assert_allclose(out, [5.0, 0.0])


def test_stockout_probability_half_at_mean():
    assert stockout_probability(np.array([10.0]), np.array([10.0]), np.array([3.0]))[0] == pytest.approx(0.5)


def test_horizon_sum_extrapolates_beyond_horizon():
    daily = np.ones((2, 28))
    np.testing.assert_allclose(horizon_sum(daily, np.array([7, 35])), [7.0, 35.0])
