import numpy as np
import pandas as pd

from src.prep import clean_sales, flag_outliers, same_weekday_impute


def test_clean_sales_dedupes_and_splits_returns():
    s = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-01"] * 3 + ["2025-01-02"]),
            "branch_id": ["B1"] * 4,
            "sku_id": ["S1"] * 4,
            "qty": [5, 5, 5, -2],
        }
    )
    s = s.iloc[[0, 1, 3]]  # one exact duplicate
    out, stats = clean_sales(s, {})
    assert stats["exact_duplicates_dropped"] == 1
    assert out["qty"].tolist() == [5, 0]
    assert out["qty_returned"].tolist() == [0, 2]


def test_same_weekday_impute_uses_same_weekday():
    Y = np.tile(np.array([10.0, 1, 1, 1, 1, 1, 0]), 8)[None, :]
    Y[0, 28] = np.nan  # a Monday-like position (value 10)
    out = same_weekday_impute(Y, 4)
    assert out[0, 28] == 10.0
    assert not np.isnan(out).any()


def test_flag_outliers_catches_keying_errors_only():
    rng = np.random.default_rng(0)
    Y = rng.poisson(5, (1, 500)).astype(float)
    Y[0, 100] = 400.0
    mask = flag_outliers(Y, 3.0, 10)
    assert mask[0, 100]
    assert mask.sum() == 1
