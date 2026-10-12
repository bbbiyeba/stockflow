import duckdb
import pandas as pd
import pytest

from src.validation import BRANCHES, FORECASTS, DataValidationError, validate, validate_large_extracts


def _branches():
    return pd.DataFrame(
        {
            "branch_id": ["B1", "B2"],
            "branch_name": ["a", "b"],
            "region": ["E", "W"],
            "lat": [40.0, 41.0],
            "lon": [-70.0, -71.0],
            "size_tier": ["Large", "Small"],
            "storage_capacity_units": [100, 200],
        }
    )


def test_valid_frame_passes():
    assert len(validate(_branches(), BRANCHES)) == 2


def test_missing_column_and_bad_values_are_reported():
    with pytest.raises(DataValidationError, match="missing required column"):
        validate(_branches().drop(columns=["region"]), BRANCHES)
    bad = _branches()
    bad.loc[0, "lat"] = 123.0
    bad.loc[1, "branch_id"] = "B1"
    with pytest.raises(DataValidationError) as e:
        validate(bad, BRANCHES)
    assert "lat" in str(e.value) and "branch_id" in str(e.value)


def test_forecast_quantiles_must_be_monotone():
    f = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-10-01"]),
            "branch_id": ["B1"],
            "sku_id": ["S1"],
            "model": ["x"],
            "forecast": [1.0],
            "p10": [2.0],
            "p50": [1.0],
            "p90": [3.0],
        }
    )
    with pytest.raises(DataValidationError, match="monotone"):
        validate(f, FORECASTS)


def test_large_extract_checks_referential_integrity():
    con = duckdb.connect()
    con.execute(
        "CREATE TABLE sales AS SELECT DATE '2026-01-01' AS date, 'B9' AS branch_id, 'S1' AS sku_id, " "1::BIGINT AS qty"
    )
    con.execute(
        "CREATE TABLE txn AS SELECT '1' AS txn_id, DATE '2026-01-01' AS date, 'B1' AS branch_id, "
        "'S1' AS sku_id, 'teleport' AS txn_type, 1::BIGINT AS qty"
    )
    with pytest.raises(DataValidationError) as e:
        validate_large_extracts(con, "sales", "txn", _branches(), pd.DataFrame({"sku_id": ["S1"]}))
    assert "unknown branch" in str(e.value) and "bad type" in str(e.value)
