"""End-to-end run of the whole pipeline on a tiny network (multi-chunk, multi-bucket)."""

import json

import pandas as pd
import pytest
import yaml

import run_pipeline
from src.config import load_config


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    root = tmp_path_factory.mktemp("tiny")
    cfg = load_config()
    cfg["paths"] = {
        k: str(root / v)
        for k, v in {
            "raw_dir": "raw",
            "processed_dir": "processed",
            "output_dir": "out",
            "reports_dir": "rep",
            "decisions_dir": "dec",
        }.items()
    }
    cfg["data"].update(
        n_branches=6, n_skus=30, start_date="2025-03-01", end_date="2026-09-30", chunk_series=60, raw_format="csv"
    )
    cfg["prep"]["bucket_series"] = 50
    cfg["forecast"].update(
        backtest_folds=2, lgbm_train_days=120, lgbm_num_boost_round=20, lgbm_max_train_rows=5_000, explain_steps=[0]
    )
    cfg["optimizer"].update(batch_arcs=40, solver_time_limit_s=20)
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    code = run_pipeline.main(["--config", str(path)])
    return root, path, code


def test_pipeline_succeeds_and_writes_outputs(tiny):
    root, _, code = tiny
    assert code == 0
    for f in [
        "forecasts.csv",
        "inventory_policy.csv",
        "transfer_recommendations.csv",
        "run_manifest.json",
        "forecast_explanations.parquet",
        "lightgbm_model.txt",
    ]:
        assert (root / "out" / f).exists(), f
    assert len(list((root / "processed" / "buckets").glob("*.npz"))) > 1  # bucketed path exercised


def test_outputs_are_consistent(tiny):
    root, _, _ = tiny
    fc = pd.read_csv(root / "out" / "forecasts.csv")
    pol = pd.read_csv(root / "out" / "inventory_policy.csv")
    assert len(fc) == len(pol) * 28
    assert (fc["p10"] <= fc["p90"]).all() and (fc["forecast"] >= 0).all()
    checks = pd.read_csv(root / "rep" / "optimizer_checks.csv")
    assert checks["violations"].sum() == 0
    man = json.loads((root / "out" / "run_manifest.json").read_text())
    assert set(man["stages"]) == {"generate", "prep", "forecast", "policy", "optimize", "sensitivity"}
    assert (root / "rep" / "sensitivity.csv").exists() and (root / "rep" / "holdout_where_selected_loses.csv").exists()
    assert all(s["ok"] for s in man["stages"].values())
    assert man["outputs"]["transfer_recommendations.csv"]["sha256"]


def test_malformed_extract_is_rejected_cleanly(tiny):
    root, path, _ = tiny
    sales = root / "raw" / "sales_daily.csv"
    original = sales.read_text()
    try:
        sales.write_text(original + "not-a-date,B01,S0001,abc\n")
        assert run_pipeline.main(["--config", str(path), "--from-stage", "prep"]) == 2
    finally:
        sales.write_text(original)


def test_monitoring_on_live_actuals(tiny):
    root, path, _ = tiny
    from src import monitoring

    fc = pd.read_csv(root / "out" / "forecasts.csv")
    actuals = fc[["date", "branch_id", "sku_id"]].assign(qty=fc["forecast"].round())
    p = root / "actuals.csv"
    actuals.to_csv(p, index=False)
    assert monitoring.main(["--actuals", str(p), "--config", str(path)]) in (0, 1)
    acc = pd.read_csv(root / "rep" / "monitoring_accuracy.csv")
    assert set(acc["abc_class"]) <= {"A", "B", "C"} and (acc["wape"] < 0.5).all()


def _render(page):
    from src.dashboard import render_page

    render_page(page)


def test_dashboard_renders_every_page(tiny, monkeypatch):
    from streamlit.testing.v1 import AppTest

    root, path, _ = tiny
    monkeypatch.setenv("STOCKFLOW_CONFIG", str(path))
    monkeypatch.setenv("STOCKFLOW_DEMO_APPROVALS", "1")
    at = AppTest.from_file("../app.py", default_timeout=60).run()
    assert not at.exception
    assert any("Transfer recommendations" in t.value for t in at.title)
    for page, title in (
        ("health", "Inventory health"),
        ("forecast", "Forecast explorer"),
        ("accuracy", "Forecast accuracy"),
        ("transfers", "Transfer recommendations"),
    ):
        pt = AppTest.from_function(_render, args=(page,), default_timeout=60).run()
        assert not pt.exception, page
        assert any(title in t.value for t in pt.title), page


def test_dashboard_empty_state_without_outputs(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    cfg = load_config()
    cfg["paths"] = {
        k: str(tmp_path / k) for k in ("raw_dir", "processed_dir", "output_dir", "reports_dir", "decisions_dir")
    }
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(cfg))
    monkeypatch.setenv("STOCKFLOW_CONFIG", str(p))
    at = AppTest.from_file("../app.py", default_timeout=60).run()
    assert not at.exception
    assert any("No pipeline outputs yet" in t.value for t in at.title)
