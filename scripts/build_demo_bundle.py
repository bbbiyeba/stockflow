"""Package a small, read-only copy of the pipeline outputs for the hosted demo dashboard.

The dashboard only reads files, so a hosted demo needs no pipeline at all: this copies the
outputs it reads into ``demo/`` (history trimmed to the last 400 days) and is pointed at by
``configs/demo.yaml``. Re-run after ``make run`` to refresh the demo.

Usage::

    python scripts/build_demo_bundle.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import get_paths, load_config  # noqa: E402

OUTPUTS = [
    "inventory_policy.parquet",
    "transfer_recommendations.parquet",
    "forecast_explanations.parquet",
    "run_manifest.json",
]
REPORTS = [
    "optimizer_impact.csv",
    "holdout_metrics_overall.csv",
    "model_selection.csv",
    "holdout_selected_vs_naive_by_segment.csv",
    "holdout_selected_by_abc.csv",
    "bias_check.csv",
    "interval_coverage.csv",
    "sensitivity.csv",
    "holdout_where_selected_loses.csv",
]


def main() -> None:
    """Copy a trimmed, read-only snapshot of the outputs into demo/ for the hosted app."""
    src = get_paths(load_config())
    demo = ROOT / "demo"
    shutil.rmtree(demo, ignore_errors=True)
    for sub in ("raw", "processed", "outputs", "reports/figures", "decisions"):
        (demo / sub).mkdir(parents=True, exist_ok=True)
    (demo / "decisions" / ".gitkeep").touch()
    (demo / "reports" / "figures" / ".gitkeep").touch()
    shutil.copy(src.raw / "branches.csv", demo / "raw" / "branches.csv")
    for f in OUTPUTS:
        shutil.copy(src.output / f, demo / "outputs" / f)
    for f in REPORTS:
        if (src.reports / f).exists():
            shutil.copy(src.reports / f, demo / "reports" / f)
    shutil.copytree(src.output / "forecasts", demo / "outputs" / "forecasts")

    hist = pd.read_parquet(src.processed / "sales_clean", columns=["date", "branch_id", "sku_id", "qty", "is_stockout"])
    hist = hist[hist["date"] > hist["date"].max() - pd.Timedelta(days=400)]
    hist["branch_id"] = hist["branch_id"].astype(str)
    # stable row order so rebuilding the bundle from the same run gives byte-identical files
    hist = hist.sort_values(["branch_id", "sku_id", "date"], kind="stable").reset_index(drop=True)
    ds.write_dataset(
        pa.Table.from_pandas(hist, preserve_index=False),
        demo / "processed" / "sales_clean",
        format="parquet",
        partitioning=ds.partitioning(pa.schema([("branch_id", pa.string())]), flavor="hive"),
        use_threads=False,
    )
    size = sum(p.stat().st_size for p in demo.rglob("*") if p.is_file())
    print(f"demo bundle: {size / 1e6:.1f} MB in {demo.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
