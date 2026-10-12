"""Run the full batch pipeline: generate data -> clean -> forecast -> policy -> optimise.

Each run gets a run id (logged on every line) and writes ``outputs/run_manifest.json``
with the config, package versions, git commit, stage timings and output hashes.

Usage::

    python run_pipeline.py [--config config.yaml] [--skip-generate] [--from-stage forecast]
"""

from __future__ import annotations

import argparse
import shutil
import sys

from data import generate
from src import forecast, optimizer, policy, prep
from src.config import PROJECT_ROOT, get_paths, load_config
from src.runlog import StageTimer, get_logger, run_id, update_manifest
from src.validation import DataValidationError

STAGES = [
    ("generate", generate.main),
    ("prep", prep.run),
    ("forecast", forecast.run),
    ("policy", policy.run),
    ("optimize", optimizer.run),
]
OUTPUTS = [
    "forecasts.csv",
    "forecast_series_stats.parquet",
    "lightgbm_model.txt",
    "inventory_policy.parquet",
    "transfer_recommendations.csv",
    "transfer_recommendations.parquet",
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--skip-generate", action="store_true", help="reuse existing raw data (e.g. real extracts)")
    ap.add_argument("--from-stage", choices=[s for s, _ in STAGES], default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    paths = get_paths(cfg)
    log = get_logger("pipeline")
    manifest = paths.output / "run_manifest.json"
    if manifest.exists():  # keep the previous manifest: warm starts verify the old model against it
        shutil.copy(manifest, paths.output / "run_manifest.previous.json")
    update_manifest(manifest, cfg=cfg, root=PROJECT_ROOT)
    log.info("run %s started", run_id())

    names = [s for s, _ in STAGES]
    start = names.index(args.from_stage) if args.from_stage else (1 if args.skip_generate else 0)
    for name, fn in STAGES[start:]:
        try:
            with StageTimer(manifest, name, log):
                fn(args.config)
        except DataValidationError as e:
            log.error("stage %s rejected the input data:\n%s", name, e)
            return 2
    update_manifest(manifest, outputs=[paths.output / o for o in OUTPUTS])
    log.info("run %s finished; manifest written to %s/%s", run_id(), paths.output.name, manifest.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
