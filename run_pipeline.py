"""Run the full pipeline: generate data -> clean -> forecast -> policy -> optimise.

Usage::

    python run_pipeline.py [--config config.yaml] [--skip-generate]
"""

from __future__ import annotations

import argparse
import time

from data import generate
from src import forecast, optimizer, policy, prep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--skip-generate", action="store_true", help="reuse existing raw data")
    args = ap.parse_args()
    stages = [("generate", generate.main), ("prep", prep.run), ("forecast", forecast.run),
              ("policy", policy.run), ("optimize", optimizer.run)]
    for name, fn in stages:
        if name == "generate" and args.skip_generate:
            continue
        t0 = time.time()
        print(f"=== {name} ===")
        fn(args.config)
        print(f"=== {name} done in {time.time() - t0:.0f}s ===\n")


if __name__ == "__main__":
    main()
