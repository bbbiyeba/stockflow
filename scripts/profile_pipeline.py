"""Profile each pipeline stage in a fresh process: wall time and peak resident memory.

Usage::

    python scripts/profile_pipeline.py [--config config.yaml] [--out reports/profile.csv]
"""

from __future__ import annotations

import argparse
import json
import subprocess  # nosec B404 - runs this interpreter on a fixed in-repo script
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGES = [
    ("generate", "data.generate", "main"),
    ("prep", "src.prep", "run"),
    ("forecast", "src.forecast", "run"),
    ("policy", "src.policy", "run"),
    ("optimize", "src.optimizer", "run"),
]
CHILD = """
import importlib, json, resource, sys, time
sys.path.insert(0, {root!r})
t0 = time.time()
ok, err = True, ""
try:
    getattr(importlib.import_module({mod!r}), {fn!r})({cfg!r})
except BaseException as e:  # report, don't crash the profiler
    ok, err = False, f"{{type(e).__name__}}: {{e}}"[:300]
print("__PROFILE__" + json.dumps({{"seconds": round(time.time() - t0, 1), "ok": ok, "error": err,
      "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)}}))
"""


def main() -> None:
    """Run every stage and record wall time and peak memory per stage."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--stages", default=None, help="comma-separated subset")
    args = ap.parse_args()
    rows = []
    for name, mod, fn in STAGES:
        if args.stages and name not in args.stages.split(","):
            continue
        code = CHILD.format(root=str(ROOT), mod=mod, fn=fn, cfg=args.config)
        t0 = time.time()
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT)  # nosec B603
        line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("__PROFILE__")), None)
        res = (
            json.loads(line[len("__PROFILE__") :])
            if line
            else {
                "seconds": round(time.time() - t0, 1),
                "ok": False,
                "peak_rss_mb": None,
                "error": f"process died (exit {proc.returncode}): {proc.stderr.strip()[-200:]}",
            }
        )
        rows.append({"stage": name, **res})
        print(
            f"{name:<9} {'ok ' if res['ok'] else 'FAIL'} {res['seconds']:>7}s  peak {res['peak_rss_mb']} MB  "
            f"{res['error']}",
            flush=True,
        )
        if not res["ok"]:
            break
    if args.out:
        import csv

        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
