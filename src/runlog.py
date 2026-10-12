"""Structured logging, run identity and the run manifest.

Every pipeline process shares one ``run_id`` (taken from ``STOCKFLOW_RUN_ID`` or created
on first use). Log lines carry it, and ``outputs/run_manifest.json`` records the config,
package versions, git commit, stage timings and SHA-256 hashes of the outputs so any
recommendation can be traced back to the exact run that produced it.

Logs contain counts, timings and paths only: never raw sales, cost or inventory records.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import platform
import shutil
import subprocess  # nosec B404 - fixed git command, no user input
import sys
import time
import uuid
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

_RUN_ID: str | None = None
PACKAGES = ["numpy", "pandas", "pyarrow", "duckdb", "lightgbm", "PuLP", "highspy", "scipy", "pandera"]


def run_id() -> str:
    """Process-wide run identifier (propagated to child processes via the environment)."""
    global _RUN_ID
    if _RUN_ID is None:
        _RUN_ID = os.environ.get("STOCKFLOW_RUN_ID") or (
            datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
        )
        os.environ["STOCKFLOW_RUN_ID"] = _RUN_ID
    return _RUN_ID


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="seconds"),
            "level": record.levelname,
            "run_id": run_id(),
            "stage": record.name.split(".")[-1],
            "msg": record.getMessage(),
        }
        payload.update(getattr(record, "fields", {}))
        return json.dumps(payload, default=str)


def get_logger(name: str) -> logging.Logger:
    """Logger writing to stderr; JSON lines if ``STOCKFLOW_LOG_FORMAT=json``."""
    log = logging.getLogger(f"stockflow.{name}")
    if not logging.getLogger("stockflow").handlers:
        root = logging.getLogger("stockflow")
        h = logging.StreamHandler(sys.stderr)
        if os.environ.get("STOCKFLOW_LOG_FORMAT", "text").lower() == "json":
            h.setFormatter(_JsonFormatter())
        else:
            h.setFormatter(
                logging.Formatter(f"%(asctime)s [{run_id()}] %(levelname)s %(name)s: %(message)s", "%H:%M:%S")
            )
        root.addHandler(h)
        root.setLevel(os.environ.get("STOCKFLOW_LOG_LEVEL", "INFO").upper())
        root.propagate = False
    return log


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file (streamed)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def config_hash(cfg: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _git_commit(root: Path) -> str | None:
    try:
        git = shutil.which("git")
        if git is None:
            return None
        out = subprocess.run(  # nosec B603 - fixed argv, no user input
            [git, "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=5, check=False
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for p in PACKAGES:
        with contextlib.suppress(metadata.PackageNotFoundError):
            out[p] = metadata.version(p)
    return out


class StageTimer:
    """Context manager that logs a stage's duration and records it in the manifest."""

    def __init__(self, manifest_path: Path, stage: str, log: logging.Logger) -> None:
        self.path, self.stage, self.log = manifest_path, stage, log

    def __enter__(self) -> StageTimer:
        self.t0 = time.time()
        self.log.info("stage %s started", self.stage)
        return self

    def __exit__(self, exc_type: Any, *_: Any) -> None:
        secs = round(time.time() - self.t0, 1)
        ok = exc_type is None
        self.log.info("stage %s %s in %.1fs", self.stage, "finished" if ok else "FAILED", secs)
        update_manifest(
            self.path,
            stages={self.stage: {"seconds": secs, "ok": ok, "finished_at": datetime.now(UTC).isoformat()}},
        )


def update_manifest(
    path: Path,
    *,
    cfg: dict[str, Any] | None = None,
    root: Path | None = None,
    stages: dict[str, Any] | None = None,
    outputs: list[Path] | None = None,
) -> dict[str, Any]:
    """Create or merge into the run manifest. A new run_id starts a fresh manifest."""
    man: dict[str, Any] = {}
    if path.exists():
        try:
            man = json.loads(path.read_text())
        except json.JSONDecodeError:
            man = {}
    if man.get("run_id") != run_id():
        man = {"run_id": run_id(), "created_at": datetime.now(UTC).isoformat(), "stages": {}, "outputs": {}}
    if cfg is not None:
        man["config"] = cfg
        man["config_hash"] = config_hash(cfg)
        man["seed"] = cfg.get("data", {}).get("seed")
        man["versions"] = _versions()
        if root is not None:
            man["git_commit"] = _git_commit(root)
    if stages:
        man["stages"].update(stages)
    for p in outputs or []:
        if p.exists() and p.is_file():
            man["outputs"][p.name] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(man, indent=2, default=str))
    tmp.replace(path)
    return man
