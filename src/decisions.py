"""Append-only log of planner decisions on transfer recommendations.

The optimiser only *recommends* transfers. A planner approves or rejects each one in the
dashboard; every decision is appended (never edited in place) to
``<decisions_dir>/transfer_decisions.csv`` with who decided, when, on which pipeline run,
and an optional note. The latest decision per transfer wins. Execution in the ERP/WMS is
a separate, manual step outside this tool.

For multi-user production use, swap this file store for a database table with the same
columns; the functions below are the only place that touches storage.
"""

from __future__ import annotations

import csv
import fcntl
import re
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

COLUMNS = ["decided_at", "run_id", "transfer_id", "decision", "reviewer", "role", "note"]
DECISIONS = ("approved", "rejected")
_ID = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")
MAX_NOTE = 500


class DecisionError(ValueError):
    """Raised for an invalid decision record."""


def _clean_text(s: str, limit: int) -> str:
    """Single-line, length-limited text; neutralises spreadsheet formula injection."""
    s = " ".join(str(s).split())[:limit]
    return "'" + s if s[:1] in ("=", "+", "-", "@") else s


def record(path: Path, run_id: str, rows: list[dict[str, str]], reviewer: str, role: str) -> int:
    """Validate and append decisions; returns the number written.

    Each row needs ``transfer_id`` and ``decision`` (approved / rejected) and may carry ``note``.
    """
    if role != "approver":
        raise DecisionError("only users with the approver role can record decisions")
    reviewer = _clean_text(reviewer, 80)
    if not reviewer:
        raise DecisionError("a reviewer name is required")
    if not _ID.match(run_id):
        raise DecisionError("invalid run id")
    now = datetime.now(UTC).isoformat(timespec="seconds")
    out = []
    for r in rows:
        tid, dec = str(r.get("transfer_id", "")), str(r.get("decision", "")).lower()
        if not _ID.match(tid):
            raise DecisionError(f"invalid transfer id {tid!r}")
        if dec not in DECISIONS:
            raise DecisionError(f"decision must be one of {DECISIONS}, got {dec!r}")
        out.append(
            {
                "decided_at": now,
                "run_id": run_id,
                "transfer_id": tid,
                "decision": dec,
                "reviewer": reviewer,
                "role": role,
                "note": _clean_text(r.get("note") or "", MAX_NOTE),
            }
        )
    if not out:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)  # serialise concurrent writers
        try:
            new = fh.tell() == 0
            w = csv.DictWriter(fh, fieldnames=COLUMNS)
            if new:
                w.writeheader()
            w.writerows(out)
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)
    return len(out)


def history(path: Path) -> pd.DataFrame:
    """All decisions recorded so far (empty frame if none)."""
    if not path.exists():
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def latest(path: Path, run_id: str | None = None) -> pd.DataFrame:
    """Latest decision per transfer (optionally for one pipeline run)."""
    h = history(path)
    if run_id is not None:
        h = h[h["run_id"] == run_id]
    return h.drop_duplicates("transfer_id", keep="last").reset_index(drop=True)
