"""Configuration loading and project-relative path resolution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG: Path = PROJECT_ROOT / "config.yaml"


@dataclass(frozen=True)
class Paths:
    """Resolved absolute directories used by the pipeline."""

    raw: Path
    processed: Path
    output: Path
    reports: Path

    @property
    def figures(self) -> Path:
        return self.reports / "figures"

    def ensure(self) -> "Paths":
        """Create all directories if they do not exist and return self."""
        for p in (self.raw, self.processed, self.output, self.reports, self.figures):
            p.mkdir(parents=True, exist_ok=True)
        return self


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML config (defaults to ``config.yaml`` at the project root)."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with open(cfg_path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def get_paths(cfg: dict[str, Any]) -> Paths:
    """Resolve the ``paths`` section relative to the project root."""
    p = cfg["paths"]
    return Paths(
        raw=PROJECT_ROOT / p["raw_dir"],
        processed=PROJECT_ROOT / p["processed_dir"],
        output=PROJECT_ROOT / p["output_dir"],
        reports=PROJECT_ROOT / p["reports_dir"],
    ).ensure()
