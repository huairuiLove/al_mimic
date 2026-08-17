"""Versioned experiment paths and provenance manifests."""

from __future__ import annotations

import platform
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from al_mimic.utils.io import write_json

ARTIFACT_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class ExperimentLayout:
    root: Path
    checkpoints: Path
    rounds: Path
    predictions: Path
    figures: Path
    logs: Path

    @classmethod
    def create(cls, root: str | Path) -> "ExperimentLayout":
        base = Path(root)
        layout = cls(
            root=base,
            checkpoints=base / "checkpoints",
            rounds=base / "rounds",
            predictions=base / "predictions",
            figures=base / "figures",
            logs=base / "logs",
        )
        for directory in asdict(layout).values():
            Path(directory).mkdir(parents=True, exist_ok=True)
        return layout


def repository_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_manifest(
    output_dir: str | Path,
    *,
    task: dict[str, Any],
    method: str | None,
    config: dict[str, Any],
) -> Path:
    return write_json(
        Path(output_dir) / "manifest.json",
        {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "task": task,
            "method": method,
            "repository_revision": repository_revision(),
            "python": platform.python_version(),
            "resolved_config": config,
        },
    )


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "ExperimentLayout",
    "repository_revision",
    "write_manifest",
]
