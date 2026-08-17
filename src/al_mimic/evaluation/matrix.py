"""Tabular evaluation matrix for comparing experiment outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable


def _flatten_metrics(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, dict):
        return {prefix.rstrip("."): value}
    flattened: dict[str, Any] = {}
    for key, child in value.items():
        name = f"{prefix}{key}"
        if isinstance(child, dict):
            flattened.update(_flatten_metrics(child, f"{name}."))
        elif isinstance(child, (int, float, str, bool)) or child is None:
            flattened[name] = child
    return flattened


def load_result(experiment: str | Path) -> dict[str, Any]:
    path = Path(experiment)
    if path.is_dir():
        path = path / "final_metrics.json"
    if not path.is_file():
        raise FileNotFoundError(f"experiment metrics not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"experiment metrics must be a JSON object: {path}")
    return payload


def build_matrix(experiments: Iterable[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for experiment in experiments:
        path = Path(experiment)
        result = load_result(path)
        row = {"experiment": path.parent.name if path.name == "final_metrics.json" else path.name}
        row.update(_flatten_metrics(result))
        rows.append(row)
    columns = sorted({key for row in rows for key in row})
    return [{key: row.get(key) for key in columns} for row in rows]


def write_matrix(rows: list[dict[str, Any]], output: str | Path) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".json":
        path.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return path
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return path
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


__all__ = ["build_matrix", "load_result", "write_matrix"]
