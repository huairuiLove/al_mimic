"""Configuration loading and validation for MIMIC-III experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a mapping")
    parent = config.pop("extends", None)
    if parent:
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = path.parent / parent_path
        config = _merge(load_config(parent_path), config)
    config["_config_path"] = str(path.resolve())
    return config


def require_paths(config: dict[str, Any]) -> dict[str, Path]:
    dataset = config.get("dataset", {})
    root = Path(dataset.get("root", "mimic-iii-clinical-database-1.4"))

    def table_path(config_key: str, table_name: str) -> Path:
        explicit = dataset.get(config_key)
        if explicit:
            return Path(explicit)
        compressed = root / f"{table_name}.csv.gz"
        return compressed if compressed.is_file() else root / f"{table_name}.csv"

    paths = {
        "root": root,
        "admissions": table_path("admissions", "ADMISSIONS"),
        "diagnoses": table_path("diagnoses", "DIAGNOSES_ICD"),
        "notes": table_path("notes", "NOTEEVENTS"),
    }
    missing = [name for name, value in paths.items() if name != "root" and not value.is_file()]
    if missing:
        raise FileNotFoundError("missing MIMIC-III files: " + ", ".join(missing))
    return paths


def resolve_path(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value)
    config_path = Path(config.get("_config_path", ".")).resolve()
    if not path.is_absolute() and config_path.parent.exists() and str(path).startswith("./"):
        return (config_path.parent / path).resolve()
    return path
