"""Configuration inheritance and path resolution shared by task plugins."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_inherited_yaml(path: str | Path) -> dict[str, Any]:
    """Load one YAML mapping and recursively merge its declared parents."""
    config_path = Path(path).expanduser().resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("configuration root must be a mapping")
    parents = payload.pop("extends", None) or []
    if isinstance(parents, (str, Path)):
        parents = [parents]
    inherited: dict[str, Any] = {}
    for entry in parents:
        parent = Path(entry).expanduser()
        if not parent.is_absolute():
            parent = config_path.parent / parent
        inherited = deep_merge(inherited, load_inherited_yaml(parent))
    inherited.pop("_config_path", None)
    config = deep_merge(inherited, payload)
    config["_config_path"] = str(config_path)
    return config


def resolve_config_path(config: dict[str, Any], value: str | Path) -> Path:
    """Resolve YAML paths relative to the final experiment configuration."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    config_path = Path(config.get("_config_path", ".")).expanduser().resolve()
    return (config_path.parent / path).resolve()


__all__ = ["deep_merge", "load_inherited_yaml", "resolve_config_path"]
