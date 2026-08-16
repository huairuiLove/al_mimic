"""Lazy registry for first-party dataset task plugins."""

from __future__ import annotations

from functools import lru_cache
from importlib import import_module
from types import MappingProxyType
from typing import Any

_TASK_PATHS = {
    "mimic_iii": "al_mimic.tasks.mimic_iii:PLUGIN",
    "brset": "al_mimic.tasks.brset:PLUGIN",
    "mds_ed": "al_mimic.tasks.mds_ed:PLUGIN",
}

TASKS = MappingProxyType(_TASK_PATHS)


class TaskLoadError(RuntimeError):
    """Raised when a selected first-party task plugin cannot be loaded."""


def normalize_task_name(name: str) -> str:
    key = str(name).strip().lower().replace("-", "_")
    aliases = {
        "mimic3": "mimic_iii",
        "mimic_3": "mimic_iii",
        "mimic_iv_mds_ed": "mds_ed",
        "mimic_iv_mds_ed_diagnoses": "mds_ed",
    }
    return aliases.get(key, key)


def available_tasks() -> tuple[str, ...]:
    return tuple(sorted(TASKS))


@lru_cache(maxsize=None)
def get_task(name: str) -> Any:
    key = normalize_task_name(name)
    try:
        import_path = TASKS[key]
    except KeyError as exc:
        choices = ", ".join(available_tasks())
        raise ValueError(f"unknown task {name!r}; choose from {choices}") from exc
    module_name, separator, attribute_name = import_path.partition(":")
    if not separator:
        raise TaskLoadError(f"invalid task import path for {key!r}: {import_path!r}")
    try:
        plugin = getattr(import_module(module_name), attribute_name)
    except (ImportError, AttributeError) as exc:
        raise TaskLoadError(f"could not load task {key!r} from {import_path!r}: {exc}") from exc
    if str(getattr(plugin, "task_id", "")) not in {key, "mimic_iv_mds_ed_diagnoses"}:
        raise TaskLoadError(f"task plugin {import_path!r} has an invalid task_id")
    return plugin


__all__ = ["TASKS", "TaskLoadError", "available_tasks", "get_task", "normalize_task_name"]
