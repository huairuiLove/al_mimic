"""Task and method capability validation without concrete imports."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .contracts import TaskCapabilities


class CapabilityError(ValueError):
    """Raised before a task/method/action combination starts."""


def task_capabilities(plugin: Any) -> TaskCapabilities:
    actions = frozenset(str(value) for value in getattr(plugin, "actions", ()))
    features = frozenset(str(value) for value in getattr(plugin, "capabilities", ()))
    methods = frozenset(str(value) for value in getattr(plugin, "supported_methods", ()))
    return TaskCapabilities(
        task_id=str(getattr(plugin, "task_id")),
        display_name=str(getattr(plugin, "display_name", getattr(plugin, "task_id", ""))),
        actions=actions,
        features=features,
        methods=methods,
        query_unit=getattr(plugin, "query_unit", None),
        supervised_only=bool(getattr(plugin, "supervised_only", False)),
    )


def require_action(plugin: Any, action: str) -> None:
    capabilities = task_capabilities(plugin)
    if action not in capabilities.actions:
        supported = ", ".join(sorted(capabilities.actions)) or "none"
        raise CapabilityError(
            f"task {capabilities.task_id!r} does not support action {action!r}; "
            f"supported actions: {supported}"
        )


def require_method(task_plugin: Any, method_plugin: Any) -> None:
    task = task_capabilities(task_plugin)
    method_id = str(getattr(method_plugin, "method_id", ""))
    if task.supervised_only or not task.methods:
        raise CapabilityError(f"task {task.task_id!r} is supervised-only and accepts no method")
    if method_id not in task.methods:
        supported = ", ".join(sorted(task.methods))
        raise CapabilityError(
            f"task {task.task_id!r} does not support method {method_id!r}; supported methods: {supported}"
        )
    required = frozenset(str(value) for value in getattr(method_plugin, "required_capabilities", ()))
    missing = sorted(required - task.features)
    if missing:
        raise CapabilityError(
            f"task {task.task_id!r} cannot run method {method_id!r}; missing capabilities: {missing}"
        )


def capability_record(plugin: Any) -> dict[str, Any]:
    record = asdict(task_capabilities(plugin))
    for key in ("actions", "features", "methods"):
        record[key] = sorted(record[key])
    return record


__all__ = [
    "CapabilityError",
    "capability_record",
    "require_action",
    "require_method",
    "task_capabilities",
]
