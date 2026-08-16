"""Stable contracts shared by tasks, methods, and orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Hashable, Mapping, Protocol, Sequence, runtime_checkable

QueryId = Hashable


@dataclass(frozen=True, slots=True)
class TaskCapabilities:
    """Features and actions a task exposes to repository orchestration."""

    task_id: str
    display_name: str
    actions: frozenset[str]
    features: frozenset[str] = frozenset()
    methods: frozenset[str] = frozenset()
    query_unit: str | None = None
    supervised_only: bool = False


@dataclass(slots=True)
class AcquisitionContext:
    """Canonical query-level data passed to one acquisition method."""

    candidate_ids: Sequence[QueryId]
    query_size: int
    config: Mapping[str, Any]
    classifier: Any = None
    candidate_outputs: Mapping[str, Any] | None = None
    labeled_outputs: Mapping[str, Any] | None = None
    reference_outputs: Mapping[str, Any] | None = None
    method_state: Any = None
    groups: Sequence[Hashable] | None = None
    initial_prevalence: Any = None
    seed: int = 0
    round_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class MethodPlugin(Protocol):
    method_id: str
    display_name: str
    required_capabilities: tuple[str, ...]
    required_context_fields: tuple[str, ...]

    def acquire(self, context: Any = None, **fields: Any) -> Any: ...


@runtime_checkable
class TaskPlugin(Protocol):
    task_id: str
    display_name: str

    def load_config(self, path: str | Path) -> dict[str, Any]: ...


__all__ = [
    "AcquisitionContext",
    "MethodPlugin",
    "QueryId",
    "TaskCapabilities",
    "TaskPlugin",
]
