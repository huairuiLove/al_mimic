"""Small, task-agnostic interfaces shared by acquisition plugins."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch

MISSING = object()


def context_field(
    context: Any,
    fields: Mapping[str, Any],
    name: str,
    default: Any = MISSING,
) -> Any:
    """Read a keyword override, mapping entry, or object attribute."""
    if name in fields:
        return fields[name]
    if isinstance(context, Mapping) and name in context:
        return context[name]
    if context is not None and hasattr(context, name):
        return getattr(context, name)
    if default is not MISSING:
        return default
    raise ValueError(f"acquisition context requires {name!r}")


def validate_query_size(query_size: int, candidate_count: int) -> int:
    """Validate an exact without-replacement query budget."""
    size = int(query_size)
    if size < 0:
        raise ValueError("query_size must be non-negative")
    if size > int(candidate_count):
        raise ValueError("query_size cannot exceed the candidate count")
    return size


def position_tuple(positions: Any) -> tuple[int, ...]:
    """Normalize tensor, ndarray, or sequence positions to Python integers."""
    if isinstance(positions, torch.Tensor):
        values = positions.detach().cpu().reshape(-1).tolist()
    elif isinstance(positions, np.ndarray):
        values = positions.reshape(-1).tolist()
    else:
        values = list(positions)
    return tuple(int(value) for value in values)


class PreparedContext(Mapping[str, Any]):
    """Non-mutating mapping and attribute overlay for prepared method inputs."""

    def __init__(self, context: Any, **prepared: Any) -> None:
        self._context = context
        self._prepared = dict(prepared)

    def __getitem__(self, name: str) -> Any:
        if name in self._prepared:
            return self._prepared[name]
        if isinstance(self._context, Mapping):
            return self._context[name]
        if self._context is not None and hasattr(self._context, name):
            return getattr(self._context, name)
        raise KeyError(name)

    def __iter__(self) -> Iterator[str]:
        names = set(self._prepared)
        if isinstance(self._context, Mapping):
            names.update(str(name) for name in self._context)
        elif self._context is not None:
            names.update(vars(self._context))
        return iter(names)

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


@dataclass(frozen=True)
class AcquisitionResult:
    """Uniform return value for all task-independent method plugins."""

    selected_ids: tuple[Any, ...]
    selected_positions: tuple[int, ...]
    scores: dict[str, Any]
    diagnostics: dict[str, Any]


def acquisition_result(
    method_id: str,
    candidate_ids: Sequence[Any],
    positions: Any,
    *,
    scores: Mapping[str, Any] | None = None,
    diagnostics: Mapping[str, Any] | None = None,
) -> AcquisitionResult:
    """Build a validated result while preserving candidate identifier types."""
    normalized_positions = position_tuple(positions)
    count = len(candidate_ids)
    if len(set(normalized_positions)) != len(normalized_positions):
        raise ValueError("selected positions must be unique")
    if any(position < 0 or position >= count for position in normalized_positions):
        raise ValueError("selected position is outside the candidate pool")
    details = dict(diagnostics or {})
    details.setdefault("method", str(method_id))
    details.setdefault("candidate_count", count)
    details.setdefault("selected_count", len(normalized_positions))
    return AcquisitionResult(
        selected_ids=tuple(candidate_ids[position] for position in normalized_positions),
        selected_positions=normalized_positions,
        scores=dict(scores or {}),
        diagnostics=details,
    )


__all__ = [
    "AcquisitionResult",
    "MISSING",
    "PreparedContext",
    "acquisition_result",
    "context_field",
    "position_tuple",
    "validate_query_size",
]
