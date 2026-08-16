"""Method lifecycle helpers used by every active-learning task."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from al_mimic.methods.api import AcquisitionResult


class MethodContractError(RuntimeError):
    """Raised when a plugin violates the common lifecycle contract."""


def context_value(context: Any, name: str, default: Any = None) -> Any:
    if isinstance(context, Mapping):
        return context.get(name, default)
    return getattr(context, name, default)


def set_context_value(context: Any, name: str, value: Any) -> None:
    if isinstance(context, dict):
        context[name] = value
        return
    setattr(context, name, value)


def fit_method(plugin: Any, context: Any) -> Any:
    hook = getattr(plugin, "fit", None)
    if hook is None:
        return None
    if not callable(hook):
        raise MethodContractError("method fit hook must be callable")
    return hook(context)


def prepare_method_context(plugin: Any, context: Any, state: Any) -> Any:
    set_context_value(context, "method_state", state)
    set_context_value(context, "probe_state", state)
    hook = getattr(plugin, "prepare_context", None)
    prepared = hook(context, state) if callable(hook) else context
    required = tuple(getattr(plugin, "required_context_fields", ()))
    missing = [name for name in required if context_value(prepared, name) is None]
    if missing:
        method_id = str(getattr(plugin, "method_id", "unknown"))
        raise MethodContractError(f"method {method_id!r} requires unresolved context fields: {missing}")
    return prepared


def acquire_method(plugin: Any, context: Any) -> AcquisitionResult:
    result = plugin.acquire(context)
    if not isinstance(result, AcquisitionResult):
        raise MethodContractError(
            "method acquire(context) must return al_mimic.methods.api.AcquisitionResult"
        )
    expected = int(context_value(context, "query_size", -1))
    if len(result.selected_ids) != expected or len(result.selected_positions) != expected:
        raise MethodContractError("method did not return the exact query budget")
    if len(set(result.selected_positions)) != expected:
        raise MethodContractError("method selected positions must be unique")
    return result


def fit_prepare_acquire(plugin: Any, context: Any) -> tuple[Any, AcquisitionResult]:
    state = fit_method(plugin, context)
    prepared = prepare_method_context(plugin, context, state)
    return state, acquire_method(plugin, prepared)


__all__ = [
    "MethodContractError",
    "acquire_method",
    "context_value",
    "fit_method",
    "fit_prepare_acquire",
    "prepare_method_context",
    "set_context_value",
]
