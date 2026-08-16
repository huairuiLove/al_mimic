"""Lazy acquisition-method registry.

Registry values are import paths, so optional method dependencies are imported
only after that method is selected.
"""

from __future__ import annotations

from functools import lru_cache
from importlib import import_module
from types import MappingProxyType
from typing import Any

_METHOD_PATHS = {
    "random": "al_mimic.methods.random:PLUGIN",
    "comal": "al_mimic.methods.comal:PLUGIN",
    "mm_comal": "al_mimic.methods.mm_comal:PLUGIN",
    "modis": "al_mimic.methods.modis:PLUGIN",
    "mosaic": "al_mimic.methods.mosaic:PLUGIN",
}

METHODS = MappingProxyType(_METHOD_PATHS)
METHOD_IMPORTS = METHODS


class MethodLoadError(RuntimeError):
    """Raised when a selected method plugin cannot be imported."""


def normalize_method_name(name: str) -> str:
    key = str(name).strip().lower().replace("-", "_")
    if not key:
        raise ValueError("method name cannot be empty")
    return key


def available_methods() -> tuple[str, ...]:
    """Return registered names without importing method implementations."""
    return tuple(sorted(METHOD_IMPORTS))


@lru_cache(maxsize=None)
def get_method(name: str) -> Any:
    """Load and return one plugin object by normalized method name."""
    key = normalize_method_name(name)
    try:
        import_path = METHOD_IMPORTS[key]
    except KeyError as exc:
        choices = ", ".join(available_methods())
        raise ValueError(f"unknown acquisition method {name!r}; choose from {choices}") from exc
    module_name, separator, attribute_name = import_path.partition(":")
    if not separator or not module_name or not attribute_name:
        raise MethodLoadError(f"invalid method import path for {key!r}: {import_path!r}")
    try:
        plugin = getattr(import_module(module_name), attribute_name)
    except (ImportError, AttributeError) as exc:
        raise MethodLoadError(
            f"could not load acquisition method {key!r} from {import_path!r}: {exc}"
        ) from exc
    if not callable(getattr(plugin, "acquire", None)):
        raise MethodLoadError(f"method plugin {import_path!r} does not define acquire(context)")
    return plugin


load_method = get_method


__all__ = [
    "METHODS",
    "METHOD_IMPORTS",
    "MethodLoadError",
    "available_methods",
    "get_method",
    "load_method",
    "normalize_method_name",
]
