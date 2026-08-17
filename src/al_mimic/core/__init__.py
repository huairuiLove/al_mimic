"""Framework contracts and orchestration shared by all first-party plugins."""

from .capabilities import CapabilityError, require_action, require_method
from .contracts import AcquisitionContext, TaskCapabilities
from .methods import acquire_method, fit_method, prepare_method_context

__all__ = [
    "AcquisitionContext",
    "CapabilityError",
    "TaskCapabilities",
    "acquire_method",
    "fit_method",
    "prepare_method_context",
    "require_action",
    "require_method",
]
