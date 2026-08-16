"""MM-CoMAL plugin and multimodal prototype scoring."""

from .plugin import PLUGIN, MMCoMALPlugin, MMCoMALState
from .scoring import (
    MMCoMALAcquisitionComponents,
    MMCoMALStatistics,
    estimate_mm_comal_statistics,
    mm_comal_acquisition_scores,
)

__all__ = [
    "MMCoMALAcquisitionComponents",
    "MMCoMALPlugin",
    "MMCoMALState",
    "MMCoMALStatistics",
    "PLUGIN",
    "estimate_mm_comal_statistics",
    "mm_comal_acquisition_scores",
]
