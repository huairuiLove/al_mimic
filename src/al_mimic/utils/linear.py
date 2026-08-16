"""Small linear-algebra helpers shared by acquisition methods."""

from __future__ import annotations

import torch


def augment_bias(features: torch.Tensor) -> torch.Tensor:
    """Append a unit coordinate without changing leading dimensions."""
    if features.ndim < 1:
        raise ValueError("features must have at least one dimension")
    ones = torch.ones(
        (*features.shape[:-1], 1),
        dtype=features.dtype,
        device=features.device,
    )
    return torch.cat((features, ones), dim=-1)


__all__ = ["augment_bias"]
