"""Task-agnostic late-fusion operations for modality-token models."""

from __future__ import annotations

import torch
import torch.nn as nn


@torch.inference_mode()
def fuse_token_batches(
    classifier: nn.Module,
    tokens: torch.Tensor,
    *,
    batch_size: int,
) -> torch.Tensor:
    """Fuse arbitrarily nested token batches while bounding model batch size."""
    if tokens.ndim < 3:
        raise ValueError("tokens must include sample, modality, and fusion dimensions")
    fusion = getattr(classifier, "fuse_from_tokens", None)
    if not callable(fusion):
        raise ValueError("classifier must define callable fuse_from_tokens(tokens)")
    feature_dim = getattr(classifier, "feature_dim", None)
    if not isinstance(feature_dim, int) or feature_dim < 1:
        raise ValueError("classifier.feature_dim must be a positive integer")
    declared_count = len(getattr(classifier, "modality_names", ()))
    modality_count = declared_count or int(tokens.shape[-2])
    if tokens.shape[-2] != modality_count:
        raise ValueError(f"tokens must end in [{modality_count}, fusion_dim]")

    leading_shape = tokens.shape[:-2]
    flat = tokens.reshape(-1, modality_count, tokens.shape[-1])
    fused = torch.empty(
        flat.shape[0],
        feature_dim,
        dtype=torch.float32,
        device=flat.device,
    )
    was_training = classifier.training
    classifier.eval()
    step = max(1, int(batch_size))
    try:
        for start in range(0, int(flat.shape[0]), step):
            stop = min(start + step, int(flat.shape[0]))
            fused[start:stop] = fusion(flat[start:stop]).float()
    finally:
        classifier.train(was_training)
    return fused.reshape(*leading_shape, feature_dim)


@torch.inference_mode()
def probabilities_from_fused(classifier: nn.Module, fused: torch.Tensor) -> torch.Tensor:
    """Evaluate probabilities through an explicit fusion API or sigmoid head."""
    method = getattr(classifier, "probabilities_from_fused", None)
    if callable(method):
        return method(fused).float()
    head = getattr(classifier, "classifier", None)
    if not callable(head):
        raise ValueError(
            "classifier must define probabilities_from_fused(fused) or a callable classifier head"
        )
    return torch.sigmoid(head(fused)).float()


__all__ = ["fuse_token_batches", "probabilities_from_fused"]
