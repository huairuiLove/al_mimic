"""Reusable late-fusion operations for MoSAIC counterfactuals."""

from __future__ import annotations

import torch
import torch.nn as nn


def augment_bias(features: torch.Tensor) -> torch.Tensor:
    ones = torch.ones((*features.shape[:-1], 1), dtype=features.dtype, device=features.device)
    return torch.cat((features, ones), dim=-1)


@torch.inference_mode()
def fuse_token_batches(
    classifier: nn.Module,
    tokens: torch.Tensor,
    *,
    batch_size: int,
) -> torch.Tensor:
    if tokens.ndim < 3:
        raise ValueError("tokens must include sample, modality, and fusion dimensions")
    declared_count = len(getattr(classifier, "modality_names", ()))
    modality_count = declared_count or int(tokens.shape[-2])
    if tokens.ndim < 3 or tokens.shape[-2] != modality_count:
        raise ValueError(f"tokens must end in [{modality_count}, fusion_dim]")
    leading_shape = tokens.shape[:-2]
    flat = tokens.reshape(-1, modality_count, tokens.shape[-1])
    fused = torch.empty(flat.shape[0], classifier.feature_dim, dtype=torch.float32, device=flat.device)
    classifier.eval()
    for start in range(0, int(flat.shape[0]), max(1, int(batch_size))):
        stop = min(start + int(batch_size), int(flat.shape[0]))
        fused[start:stop] = classifier.fuse_from_tokens(flat[start:stop]).float()
    return fused.reshape(*leading_shape, classifier.feature_dim)


@torch.inference_mode()
def probabilities_from_fused(classifier: nn.Module, fused: torch.Tensor) -> torch.Tensor:
    method = getattr(classifier, "probabilities_from_fused", None)
    if callable(method):
        return method(fused).float()
    return torch.sigmoid(classifier.classifier(fused)).float()
