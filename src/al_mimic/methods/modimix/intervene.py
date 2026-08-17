"""Deterministic on-manifold token interventions for ModiMix."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PrototypeDiagnostics:
    mean_norm_ratio: list[float]
    prototype_norm_ratio: list[float]
    mean_to_medoid_distance_ratio: list[float]
    medoid_indices: list[int]


@torch.inference_mode()
def modality_prototypes(
    tokens: torch.Tensor,
    *,
    method: str = "mean",
    pairwise_sample_size: int = 512,
) -> tuple[torch.Tensor, PrototypeDiagnostics]:
    """Build one labeled-pool prototype per modality and report its typicality."""
    if tokens.ndim != 3:
        raise ValueError("tokens must have shape [N, M, D]")
    if tokens.shape[0] == 0:
        raise ValueError("at least one labeled token is required")
    prototype_method = str(method).lower()
    if prototype_method not in {"mean", "medoid"}:
        raise ValueError("prototype method must be mean or medoid")

    values = tokens.detach().float()
    means = values.mean(dim=0)
    distances_to_mean = torch.linalg.vector_norm(values - means[None, :, :], dim=-1)
    medoid_indices = distances_to_mean.argmin(dim=0)
    modality_indices = torch.arange(values.shape[1], device=values.device)
    medoids = values[medoid_indices, modality_indices]
    prototypes = means if prototype_method == "mean" else medoids

    individual_norms = torch.linalg.vector_norm(values, dim=-1).mean(dim=0).clamp_min(1e-12)
    mean_norm_ratio = torch.linalg.vector_norm(means, dim=-1) / individual_norms
    prototype_norm_ratio = torch.linalg.vector_norm(prototypes, dim=-1) / individual_norms

    sample_count = min(max(2, int(pairwise_sample_size)), int(values.shape[0]))
    if values.shape[0] == 1:
        pairwise_means = values.new_ones(values.shape[1])
    else:
        sampled_indices = torch.linspace(
            0,
            values.shape[0] - 1,
            steps=sample_count,
            device=values.device,
        ).long()
        sampled = values.index_select(0, sampled_indices)
        pairwise_means = torch.stack(
            [torch.pdist(sampled[:, modality]).mean() for modality in range(values.shape[1])]
        ).clamp_min(1e-12)
    mean_to_medoid_ratio = distances_to_mean[medoid_indices, modality_indices] / pairwise_means

    diagnostics = PrototypeDiagnostics(
        mean_norm_ratio=[float(value) for value in mean_norm_ratio],
        prototype_norm_ratio=[float(value) for value in prototype_norm_ratio],
        mean_to_medoid_distance_ratio=[float(value) for value in mean_to_medoid_ratio],
        medoid_indices=[int(value) for value in medoid_indices],
    )
    return prototypes, diagnostics


def interpolate_modality_token(
    tokens: torch.Tensor,
    prototypes: torch.Tensor,
    modality: int,
    alpha: float | torch.Tensor,
) -> torch.Tensor:
    """Move one modality token toward its labeled-pool prototype."""
    if tokens.ndim != 3:
        raise ValueError("tokens must have shape [N, M, D]")
    if prototypes.shape != tokens.shape[1:]:
        raise ValueError("prototypes must have shape [M, D]")
    if not 0 <= int(modality) < tokens.shape[1]:
        raise ValueError("modality index is out of range")
    mixing = torch.as_tensor(alpha, dtype=tokens.dtype, device=tokens.device)
    if mixing.ndim == 1:
        mixing = mixing[:, None]
    result = tokens.clone()
    result[:, modality] = mixing * tokens[:, modality] + (1.0 - mixing) * prototypes[modality]
    return result
