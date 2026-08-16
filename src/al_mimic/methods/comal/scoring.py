"""CoMAL paper acquisition scoring on label-prototype evidence."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from al_mimic.utils.prototypes import (
    own_prototype_similarity,
)
from al_mimic.utils.prototypes import (
    positive_similarity_thresholds as _positive_similarity_thresholds,
)


@dataclass(frozen=True)
class AcquisitionComponents:
    uncertainty: torch.Tensor
    prototype_novelty: torch.Tensor
    cardinality_mismatch: torch.Tensor
    combined: torch.Tensor


@dataclass(frozen=True)
class PaperAcquisitionComponents:
    inverse_positive_evidence: torch.Tensor
    cardinality_mismatch: torch.Tensor
    prototype_positive_count: torch.Tensor
    combined: torch.Tensor


def _resolve_own_similarity(
    probabilities: torch.Tensor,
    latent_features: torch.Tensor | None,
    prototypes: torch.Tensor,
    own_similarity: torch.Tensor | None,
) -> torch.Tensor:
    similarity = own_similarity
    if similarity is None:
        if latent_features is None:
            raise ValueError("latent_features or own_similarity is required")
        similarity = own_prototype_similarity(latent_features, prototypes, int(probabilities.shape[1]))
    if similarity.shape != probabilities.shape:
        raise ValueError("own_similarity and probabilities must have shape [N,L]")
    return similarity.float()


@torch.inference_mode()
def positive_similarity_thresholds(
    latent_features: torch.Tensor | None,
    labels: torch.Tensor,
    prototypes: torch.Tensor,
    *,
    own_similarity: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compatibility wrapper for the original CoMAL threshold signature."""
    similarity = own_similarity
    if similarity is None:
        if latent_features is None:
            raise ValueError("latent_features or own_similarity is required")
        similarity = own_prototype_similarity(latent_features, prototypes, int(labels.shape[1]))
    return _positive_similarity_thresholds(labels, own_similarity=similarity)


@torch.inference_mode()
def paper_comal_acquisition_scores(
    probabilities: torch.Tensor,
    latent_features: torch.Tensor | None,
    prototypes: torch.Tensor,
    positive_thresholds: torch.Tensor,
    *,
    expected_cardinality: float | torch.Tensor,
    own_similarity: torch.Tensor | None = None,
) -> PaperAcquisitionComponents:
    """Reproduce the released CoMAL acquisition score."""
    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape [N,L]")
    if positive_thresholds.shape != probabilities.shape[1:]:
        raise ValueError("positive_thresholds must have shape [L]")
    similarity = _resolve_own_similarity(probabilities, latent_features, prototypes, own_similarity)
    prototype_positive = similarity > positive_thresholds[None, :]
    prototype_positive_count = prototype_positive.sum(dim=1).float()
    cardinality_mismatch = (prototype_positive_count - expected_cardinality).abs()
    classifier_positive = probabilities.float() >= 0.5
    evidence = ((similarity + 1.0) * 0.5).clamp_min(1e-10)
    positive_evidence = (classifier_positive.float() * evidence).sum(dim=1)
    inverse_positive_evidence = (positive_evidence + probabilities.shape[1] * 1e-10).reciprocal()
    combined = inverse_positive_evidence.sqrt() * cardinality_mismatch.sqrt()
    return PaperAcquisitionComponents(
        inverse_positive_evidence=inverse_positive_evidence,
        cardinality_mismatch=cardinality_mismatch,
        prototype_positive_count=prototype_positive_count,
        combined=combined,
    )


@torch.inference_mode()
def comal_acquisition_scores(
    probabilities: torch.Tensor,
    latent_features: torch.Tensor | None,
    prototypes: torch.Tensor,
    *,
    expected_cardinality: float | torch.Tensor,
    uncertainty_weight: float = 0.5,
    prototype_weight: float = 0.35,
    cardinality_weight: float = 0.15,
    own_similarity: torch.Tensor | None = None,
) -> AcquisitionComponents:
    """Score uncertain samples with novel predicted-positive prototypes."""
    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape [N,L]")
    similarity = _resolve_own_similarity(probabilities, latent_features, prototypes, own_similarity)
    values = probabilities.float()
    uncertainty = (1.0 - (2.0 * values - 1.0).abs()).mean(dim=1)
    predicted_positive = values.ge(0.5)
    fallback = values.argmax(dim=1, keepdim=True)
    predicted_positive = predicted_positive.scatter(1, fallback, True)
    selected_similarity = (similarity * predicted_positive).sum(dim=1) / predicted_positive.sum(
        dim=1
    ).clamp_min(1)
    prototype_novelty = ((1.0 - selected_similarity) * 0.5).clamp(0.0, 1.0)
    cardinality_mismatch = (values.sum(dim=1) - expected_cardinality).abs() / max(float(values.shape[1]), 1.0)
    combined = (
        float(uncertainty_weight) * uncertainty
        + float(prototype_weight) * prototype_novelty
        + float(cardinality_weight) * cardinality_mismatch
    )
    return AcquisitionComponents(
        uncertainty=uncertainty,
        prototype_novelty=prototype_novelty,
        cardinality_mismatch=cardinality_mismatch,
        combined=combined,
    )


__all__ = [
    "AcquisitionComponents",
    "PaperAcquisitionComponents",
    "comal_acquisition_scores",
    "paper_comal_acquisition_scores",
    "positive_similarity_thresholds",
]
