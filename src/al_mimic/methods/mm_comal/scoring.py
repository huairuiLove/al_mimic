"""Reliability-weighted multimodal CoMAL acquisition scoring."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class MMCoMALStatistics:
    reliability: torch.Tensor
    weights: torch.Tensor
    thresholds: torch.Tensor
    positive_counts: torch.Tensor
    included_views: torch.Tensor


@dataclass(frozen=True)
class MMCoMALAcquisitionComponents:
    inverse_positive_evidence: torch.Tensor
    cardinality_mismatch: torch.Tensor
    prototype_positive_count: torch.Tensor
    dispersion: torch.Tensor
    base_score: torch.Tensor
    combined: torch.Tensor


def _masked_mean(values: torch.Tensor, mask: torch.Tensor, *, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    weights = mask.expand_as(values).to(dtype=values.dtype)
    count = weights.sum(dim=dim)
    return (values * weights).sum(dim=dim) / count.clamp_min(1.0), count


@torch.inference_mode()
def estimate_mm_comal_statistics(
    own_similarity: torch.Tensor,
    labels: torch.Tensor,
    *,
    reliability_shrinkage: float = 10.0,
    threshold_shrinkage: float = 10.0,
    threshold_estimator: str = "shrunk",
    include_fused_in_weights: bool = False,
    equal_weights: bool = False,
) -> MMCoMALStatistics:
    """Estimate per-view reliability weights and positive-region thresholds."""
    if own_similarity.ndim != 3:
        raise ValueError("own_similarity must have shape [N,V,L]")
    if labels.shape != (own_similarity.shape[0], own_similarity.shape[2]):
        raise ValueError("labels must have shape [N,L]")
    estimator = str(threshold_estimator).lower()
    if estimator not in {"shrunk", "midpoint"}:
        raise ValueError("threshold_estimator must be shrunk or midpoint")

    similarities = own_similarity.float()
    evidence = ((similarities + 1.0) * 0.5).clamp(0.0, 1.0)
    positive = labels[:, None, :] >= 0.5
    negative = ~positive
    positive_mean, positive_counts = _masked_mean(evidence, positive, dim=0)
    negative_mean, _ = _masked_mean(evidence, negative, dim=0)
    reliability = (positive_mean - negative_mean).clamp_min(0.0)

    label_positive_counts = positive_counts[0]
    pooled_denominator = label_positive_counts.sum().clamp_min(1.0)
    pooled_reliability = (reliability * label_positive_counts[None, :]).sum(dim=1) / pooled_denominator
    reliability_strength = max(float(reliability_shrinkage), 0.0)
    shrunk_reliability = (
        label_positive_counts[None, :] * reliability + reliability_strength * pooled_reliability[:, None]
    ) / (label_positive_counts[None, :] + reliability_strength).clamp_min(1.0)

    num_views = int(similarities.shape[1])
    included_count = num_views if include_fused_in_weights or num_views == 1 else num_views - 1
    included_views = torch.arange(included_count, device=similarities.device)
    weights = torch.zeros_like(shrunk_reliability)
    selected = shrunk_reliability.index_select(0, included_views)
    if equal_weights:
        weights[included_views] = 1.0 / max(included_count, 1)
    else:
        denominators = selected.sum(dim=0, keepdim=True)
        uniform = torch.full_like(selected, 1.0 / max(included_count, 1))
        weights[included_views] = torch.where(
            denominators > 1e-12,
            selected / denominators.clamp_min(1e-12),
            uniform,
        )

    if estimator == "midpoint":
        large = torch.finfo(similarities.dtype).max
        minima = similarities.masked_fill(~positive, large).min(dim=0).values
        maxima = similarities.masked_fill(~positive, -large).max(dim=0).values
        thresholds = torch.where(positive_counts > 0, (minima + maxima) * 0.5, 0.0)
    else:
        positive_similarity_mean, _ = _masked_mean(similarities, positive, dim=0)
        negative_similarity_mean, _ = _masked_mean(similarities, negative, dim=0)
        raw_thresholds = (positive_similarity_mean + negative_similarity_mean) * 0.5
        pooled_thresholds = (raw_thresholds * label_positive_counts[None, :]).sum(dim=1) / pooled_denominator
        threshold_strength = max(float(threshold_shrinkage), 0.0)
        thresholds = (
            label_positive_counts[None, :] * raw_thresholds + threshold_strength * pooled_thresholds[:, None]
        ) / (label_positive_counts[None, :] + threshold_strength).clamp_min(1.0)

    return MMCoMALStatistics(
        reliability=shrunk_reliability,
        weights=weights,
        thresholds=thresholds,
        positive_counts=label_positive_counts,
        included_views=included_views,
    )


@torch.inference_mode()
def mm_comal_acquisition_scores(
    probabilities: torch.Tensor,
    own_similarity: torch.Tensor,
    statistics: MMCoMALStatistics,
    *,
    expected_cardinality: float | torch.Tensor,
    alpha: float = 1.0,
    dispersion: str = "weighted_mad",
) -> MMCoMALAcquisitionComponents:
    """Score aggregate prototype evidence and cross-view dispersion."""
    if own_similarity.ndim != 3:
        raise ValueError("own_similarity must have shape [N,V,L]")
    if probabilities.shape != (own_similarity.shape[0], own_similarity.shape[2]):
        raise ValueError("probabilities must have shape [N,L]")
    if statistics.weights.shape != own_similarity.shape[1:]:
        raise ValueError("MM-CoMAL statistics do not match candidate views")
    dispersion_name = str(dispersion).lower()
    if dispersion_name not in {"weighted_mad", "range", "std"}:
        raise ValueError("dispersion must be weighted_mad, range, or std")

    evidence_by_view = ((own_similarity.float() + 1.0) * 0.5).clamp(0.0, 1.0)
    weights = statistics.weights
    alpha_value = float(alpha)
    if alpha_value == 0.0:
        aggregate_evidence = evidence_by_view[:, -1]
        aggregate_thresholds = statistics.thresholds[-1]
    else:
        aggregate_evidence = (evidence_by_view * weights[None, :, :]).sum(dim=1)
        aggregate_thresholds = (statistics.thresholds * weights).sum(dim=0)

    prototype_positive = aggregate_evidence > ((aggregate_thresholds + 1.0) * 0.5)[None, :]
    prototype_positive_count = prototype_positive.sum(dim=1).float()
    cardinality_mismatch = (prototype_positive_count - expected_cardinality).abs()
    classifier_positive = probabilities.float() >= 0.5
    positive_evidence = (classifier_positive.float() * aggregate_evidence.clamp_min(1e-10)).sum(dim=1)
    inverse_positive_evidence = (positive_evidence + probabilities.shape[1] * 1e-10).reciprocal()
    base_score = inverse_positive_evidence.sqrt() * cardinality_mismatch.sqrt()

    selected_evidence = evidence_by_view.index_select(1, statistics.included_views)
    selected_weights = weights.index_select(0, statistics.included_views)
    if dispersion_name == "weighted_mad":
        per_label_dispersion = (
            selected_weights[None, :, :] * (selected_evidence - aggregate_evidence[:, None, :]).abs()
        ).sum(dim=1)
    elif dispersion_name == "range":
        per_label_dispersion = selected_evidence.amax(dim=1) - selected_evidence.amin(dim=1)
    else:
        variance = (
            selected_weights[None, :, :] * (selected_evidence - aggregate_evidence[:, None, :]).square()
        ).sum(dim=1)
        per_label_dispersion = variance.clamp_min(0.0).sqrt()
    positive_count = classifier_positive.sum(dim=1)
    sample_dispersion = (per_label_dispersion * classifier_positive.float()).sum(
        dim=1
    ) / positive_count.clamp_min(1)
    sample_dispersion = torch.where(
        positive_count > 0, sample_dispersion, torch.zeros_like(sample_dispersion)
    )
    combined = base_score * (1.0 + alpha_value * sample_dispersion)
    return MMCoMALAcquisitionComponents(
        inverse_positive_evidence=inverse_positive_evidence,
        cardinality_mismatch=cardinality_mismatch,
        prototype_positive_count=prototype_positive_count,
        dispersion=sample_dispersion,
        base_score=base_score,
        combined=combined,
    )


__all__ = [
    "MMCoMALAcquisitionComponents",
    "MMCoMALStatistics",
    "estimate_mm_comal_statistics",
    "mm_comal_acquisition_scores",
]
