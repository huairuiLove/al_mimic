"""CoMAL-compatible classifier and label-wise contrastive module."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextMLPClassifier(nn.Module):
    """Three-linear-layer head trained on cached note embeddings."""

    def __init__(self, input_dim: int, num_labels: int, hidden_dims: tuple[int, int], dropout: float) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dims[1], num_labels)
        self.feature_dim = hidden_dims[1]

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        fused = self.backbone(features.float())
        return {"logits": self.classifier(fused), "features": fused}


class CoMALModule(nn.Module):
    """Label-wise latent reconstruction module used by CoMAL.

    This follows the original ``MLP_VAE`` topology while accepting the adapter's
    cached text features.  Positive labels have one prototype per ICD group and
    all negatives share a background prototype, matching ``cl_neg_mode=1``.
    """

    def __init__(self, input_dim: int, num_labels: int, label_dim: int = 64, prototype_dim: int = 64) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.prototype_dim = prototype_dim
        self.to_label = nn.Linear(input_dim, num_labels * label_dim)
        self.to_latent = nn.Linear(label_dim, prototype_dim)
        self.from_latent = nn.Linear(prototype_dim, label_dim)
        self.aggregate = nn.Linear(num_labels * label_dim, input_dim)
        self.reconstruction_classifier = nn.Linear(input_dim, num_labels)
        self.register_buffer("prototypes", torch.zeros(num_labels + 1, prototype_dim))
        self.register_buffer("prototype_counts", torch.zeros(num_labels + 1))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in (
            self.to_label,
            self.to_latent,
            self.from_latent,
            self.aggregate,
            self.reconstruction_classifier,
        ):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        batch = features.shape[0]
        label_features = self.to_label(features).view(batch, self.num_labels, -1)
        latent = self.to_latent(label_features)
        decoded = self.from_latent(latent).reshape(batch, -1)
        reconstructed = self.aggregate(decoded)
        reconstructed_logits = self.reconstruction_classifier(reconstructed)
        similarities = F.normalize(latent, dim=-1) @ self.prototypes.T
        return {
            "latent_features": latent,
            "reconstructed_features": reconstructed,
            "reconstructed_logits": reconstructed_logits,
            "prototype_similarities": similarities,
        }

    @torch.no_grad()
    def set_prototypes(self, sums: torch.Tensor, counts: torch.Tensor) -> None:
        normalized = F.normalize(sums, dim=-1)
        self.prototypes.copy_(torch.where(counts[:, None] > 0, normalized, self.prototypes))
        self.prototype_counts.copy_(counts)


def supervised_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float = 0.07,
    anchor_chunk_size: int = 1024,
) -> torch.Tensor:
    """Memory-bounded CoMAL positive/shared-negative contrastive loss."""
    if features.ndim != 3 or labels.shape != features.shape[:2]:
        raise ValueError("expected latent features [B,L,D] and labels [B,L]")
    batch_size, num_labels, feature_dim = features.shape
    flat = F.normalize(features, dim=-1).reshape(batch_size * num_labels, feature_dim)
    label_ids = torch.arange(num_labels, device=labels.device).expand(batch_size, -1)
    class_ids = torch.where(labels >= 0.5, label_ids, num_labels).reshape(-1)
    total = flat.shape[0]
    losses: list[torch.Tensor] = []
    for start in range(0, total, anchor_chunk_size):
        stop = min(start + anchor_chunk_size, total)
        logits = flat[start:stop] @ flat.T / max(temperature, 1e-6)
        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        positive = class_ids[start:stop, None].eq(class_ids[None, :])
        rows = torch.arange(stop - start, device=features.device)
        columns = torch.arange(start, stop, device=features.device)
        valid = torch.ones_like(positive, dtype=torch.bool)
        valid[rows, columns] = False
        positive &= valid
        log_prob = logits - torch.logsumexp(logits.masked_fill(~valid, -torch.inf), dim=1, keepdim=True)
        counts = positive.sum(dim=1)
        per_anchor = -(log_prob.masked_fill(~positive, 0).sum(dim=1) / counts.clamp_min(1))
        losses.append(per_anchor[counts > 0])
    valid_losses = [value for value in losses if value.numel()]
    return torch.cat(valid_losses).mean() if valid_losses else features.sum() * 0


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


@torch.inference_mode()
def positive_similarity_thresholds(
    latent_features: torch.Tensor,
    labels: torch.Tensor,
    prototypes: torch.Tensor,
) -> torch.Tensor:
    """Midpoint of labeled positive min/max similarity from original CoMAL."""
    own_similarity = torch.einsum(
        "nld,ld->nl",
        F.normalize(latent_features.float(), dim=-1),
        F.normalize(prototypes[: labels.shape[1]].float(), dim=-1),
    )
    thresholds = torch.zeros(labels.shape[1], dtype=own_similarity.dtype, device=own_similarity.device)
    for label_index in range(labels.shape[1]):
        values = own_similarity[labels[:, label_index] >= 0.5, label_index]
        thresholds[label_index] = (values.min() + values.max()) * 0.5 if values.numel() else 0.0
    return thresholds


def paper_comal_acquisition_scores(
    probabilities: torch.Tensor,
    latent_features: torch.Tensor,
    prototypes: torch.Tensor,
    positive_thresholds: torch.Tensor,
    *,
    expected_cardinality: float,
) -> PaperAcquisitionComponents:
    """CoMAL score from ``selection_methods.query_samples`` in the release."""
    own_similarity = torch.einsum(
        "nld,ld->nl",
        F.normalize(latent_features.float(), dim=-1),
        F.normalize(prototypes[: probabilities.shape[1]].float(), dim=-1),
    )
    prototype_positive = own_similarity > positive_thresholds[None, :]
    prototype_positive_count = prototype_positive.sum(dim=1).float()
    cardinality_mismatch = (prototype_positive_count - expected_cardinality).abs()
    classifier_positive = probabilities >= 0.5
    positive_evidence = (classifier_positive.float() * ((own_similarity + 1.0) * 0.5).clamp_min(1e-10)).sum(
        dim=1
    )
    inverse_positive_evidence = (positive_evidence + probabilities.shape[1] * 1e-10).reciprocal()
    combined = inverse_positive_evidence.sqrt() * cardinality_mismatch.sqrt()
    return PaperAcquisitionComponents(
        inverse_positive_evidence,
        cardinality_mismatch,
        prototype_positive_count,
        combined,
    )


def comal_acquisition_scores(
    probabilities: torch.Tensor,
    latent_features: torch.Tensor,
    prototypes: torch.Tensor,
    *,
    expected_cardinality: float,
    uncertainty_weight: float = 0.5,
    prototype_weight: float = 0.35,
    cardinality_weight: float = 0.15,
) -> AcquisitionComponents:
    """Rank uncertain notes whose predicted positives are far from prototypes."""
    uncertainty = (1.0 - (2.0 * probabilities - 1.0).abs()).mean(dim=1)
    normalized = F.normalize(latent_features, dim=-1)
    positive_prototypes = prototypes[: probabilities.shape[1]]
    similarity = torch.einsum("nld,ld->nl", normalized, positive_prototypes)
    predicted_positive = probabilities.ge(0.5)
    fallback = torch.topk(probabilities, k=1, dim=1).indices
    predicted_positive = predicted_positive.scatter(1, fallback, True)
    selected_similarity = (similarity * predicted_positive).sum(dim=1) / predicted_positive.sum(
        dim=1
    ).clamp_min(1)
    prototype_novelty = ((1.0 - selected_similarity) * 0.5).clamp(0, 1)
    predicted_cardinality = probabilities.sum(dim=1)
    cardinality_mismatch = (predicted_cardinality - expected_cardinality).abs() / max(
        float(probabilities.shape[1]), 1.0
    )
    combined = (
        uncertainty_weight * uncertainty
        + prototype_weight * prototype_novelty
        + cardinality_weight * cardinality_mismatch
    )
    return AcquisitionComponents(uncertainty, prototype_novelty, cardinality_mismatch, combined)
