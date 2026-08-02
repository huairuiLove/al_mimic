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
        drop: nn.Module = nn.Identity() if float(dropout) <= 0.0 else nn.Dropout(dropout)
        gelu = nn.GELU(approximate="tanh")
        self.backbone = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dims[0]),
            gelu,
            drop,
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.GELU(approximate="tanh"),
            nn.Identity() if float(dropout) <= 0.0 else nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_dims[1], num_labels)
        self.feature_dim = hidden_dims[1]

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        # Resident caches are already float32; skip a redundant cast on the hot path.
        inputs = features if features.dtype == torch.float32 else features.float()
        fused = self.backbone(inputs)
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

    def forward(
        self,
        features: torch.Tensor,
        *,
        compute_similarities: bool | str = True,
        compute_reconstruction: bool = True,
    ) -> dict[str, torch.Tensor]:
        batch = features.shape[0]
        label_features = self.to_label(features).view(batch, self.num_labels, -1)
        latent = self.to_latent(label_features)
        result: dict[str, torch.Tensor] = {"latent_features": latent}
        # Train needs the decoder; predict / prototype refresh only need latents (+ optional sims).
        if compute_reconstruction:
            decoded = self.from_latent(latent).view(batch, -1)
            reconstructed = self.aggregate(decoded)
            result["reconstructed_features"] = reconstructed
            result["reconstructed_logits"] = self.reconstruction_classifier(reconstructed)
        # similarity modes: True/"full" -> [N,L,L+1]; "own_bg" -> [N,L,2]; False/"none" -> omit.
        if compute_similarities is True or compute_similarities == "full":
            result["prototype_similarities"] = F.normalize(latent, dim=-1) @ self.prototypes.T
        elif compute_similarities == "own_bg":
            normalized = F.normalize(latent, dim=-1)
            own = torch.einsum("nld,ld->nl", normalized, self.prototypes[:-1])
            background = torch.einsum("nld,d->nl", normalized, self.prototypes[-1])
            result["prototype_similarities"] = torch.stack((own, background), dim=-1)
        elif compute_similarities not in {False, "none"}:
            raise ValueError("compute_similarities must be True/'full', 'own_bg', or False/'none'")
        return result

    @torch.no_grad()
    def set_prototypes(self, sums: torch.Tensor, counts: torch.Tensor) -> None:
        normalized = F.normalize(sums, dim=-1)
        self.prototypes.copy_(torch.where(counts[:, None] > 0, normalized, self.prototypes))
        self.prototype_counts.copy_(counts)


def _contrastive_chunk(
    flat: torch.Tensor,
    class_ids: torch.Tensor,
    start: int,
    stop: int,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    chunk = flat[start:stop]
    width = stop - start
    logits = chunk.matmul(flat.transpose(0, 1)).div_(temperature)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    positive = class_ids[start:stop, None].eq(class_ids[None, :])
    eye = torch.arange(width, device=flat.device)
    self_columns = start + eye
    # Exclude self from positives; mask only the softmax denominator.
    # Do not store -inf in logits that later multiply by a zero positive mask
    # (IEEE (-inf)*0 == NaN).
    if start == 0 and stop == int(flat.shape[0]):
        # Square full-batch path: fill_diagonal_ is cheaper than a dense bool mask.
        # Clone only denom logits; mutating the autograd-bearing logits breaks backward.
        positive.fill_diagonal_(False)
        denom_logits = logits.clone()
        denom_logits.fill_diagonal_(float("-inf"))
        denom = torch.logsumexp(denom_logits, dim=1, keepdim=True)
    else:
        self_mask = torch.zeros_like(positive)
        self_mask[eye, self_columns] = True
        positive = positive & ~self_mask
        denom = torch.logsumexp(logits.masked_fill(self_mask, float("-inf")), dim=1, keepdim=True)
    log_prob = logits - denom
    pos = positive.to(dtype=log_prob.dtype)
    counts = pos.sum(dim=1)
    per_anchor = -(log_prob * pos).sum(dim=1) / counts.clamp_min(1)
    active = counts > 0
    selected = per_anchor * active.to(dtype=per_anchor.dtype)
    return selected.sum(), active.sum()


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
    batch_size, num_labels, _feature_dim = features.shape
    flat = F.normalize(features, dim=-1).reshape(batch_size * num_labels, features.shape[-1])
    label_ids = torch.arange(num_labels, device=labels.device).expand(batch_size, -1)
    class_ids = torch.where(labels >= 0.5, label_ids, num_labels).reshape(-1)
    total = int(flat.shape[0])
    temperature = max(float(temperature), 1e-6)
    # Prefer one full pairwise GEMM when it fits; chunking is only a memory guard.
    bytes_needed = total * total * flat.element_size()
    max_full_bytes = 2 * 1024**3 if flat.is_cuda else 512 * 1024**2
    if bytes_needed <= max_full_bytes:
        loss_sum, loss_count = _contrastive_chunk(flat, class_ids, 0, total, temperature)
        return torch.where(loss_count > 0, loss_sum / loss_count.clamp_min(1), features.sum() * 0)
    step = max(1, int(anchor_chunk_size))
    loss_sum = flat.new_zeros(())
    loss_count = flat.new_zeros(())
    for start in range(0, total, step):
        stop = min(start + step, total)
        part_sum, part_count = _contrastive_chunk(flat, class_ids, start, stop, temperature)
        loss_sum = loss_sum + part_sum
        loss_count = loss_count + part_count
    return torch.where(loss_count > 0, loss_sum / loss_count.clamp_min(1), features.sum() * 0)


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


def own_prototype_similarity(
    latent_features: torch.Tensor,
    prototypes: torch.Tensor,
    num_labels: int,
) -> torch.Tensor:
    # Prototypes are unit-normalized in set_prototypes; only normalize latents.
    return torch.einsum(
        "nld,ld->nl",
        F.normalize(latent_features.float(), dim=-1),
        prototypes[:num_labels].float(),
    )


@torch.inference_mode()
def positive_similarity_thresholds(
    latent_features: torch.Tensor | None,
    labels: torch.Tensor,
    prototypes: torch.Tensor,
    *,
    own_similarity: torch.Tensor | None = None,
) -> torch.Tensor:
    """Midpoint of labeled positive min/max similarity from original CoMAL."""
    if own_similarity is None:
        if latent_features is None:
            raise ValueError("latent_features or own_similarity is required")
        own_similarity = own_prototype_similarity(latent_features, prototypes, int(labels.shape[1]))
    positive_mask = labels >= 0.5
    # Vectorized per-label min/max over masked positions; empty labels stay 0.
    large = torch.finfo(own_similarity.dtype).max
    masked_min = own_similarity.masked_fill(~positive_mask, large)
    masked_max = own_similarity.masked_fill(~positive_mask, -large)
    minima = masked_min.min(dim=0).values
    maxima = masked_max.max(dim=0).values
    has_positive = positive_mask.any(dim=0)
    thresholds = torch.zeros(labels.shape[1], dtype=own_similarity.dtype, device=own_similarity.device)
    thresholds = torch.where(has_positive, (minima + maxima) * 0.5, thresholds)
    return thresholds


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
    """CoMAL score from ``selection_methods.query_samples`` in the release."""
    if own_similarity is None:
        if latent_features is None:
            raise ValueError("latent_features or own_similarity is required")
        own_similarity = own_prototype_similarity(
            latent_features, prototypes, int(probabilities.shape[1])
        )
    prototype_positive = own_similarity > positive_thresholds[None, :]
    prototype_positive_count = prototype_positive.sum(dim=1).float()
    # Keep expected_cardinality on-device when passed as a tensor to avoid a mid-pipeline sync.
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
    """Rank uncertain notes whose predicted positives are far from prototypes."""
    uncertainty = (1.0 - (2.0 * probabilities - 1.0).abs()).mean(dim=1)
    if own_similarity is None:
        if latent_features is None:
            raise ValueError("latent_features or own_similarity is required")
        similarity = own_prototype_similarity(latent_features, prototypes, int(probabilities.shape[1]))
    else:
        similarity = own_similarity
    predicted_positive = probabilities.ge(0.5)
    fallback = probabilities.argmax(dim=1, keepdim=True)
    predicted_positive = predicted_positive.scatter(1, fallback, True)
    selected_similarity = (similarity * predicted_positive).sum(dim=1) / predicted_positive.sum(
        dim=1
    ).clamp_min(1)
    prototype_novelty = ((1.0 - selected_similarity) * 0.5).clamp(0, 1)
    predicted_cardinality = probabilities.sum(dim=1)
    label_scale = max(float(probabilities.shape[1]), 1.0)
    cardinality_mismatch = (predicted_cardinality - expected_cardinality).abs() / label_scale
    combined = (
        uncertainty_weight * uncertainty
        + prototype_weight * prototype_novelty
        + cardinality_weight * cardinality_mismatch
    )
    return AcquisitionComponents(uncertainty, prototype_novelty, cardinality_mismatch, combined)
