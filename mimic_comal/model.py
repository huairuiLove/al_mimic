"""Yang-Wu multimodal classifier and active-learning acquisition modules."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


class YangWuGate(nn.Module):
    """Clinical-note-primary multimodal adaptation gate from Yang and Wu."""

    def __init__(
        self,
        text_dim: int,
        time_invariant_dim: int,
        time_series_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.static_weight = nn.Linear(text_dim + time_invariant_dim, 1)
        self.time_series_weight = nn.Linear(text_dim + time_series_dim, 1)
        self.adjustment = nn.Linear(time_invariant_dim + time_series_dim, text_dim)
        self.beta = nn.Parameter(torch.randn(1))
        self.norm = nn.LayerNorm(text_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        text: torch.Tensor,
        time_invariant: torch.Tensor,
        time_series: torch.Tensor,
    ) -> torch.Tensor:
        static_gate = torch.sigmoid(
            self.static_weight(torch.cat((text, time_invariant), dim=-1))
        )
        series_gate = torch.sigmoid(
            self.time_series_weight(torch.cat((text, time_series), dim=-1))
        )
        adjustment = self.adjustment(
            torch.cat((static_gate * time_invariant, series_gate * time_series), dim=-1)
        )
        ratio = text.norm() / adjustment.norm().clamp_min(1e-12)
        alpha = torch.minimum(ratio * self.beta, adjustment.new_ones(()))
        return self.dropout(self.norm(text + alpha * adjustment))


class YangWuBertEncoderClassifier(nn.Module):
    """BertEncoder for 48-hour, 1,042-label MIMIC-III diagnosis prediction.

    The classifier follows the paper architecture. ``modality_tokens`` are an
    exact telescoping decomposition of the frozen model's fused representation;
    they are exposed only for the four active-learning methods.
    """

    modality_names = ("clinical_notes", "time_series", "time_invariant")

    def __init__(
        self,
        clinicalbert_checkpoint: str | None,
        *,
        num_labels: int = 1042,
        time_invariant_dim: int = 97,
        time_invariant_hidden_dim: int = 64,
        time_series_dim: int = 7411,
        time_series_hidden_dim: int = 1024,
        time_series_layers: int = 3,
        time_series_heads: int = 16,
        text_hidden_dim: int = 768,
        dropout: float = 0.1,
        text_encoder: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if time_series_hidden_dim % time_series_heads:
            raise ValueError("time-series hidden dimension must be divisible by attention heads")
        self.text_encoder = text_encoder or self._load_clinicalbert(clinicalbert_checkpoint)
        self.time_invariant_encoder = nn.Linear(
            time_invariant_dim, time_invariant_hidden_dim
        )
        self.time_series_projection = nn.Linear(time_series_dim, time_series_hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=time_series_hidden_dim,
            nhead=time_series_heads,
            dropout=dropout,
        )
        self.time_series_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=time_series_layers
        )
        self.gate = YangWuGate(
            text_hidden_dim,
            time_invariant_hidden_dim,
            time_series_hidden_dim,
            dropout,
        )
        self.classifier = nn.Linear(text_hidden_dim, num_labels)
        self.feature_dim = text_hidden_dim
        self.num_labels = num_labels

    @staticmethod
    def _load_clinicalbert(checkpoint: str | None) -> nn.Module:
        if not checkpoint:
            raise ValueError("a ClinicalBERT checkpoint is required")
        try:
            from transformers import BertConfig, BertModel
        except ImportError as exc:  # pragma: no cover - dependency error is environment-specific
            raise RuntimeError("transformers is required for the Yang-Wu baseline") from exc
        encoder = BertModel(BertConfig())
        try:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        except TypeError:  # pragma: no cover - old torch compatibility
            state = torch.load(checkpoint, map_location="cpu")
        if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
            state = state["state_dict"]
        if not isinstance(state, dict):
            raise ValueError("ClinicalBERT checkpoint must contain a state dictionary")
        model_state = encoder.state_dict()
        compatible = {
            str(key).removeprefix("bert."): value
            for key, value in state.items()
            if str(key).removeprefix("bert.") in model_state
            and model_state[str(key).removeprefix("bert.")].shape == value.shape
        }
        coverage = sum(value.numel() for value in compatible.values()) / sum(
            value.numel() for value in model_state.values()
        )
        if coverage < 0.95:
            raise ValueError(
                f"ClinicalBERT checkpoint covers only {coverage:.1%} of BERT-base parameters"
            )
        model_state.update(compatible)
        encoder.load_state_dict(model_state)
        return encoder

    def _encode_text(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        output = self.text_encoder(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )
        if hasattr(output, "pooler_output") and output.pooler_output is not None:
            return output.pooler_output
        if isinstance(output, (tuple, list)) and len(output) > 1:
            return output[1]
        raise ValueError("ClinicalBERT encoder must return a pooled BERT-base representation")

    def encode_modalities(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
        static = F.relu(self.time_invariant_encoder(batch["time_invariant"].float()))
        series = F.relu(self.time_series_projection(batch["time_series"].float()))
        series = self.time_series_encoder(series.transpose(0, 1))[0]
        text = self._encode_text(
            batch["input_ids"].long(),
            batch["token_type_ids"].long(),
            batch["attention_mask"].long(),
        )
        return text, series, static

    def _decompose(
        self,
        text: torch.Tensor,
        series: torch.Tensor,
        static: torch.Tensor,
        fused: torch.Tensor,
    ) -> torch.Tensor:
        zeros_static = torch.zeros_like(static)
        zeros_series = torch.zeros_like(series)
        text_only = self.gate(text, zeros_static, zeros_series)
        text_static = self.gate(text, static, zeros_series)
        static_contribution = text_static - text_only
        series_contribution = fused - text_static
        return torch.stack((text_only, series_contribution, static_contribution), dim=1)

    def fuse_from_tokens(self, tokens: torch.Tensor, *, apply_dropout: bool = False) -> torch.Tensor:
        del apply_dropout
        if tokens.ndim != 3 or tokens.shape[1:] != (3, self.feature_dim):
            raise ValueError(f"tokens must have shape [N, 3, {self.feature_dim}]")
        return tokens.sum(dim=1)

    def probabilities_from_fused(self, fused: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.classifier(fused))

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        *,
        return_tokens: bool = False,
    ) -> dict[str, torch.Tensor]:
        text, series, static = self.encode_modalities(batch)
        fused = self.gate(text, static, series)
        logits = self.classifier(fused)
        result = {
            "logits": logits,
            "probabilities": torch.sigmoid(logits),
            "features": fused,
        }
        if return_tokens:
            result["modality_tokens"] = self._decompose(text, series, static, fused)
        return result


class CoMALModule(nn.Module):
    """Label-wise latent reconstruction module used by CoMAL.

    This follows the original ``MLP_VAE`` topology while accepting frozen
    BertEncoder features. Positive labels have one prototype per ICD group and
    all negatives share a background prototype, matching ``cl_neg_mode=1``.
    """

    def __init__(
        self,
        input_dim: int,
        num_labels: int,
        label_dim: int = 64,
        prototype_dim: int = 64,
        *,
        num_views: int = 1,
    ) -> None:
        super().__init__()
        self.num_labels = num_labels
        self.prototype_dim = prototype_dim
        self.num_views = int(num_views)
        if self.num_views < 1:
            raise ValueError("num_views must be positive")
        self.to_label = nn.Linear(input_dim, num_labels * label_dim)
        self.to_latent = nn.Linear(label_dim, prototype_dim)
        self.from_latent = nn.Linear(prototype_dim, label_dim)
        self.aggregate = nn.Linear(num_labels * label_dim, input_dim)
        self.reconstruction_classifier = nn.Linear(input_dim, num_labels)
        if self.num_views == 1:
            self.register_parameter("view_code", None)
            self.register_buffer("prototypes", torch.zeros(num_labels + 1, prototype_dim))
            self.register_buffer("prototype_counts", torch.zeros(num_labels + 1))
        else:
            self.view_code = nn.Parameter(torch.zeros(self.num_views, input_dim))
            self.register_buffer("prototypes", torch.zeros(self.num_views, num_labels + 1, prototype_dim))
            self.register_buffer("prototype_counts", torch.zeros(self.num_views, num_labels + 1))
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
        if features.ndim == 2:
            view_index = self.num_views - 1
            inputs = features if self.view_code is None else features + self.view_code[view_index]
            label_features = self.to_label(inputs).view(batch, self.num_labels, -1)
            latent = self.to_latent(label_features)
            prototype_bank = self.prototypes if self.num_views == 1 else self.prototypes[view_index]
        elif features.ndim == 3 and self.num_views > 1:
            if features.shape[1] != self.num_views:
                raise ValueError(f"expected {self.num_views} views, got {features.shape[1]}")
            inputs = features + self.view_code[None, :, :]
            label_features = self.to_label(inputs).view(batch, self.num_views, self.num_labels, -1)
            latent = self.to_latent(label_features)
            prototype_bank = self.prototypes
        else:
            raise ValueError("CoMAL features must have shape [N,D] or [N,V,D]")
        result: dict[str, torch.Tensor] = {"latent_features": latent}
        # Train needs the decoder; predict / prototype refresh only need latents (+ optional sims).
        if compute_reconstruction:
            reconstruction_latent = latent if latent.ndim == 3 else latent[:, -1]
            decoded = self.from_latent(reconstruction_latent).view(batch, -1)
            reconstructed = self.aggregate(decoded)
            result["reconstructed_features"] = reconstructed
            result["reconstructed_logits"] = self.reconstruction_classifier(reconstructed)
        # similarity modes: True/"full" -> [N,L,L+1]; "own_bg" -> [N,L,2]; False/"none" -> omit.
        if compute_similarities is True or compute_similarities == "full":
            normalized = F.normalize(latent, dim=-1)
            if normalized.ndim == 3:
                result["prototype_similarities"] = normalized @ prototype_bank.T
            else:
                result["prototype_similarities"] = torch.einsum(
                    "nvld,vkd->nvlk", normalized, prototype_bank
                )
        elif compute_similarities == "own_bg":
            normalized = F.normalize(latent, dim=-1)
            if normalized.ndim == 3:
                own = torch.einsum("nld,ld->nl", normalized, prototype_bank[:-1])
                background = normalized.matmul(prototype_bank[-1])
            else:
                own = torch.einsum("nvld,vld->nvl", normalized, prototype_bank[:, :-1])
                background = torch.einsum("nvld,vd->nvl", normalized, prototype_bank[:, -1])
            result["prototype_similarities"] = torch.stack((own, background), dim=-1)
        elif compute_similarities not in {False, "none"}:
            raise ValueError("compute_similarities must be True/'full', 'own_bg', or False/'none'")
        return result

    @torch.no_grad()
    def set_prototypes(self, sums: torch.Tensor, counts: torch.Tensor) -> None:
        if sums.shape != self.prototypes.shape or counts.shape != self.prototype_counts.shape:
            raise ValueError("prototype sums/counts do not match configured views")
        normalized = F.normalize(sums, dim=-1)
        self.prototypes.copy_(torch.where(counts.unsqueeze(-1) > 0, normalized, self.prototypes))
        self.prototype_counts.copy_(counts)


def _contrastive_chunk(
    flat: torch.Tensor,
    class_ids: torch.Tensor,
    start: int,
    stop: int,
    temperature: float,
    view_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    chunk = flat[start:stop]
    width = stop - start
    logits = chunk.matmul(flat.transpose(0, 1)).div_(temperature)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    positive = class_ids[start:stop, None].eq(class_ids[None, :])
    eligible = None
    if view_ids is not None:
        eligible = view_ids[start:stop, None].ne(view_ids[None, :])
        positive = positive & eligible
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
        if eligible is None:
            denom_logits.fill_diagonal_(float("-inf"))
        else:
            denom_logits.masked_fill_(~eligible, float("-inf"))
        denom = torch.logsumexp(denom_logits, dim=1, keepdim=True)
    else:
        self_mask = torch.zeros_like(positive)
        self_mask[eye, self_columns] = True
        positive = positive & ~self_mask
        excluded = self_mask if eligible is None else self_mask | ~eligible
        denom = torch.logsumexp(logits.masked_fill(excluded, float("-inf")), dim=1, keepdim=True)
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
    view_ids: torch.Tensor | None = None,
) -> torch.Tensor:
    """Memory-bounded CoMAL positive/shared-negative contrastive loss."""
    if features.ndim != 3 or labels.shape != features.shape[:2]:
        raise ValueError("expected latent features [B,L,D] and labels [B,L]")
    batch_size, num_labels, _feature_dim = features.shape
    flat = F.normalize(features, dim=-1).reshape(batch_size * num_labels, features.shape[-1])
    flat_view_ids = None
    if view_ids is not None:
        if view_ids.shape != labels.shape:
            raise ValueError("view_ids must have the same shape as labels")
        flat_view_ids = view_ids.reshape(-1)
    label_ids = torch.arange(num_labels, device=labels.device).expand(batch_size, -1)
    class_ids = torch.where(labels >= 0.5, label_ids, num_labels).reshape(-1)
    total = int(flat.shape[0])
    temperature = max(float(temperature), 1e-6)
    # Prefer one full pairwise GEMM when it fits; chunking is only a memory guard.
    bytes_needed = total * total * flat.element_size()
    max_full_bytes = 2 * 1024**3 if flat.is_cuda else 512 * 1024**2
    if bytes_needed <= max_full_bytes:
        loss_sum, loss_count = _contrastive_chunk(
            flat, class_ids, 0, total, temperature, flat_view_ids
        )
        return torch.where(loss_count > 0, loss_sum / loss_count.clamp_min(1), features.sum() * 0)
    step = max(1, int(anchor_chunk_size))
    loss_sum = flat.new_zeros(())
    loss_count = flat.new_zeros(())
    for start in range(0, total, step):
        stop = min(start + step, total)
        part_sum, part_count = _contrastive_chunk(
            flat, class_ids, start, stop, temperature, flat_view_ids
        )
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


def own_prototype_similarity(
    latent_features: torch.Tensor,
    prototypes: torch.Tensor,
    num_labels: int,
) -> torch.Tensor:
    # Prototypes are unit-normalized in set_prototypes; only normalize latents.
    latents = latent_features if latent_features.dtype == torch.float32 else latent_features.float()
    proto = prototypes[:num_labels]
    if proto.dtype != torch.float32:
        proto = proto.float()
    return torch.einsum("nld,ld->nl", F.normalize(latents, dim=-1), proto)


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


def _masked_mean(
    values: torch.Tensor, mask: torch.Tensor, *, dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    weights = mask.expand_as(values).to(dtype=values.dtype)
    count = weights.sum(dim=dim)
    total = (values * weights).sum(dim=dim)
    return total / count.clamp_min(1.0), count


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
    """Estimate view/label reliability weights and positive-region thresholds."""
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
    negative_mean, _negative_counts = _masked_mean(evidence, negative, dim=0)
    reliability = (positive_mean - negative_mean).clamp_min(0.0)

    label_positive_counts = positive_counts[0]
    pooled_denominator = label_positive_counts.sum().clamp_min(1.0)
    pooled_reliability = (
        reliability * label_positive_counts[None, :]
    ).sum(dim=1) / pooled_denominator
    reliability_strength = max(float(reliability_shrinkage), 0.0)
    shrunk_reliability = (
        label_positive_counts[None, :] * reliability
        + reliability_strength * pooled_reliability[:, None]
    ) / (label_positive_counts[None, :] + reliability_strength).clamp_min(1.0)

    num_views = int(similarities.shape[1])
    included_count = num_views if include_fused_in_weights or num_views == 1 else num_views - 1
    included_views = torch.arange(included_count, device=similarities.device)
    weights = torch.zeros_like(shrunk_reliability)
    if equal_weights:
        weights[included_views] = 1.0 / max(included_count, 1)
    else:
        selected = shrunk_reliability[included_views]
        denominators = selected.sum(dim=0, keepdim=True)
        uniform = torch.full_like(selected, 1.0 / max(included_count, 1))
        weights[included_views] = torch.where(
            denominators > 1e-12, selected / denominators.clamp_min(1e-12), uniform
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
        pooled_thresholds = (
            raw_thresholds * label_positive_counts[None, :]
        ).sum(dim=1) / pooled_denominator
        threshold_strength = max(float(threshold_shrinkage), 0.0)
        thresholds = (
            label_positive_counts[None, :] * raw_thresholds
            + threshold_strength * pooled_thresholds[:, None]
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
    """Modality-resolved extension of the released CoMAL paper score."""
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
        # This compatibility path makes the documented alpha=0, midpoint ablation
        # exactly equal to paper CoMAL on the fused (last) view.
        aggregate_evidence = evidence_by_view[:, -1]
        aggregate_thresholds = statistics.thresholds[-1]
    else:
        aggregate_evidence = (evidence_by_view * weights[None, :, :]).sum(dim=1)
        aggregate_thresholds = (statistics.thresholds * weights).sum(dim=0)

    prototype_positive = aggregate_evidence > ((aggregate_thresholds + 1.0) * 0.5)[None, :]
    prototype_positive_count = prototype_positive.sum(dim=1).float()
    cardinality_mismatch = (prototype_positive_count - expected_cardinality).abs()
    classifier_positive = probabilities >= 0.5
    positive_evidence = (
        classifier_positive.float() * aggregate_evidence.clamp_min(1e-10)
    ).sum(dim=1)
    inverse_positive_evidence = (
        positive_evidence + probabilities.shape[1] * 1e-10
    ).reciprocal()
    base_score = inverse_positive_evidence.sqrt() * cardinality_mismatch.sqrt()

    selected_views = statistics.included_views
    selected_evidence = evidence_by_view.index_select(1, selected_views)
    selected_weights = weights.index_select(0, selected_views)
    if dispersion_name == "weighted_mad":
        per_label_dispersion = (
            selected_weights[None, :, :]
            * (selected_evidence - aggregate_evidence[:, None, :]).abs()
        ).sum(dim=1)
    elif dispersion_name == "range":
        per_label_dispersion = selected_evidence.amax(dim=1) - selected_evidence.amin(dim=1)
    else:
        variance = (
            selected_weights[None, :, :]
            * (selected_evidence - aggregate_evidence[:, None, :]).square()
        ).sum(dim=1)
        per_label_dispersion = variance.clamp_min(0.0).sqrt()
    positive_count = classifier_positive.sum(dim=1)
    sample_dispersion = (
        per_label_dispersion * classifier_positive.float()
    ).sum(dim=1) / positive_count.clamp_min(1)
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
        own_similarity = own_prototype_similarity(latent_features, prototypes, int(probabilities.shape[1]))
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
