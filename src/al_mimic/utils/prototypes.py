"""Generic label-prototype modules and streaming prototype operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from al_mimic.utils.contrastive import (
    multiview_contrastive_loss,
    sample_label_dimensions,
)


@dataclass(frozen=True)
class PrototypeFitOptions:
    epochs: int = 20
    batch_size: int = 64
    eval_batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    gradient_clip: float = 1.0
    maximum_labels: int = 256
    temperature: float = 0.07
    anchor_chunk_size: int = 1024
    cross_view_weight: float = 0.15
    reconstruction_weight: float = 0.2
    classification_weight: float = 0.5
    seed: int = 0

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None) -> PrototypeFitOptions:
        root = config or {}
        method = root.get("comal", {})
        training = root.get("training", {})
        if not isinstance(method, Mapping):
            method = {}
        if not isinstance(training, Mapping):
            training = {}
        options = cls(
            epochs=int(method.get("epochs", training.get("comal_epochs", 20))),
            batch_size=int(
                method.get(
                    "batch_size",
                    training.get("comal_batch_size", training.get("batch_size", 64)),
                )
            ),
            eval_batch_size=int(method.get("eval_batch_size", training.get("eval_batch_size", 256))),
            learning_rate=float(method.get("learning_rate", 1e-3)),
            weight_decay=float(method.get("weight_decay", training.get("weight_decay", 0.0))),
            gradient_clip=float(method.get("gradient_clip", training.get("gradient_clip", 1.0))),
            maximum_labels=int(method.get("contrastive_label_sample_size", 256)),
            temperature=float(method.get("temperature", 0.07)),
            anchor_chunk_size=int(method.get("anchor_chunk_size", 1024)),
            cross_view_weight=float(method.get("cross_modal_weight", 0.15)),
            reconstruction_weight=float(method.get("reconstruction_weight", 0.2)),
            classification_weight=float(method.get("classification_weight", 0.5)),
            seed=int(method.get("seed", training.get("seed", 0))),
        )
        if min(options.epochs, options.batch_size, options.eval_batch_size) < 1:
            raise ValueError("prototype epochs and batch sizes must be positive")
        if options.learning_rate <= 0 or options.weight_decay < 0:
            raise ValueError("prototype learning rate must be positive and weight decay non-negative")
        if options.gradient_clip < 0:
            raise ValueError("prototype gradient_clip must be non-negative")
        return options


@dataclass(frozen=True)
class PrototypeFitState:
    module: LabelPrototypeAutoencoder
    history: tuple[float, ...]
    labeled_outputs: dict[str, torch.Tensor]
    labeled_own_similarity: torch.Tensor
    labeled_view_own_similarity: torch.Tensor | None
    prototypes: torch.Tensor
    eval_batch_size: int


class LabelPrototypeAutoencoder(nn.Module):
    """Learn label-wise latent features with one shared background prototype."""

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
        if min(input_dim, num_labels, label_dim, prototype_dim, num_views) < 1:
            raise ValueError("prototype module dimensions must be positive")
        self.num_labels = int(num_labels)
        self.prototype_dim = int(prototype_dim)
        self.num_views = int(num_views)
        self.to_label = nn.Linear(input_dim, num_labels * label_dim)
        self.to_latent = nn.Linear(label_dim, prototype_dim)
        self.from_latent = nn.Linear(prototype_dim, label_dim)
        self.aggregate = nn.Linear(num_labels * label_dim, input_dim)
        self.reconstruction_classifier = nn.Linear(input_dim, num_labels)
        if self.num_views == 1:
            self.register_parameter("view_code", None)
            self.register_buffer(
                "prototypes",
                torch.zeros(num_labels + 1, prototype_dim),
            )
            self.register_buffer("prototype_counts", torch.zeros(num_labels + 1))
        else:
            self.view_code = nn.Parameter(torch.zeros(num_views, input_dim))
            self.register_buffer(
                "prototypes",
                torch.zeros(num_views, num_labels + 1, prototype_dim),
            )
            self.register_buffer(
                "prototype_counts",
                torch.zeros(num_views, num_labels + 1),
            )
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
        if features.ndim not in {2, 3}:
            raise ValueError("features must have shape [N,D] or [N,V,D]")
        batch = features.shape[0]
        if features.ndim == 2:
            view_index = self.num_views - 1
            inputs = features if self.view_code is None else features + self.view_code[view_index]
            label_features = self.to_label(inputs).view(batch, self.num_labels, -1)
            latent = self.to_latent(label_features)
            prototype_bank = self.prototypes if self.num_views == 1 else self.prototypes[view_index]
        elif self.num_views > 1:
            if features.shape[1] != self.num_views:
                raise ValueError(f"expected {self.num_views} views, got {features.shape[1]}")
            inputs = features + self.view_code[None, :, :]
            label_features = self.to_label(inputs).view(
                batch,
                self.num_views,
                self.num_labels,
                -1,
            )
            latent = self.to_latent(label_features)
            prototype_bank = self.prototypes
        else:
            raise ValueError("single-view prototype module requires features [N,D]")

        result: dict[str, torch.Tensor] = {"latent_features": latent}
        if compute_reconstruction:
            reconstruction_latent = latent if latent.ndim == 3 else latent[:, -1]
            decoded = self.from_latent(reconstruction_latent).reshape(batch, -1)
            reconstructed = self.aggregate(decoded)
            result["reconstructed_features"] = reconstructed
            result["reconstructed_logits"] = self.reconstruction_classifier(reconstructed)
        if compute_similarities is True or compute_similarities == "full":
            normalized = F.normalize(latent, dim=-1)
            if normalized.ndim == 3:
                result["prototype_similarities"] = normalized @ prototype_bank.T
            else:
                result["prototype_similarities"] = torch.einsum(
                    "nvld,vkd->nvlk",
                    normalized,
                    prototype_bank,
                )
        elif compute_similarities == "own_bg":
            normalized = F.normalize(latent, dim=-1)
            if normalized.ndim == 3:
                own = torch.einsum(
                    "nld,ld->nl",
                    normalized,
                    prototype_bank[:-1],
                )
                background = normalized.matmul(prototype_bank[-1])
            else:
                own = torch.einsum(
                    "nvld,vld->nvl",
                    normalized,
                    prototype_bank[:, :-1],
                )
                background = torch.einsum(
                    "nvld,vd->nvl",
                    normalized,
                    prototype_bank[:, -1],
                )
            result["prototype_similarities"] = torch.stack(
                (own, background),
                dim=-1,
            )
        elif compute_similarities not in {False, "none"}:
            raise ValueError("compute_similarities must be True/'full', 'own_bg', or False/'none'")
        return result

    @torch.no_grad()
    def set_prototypes(self, sums: torch.Tensor, counts: torch.Tensor) -> None:
        if sums.shape != self.prototypes.shape:
            raise ValueError("prototype sums do not match configured views and labels")
        if counts.shape != self.prototype_counts.shape:
            raise ValueError("prototype counts do not match configured views and labels")
        normalized = F.normalize(sums, dim=-1)
        self.prototypes.copy_(
            torch.where(
                counts.unsqueeze(-1) > 0,
                normalized,
                self.prototypes,
            )
        )
        self.prototype_counts.copy_(counts)


def stack_output_views(
    outputs: dict[str, torch.Tensor],
    *,
    num_views: int,
) -> torch.Tensor:
    """Build single- or multi-view prototype inputs from canonical outputs."""
    if "features" not in outputs:
        raise ValueError("outputs must include features")
    if num_views == 1:
        return outputs["features"]
    if "modality_tokens" not in outputs:
        raise ValueError("multi-view prototype inputs require modality_tokens")
    tokens = outputs["modality_tokens"]
    if tokens.ndim != 3 or tokens.shape[1] + 1 != num_views:
        raise ValueError("modality token count does not match configured prototype views")
    return torch.cat((tokens, outputs["features"][:, None, :]), dim=1)


@torch.inference_mode()
def refresh_prototypes(
    module: LabelPrototypeAutoencoder,
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    batch_size: int,
) -> None:
    """Refresh positive-label and shared-background prototype means."""
    if labels.ndim != 2 or labels.shape[0] != features.shape[0]:
        raise ValueError("features and labels must be row-aligned")
    if labels.shape[1] != module.num_labels:
        raise ValueError("labels do not match the prototype module")
    if features.shape[0] == 0:
        raise ValueError("at least one labeled feature is required")
    sums = torch.zeros_like(module.prototypes, dtype=torch.float32)
    counts = torch.zeros_like(module.prototype_counts, dtype=torch.float32)
    step = max(1, int(batch_size))
    for start in range(0, int(features.shape[0]), step):
        stop = min(start + step, int(features.shape[0]))
        latent = F.normalize(
            module(
                features[start:stop],
                compute_similarities=False,
                compute_reconstruction=False,
            )["latent_features"].float(),
            dim=-1,
        )
        targets = labels[start:stop].float()
        negative = 1.0 - targets
        if latent.ndim == 3:
            sums[:-1] += torch.einsum("bl,bld->ld", targets, latent)
            counts[:-1] += targets.sum(dim=0)
            sums[-1] += torch.einsum("bl,bld->d", negative, latent)
            counts[-1] += negative.sum()
        else:
            sums[:, :-1] += torch.einsum("bl,bvld->vld", targets, latent)
            counts[:, :-1] += targets.sum(dim=0)[None, :]
            sums[:, -1] += torch.einsum("bl,bvld->vd", negative, latent)
            counts[:, -1] += negative.sum()
    module.set_prototypes(sums, counts)


@torch.inference_mode()
def attach_prototype_outputs(
    module: LabelPrototypeAutoencoder,
    outputs: dict[str, torch.Tensor],
    *,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    """Attach own/background prototype similarities to canonical outputs."""
    features = stack_output_views(outputs, num_views=module.num_views)
    chunks: list[torch.Tensor] = []
    step = max(1, int(batch_size))
    for start in range(0, int(features.shape[0]), step):
        stop = min(start + step, int(features.shape[0]))
        chunks.append(
            module(
                features[start:stop],
                compute_similarities="own_bg",
                compute_reconstruction=False,
            )["prototype_similarities"].float()
        )
    if chunks:
        similarities = torch.cat(chunks, dim=0)
    else:
        shape = (0, module.num_labels, 2)
        if module.num_views > 1:
            shape = (0, module.num_views, module.num_labels, 2)
        similarities = torch.empty(
            shape,
            dtype=torch.float32,
            device=features.device,
        )
    result = dict(outputs)
    result["prototype_similarities"] = similarities
    if module.num_views > 1:
        result["view_own_similarity"] = similarities[..., 0]
        result["own_similarity"] = result["view_own_similarity"][:, -1]
    else:
        result["own_similarity"] = similarities[..., 0]
    return result


@torch.inference_mode()
def finalize_prototype_outputs(
    module: LabelPrototypeAutoencoder,
    outputs: dict[str, torch.Tensor],
    *,
    batch_size: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor | None]:
    """Refresh a prototype bank and attach labeled-pool similarities."""
    if "labels" not in outputs:
        raise ValueError("outputs must include labels")
    features = stack_output_views(outputs, num_views=module.num_views)
    refresh_prototypes(
        module,
        features,
        outputs["labels"],
        batch_size=batch_size,
    )
    attached = attach_prototype_outputs(module, outputs, batch_size=batch_size)
    similarities = attached["prototype_similarities"]
    if module.num_views > 1:
        view_own = similarities[..., 0]
        own = view_own[:, -1]
    else:
        view_own = None
        own = similarities[..., 0]
    return attached, own, view_own


def prototype_training_loss(
    module: LabelPrototypeAutoencoder,
    classifier_outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    *,
    maximum_labels: int = 256,
    temperature: float = 0.07,
    anchor_chunk_size: int = 1024,
    cross_view_weight: float = 0.15,
    reconstruction_weight: float = 0.2,
    classification_weight: float = 0.5,
) -> torch.Tensor:
    """Train a prototype branch without backpropagating into its classifier."""
    detached_outputs = {name: value.detach() for name, value in classifier_outputs.items()}
    features = stack_output_views(detached_outputs, num_views=module.num_views)
    output = module(features, compute_similarities=False)
    latent, sampled_labels = sample_label_dimensions(output["latent_features"], labels, maximum_labels)
    contrastive = multiview_contrastive_loss(
        latent,
        sampled_labels,
        temperature=temperature,
        anchor_chunk_size=anchor_chunk_size,
        cross_view_weight=cross_view_weight,
    )
    fused_target = features if features.ndim == 2 else features[:, -1]
    reconstruction = F.mse_loss(output["reconstructed_features"], fused_target)
    classification = F.binary_cross_entropy_with_logits(output["reconstructed_logits"], labels.float())
    return (
        contrastive
        + float(reconstruction_weight) * reconstruction
        + float(classification_weight) * classification
    )


def build_and_fit_prototype_module(
    module_class: type[LabelPrototypeAutoencoder],
    labeled_outputs: Mapping[str, torch.Tensor],
    options: PrototypeFitOptions,
    *,
    num_views: int,
    label_dim: int,
    prototype_dim: int,
) -> PrototypeFitState:
    """Infer dimensions, initialize reproducibly, and fit a prototype module."""
    if "features" not in labeled_outputs or "labels" not in labeled_outputs:
        raise ValueError("labeled_outputs must include features and labels")
    features = labeled_outputs["features"]
    labels = labeled_outputs["labels"]
    if features.ndim != 2 or labels.ndim != 2 or features.shape[0] != labels.shape[0]:
        raise ValueError("labeled features and labels must have shapes [N,D] and [N,L]")
    if num_views > 1:
        tokens = labeled_outputs.get("modality_tokens")
        if tokens is None or tokens.ndim != 3 or tokens.shape[1] + 1 != num_views:
            raise ValueError("multi-view fitting requires row-aligned modality_tokens")
    cuda_devices: list[int] = []
    if features.device.type == "cuda":
        cuda_devices = [
            features.device.index if features.device.index is not None else torch.cuda.current_device()
        ]
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(options.seed)
        module = module_class(
            input_dim=int(features.shape[1]),
            num_labels=int(labels.shape[1]),
            label_dim=int(label_dim),
            prototype_dim=int(prototype_dim),
            num_views=int(num_views),
        ).to(features.device)
        return fit_prototype_module(module, labeled_outputs, options)


def fit_prototype_module(
    module: LabelPrototypeAutoencoder,
    labeled_outputs: Mapping[str, torch.Tensor],
    options: PrototypeFitOptions,
) -> PrototypeFitState:
    """Optimize, refresh, and attach a prototype module from in-memory outputs."""
    required = {"features", "labels"}
    if not required.issubset(labeled_outputs):
        raise ValueError("labeled_outputs must include features and labels")
    detached = {name: value.detach() for name, value in labeled_outputs.items()}
    labels = detached["labels"].float()
    features = stack_output_views(detached, num_views=module.num_views)
    if labels.ndim != 2 or labels.shape[0] != features.shape[0]:
        raise ValueError("labeled features and labels must be row-aligned")
    if labels.shape[1] != module.num_labels:
        raise ValueError("labeled outputs do not match the prototype module")
    if labels.shape[0] == 0:
        raise ValueError("at least one labeled output is required")
    try:
        module_device = next(module.parameters()).device
    except StopIteration as exc:
        raise ValueError("prototype module must have trainable parameters") from exc
    if module_device != features.device:
        raise ValueError("prototype module and labeled outputs must use the same device")

    optimizer = AdamW(
        module.parameters(),
        lr=options.learning_rate,
        weight_decay=options.weight_decay,
    )
    count = int(labels.shape[0])
    step = min(options.batch_size, count)
    cuda_devices: list[int] = []
    if features.device.type == "cuda":
        cuda_devices = [
            features.device.index if features.device.index is not None else torch.cuda.current_device()
        ]
    history: list[float] = []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(options.seed)
        module.train()
        for _ in range(options.epochs):
            order = torch.randperm(count, device=features.device)
            epoch_loss = features.new_zeros((), dtype=torch.float32)
            batches = 0
            for start in range(0, count, step):
                selected = order[start : start + step]
                batch_outputs = {
                    name: value.index_select(0, selected)
                    for name, value in detached.items()
                    if value.ndim > 0 and value.shape[0] == count
                }
                targets = labels.index_select(0, selected)
                optimizer.zero_grad(set_to_none=True)
                loss = prototype_training_loss(
                    module,
                    batch_outputs,
                    targets,
                    maximum_labels=options.maximum_labels,
                    temperature=options.temperature,
                    anchor_chunk_size=options.anchor_chunk_size,
                    cross_view_weight=options.cross_view_weight,
                    reconstruction_weight=options.reconstruction_weight,
                    classification_weight=options.classification_weight,
                )
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("prototype training loss became non-finite")
                loss.backward()
                if options.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(module.parameters(), options.gradient_clip)
                optimizer.step()
                epoch_loss = epoch_loss + loss.detach().float()
                batches += 1
            history.append(float(epoch_loss / batches))

    module.eval()
    attached, own, view_own = finalize_prototype_outputs(
        module,
        detached,
        batch_size=options.eval_batch_size,
    )
    return PrototypeFitState(
        module=module,
        history=tuple(history),
        labeled_outputs=attached,
        labeled_own_similarity=own,
        labeled_view_own_similarity=view_own,
        prototypes=module.prototypes,
        eval_batch_size=options.eval_batch_size,
    )


def own_prototype_similarity(
    latent_features: torch.Tensor,
    prototypes: torch.Tensor,
    num_labels: int,
) -> torch.Tensor:
    """Return each label latent's cosine similarity to its own prototype."""
    if latent_features.ndim != 3:
        raise ValueError("latent_features must have shape [N,L,D]")
    if latent_features.shape[1] != num_labels:
        raise ValueError("latent label dimension does not match num_labels")
    latents = latent_features.float()
    own = prototypes[:num_labels].float()
    return torch.einsum("nld,ld->nl", F.normalize(latents, dim=-1), own)


@torch.inference_mode()
def positive_similarity_thresholds(
    labels: torch.Tensor,
    *,
    own_similarity: torch.Tensor,
) -> torch.Tensor:
    """Compute per-label positive-region midpoint thresholds."""
    if labels.shape != own_similarity.shape or labels.ndim != 2:
        raise ValueError("labels and own_similarity must have shape [N,L]")
    positive = labels >= 0.5
    large = torch.finfo(own_similarity.dtype).max
    minima = own_similarity.masked_fill(~positive, large).min(dim=0).values
    maxima = own_similarity.masked_fill(~positive, -large).max(dim=0).values
    return torch.where(
        positive.any(dim=0),
        (minima + maxima) * 0.5,
        torch.zeros_like(minima),
    )


__all__ = [
    "LabelPrototypeAutoencoder",
    "PrototypeFitOptions",
    "PrototypeFitState",
    "attach_prototype_outputs",
    "build_and_fit_prototype_module",
    "finalize_prototype_outputs",
    "fit_prototype_module",
    "own_prototype_similarity",
    "positive_similarity_thresholds",
    "prototype_training_loss",
    "refresh_prototypes",
    "stack_output_views",
]
