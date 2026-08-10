"""Formal round training for the Yang-Wu multimodal diagnosis baseline."""

from __future__ import annotations

import math
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR

from .config import require_paths
from .model import CoMALModule, YangWuBertEncoderClassifier, supervised_contrastive_loss
from .multimodal_data import YangWuFeatureStore


@dataclass
class TrainedMultimodalRound:
    classifier: YangWuBertEncoderClassifier
    comal: CoMALModule | None
    history: dict[str, list[float]]
    timings: dict[str, float]
    labeled_outputs: dict[str, torch.Tensor] | None = None
    labeled_own_similarity: torch.Tensor | None = None
    labeled_view_own_similarity: torch.Tensor | None = None
    modis_state: Any | None = None


def _autocast(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=device.type == "cuda")
        for key, value in batch.items()
    }


def build_classifier(
    config: dict[str, Any], device: torch.device
) -> YangWuBertEncoderClassifier:
    model = config.get("model", {})
    preprocessing = config.get("preprocessing", {})
    checkpoint = require_paths(config)["clinicalbert_checkpoint"]
    classifier = YangWuBertEncoderClassifier(
        str(checkpoint),
        num_labels=int(model["output_size"]),
        time_invariant_dim=int(preprocessing["time_invariant_dim"]),
        time_invariant_hidden_dim=int(model["time_invariant_hidden_dim"]),
        time_series_dim=int(preprocessing["time_series_dim"]),
        time_series_hidden_dim=int(model["time_series_hidden_dim"]),
        time_series_layers=int(model["time_series_layers"]),
        time_series_heads=int(model["time_series_heads"]),
        text_hidden_dim=int(model["text_hidden_dim"]),
        dropout=float(model["dropout"]),
    )
    return classifier.to(device)


def _linear_warmup_decay(total_steps: int, warmup_proportion: float):
    warmup_steps = int(total_steps * warmup_proportion)

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        return max(0.0, float(total_steps - step) / max(1, total_steps - warmup_steps))

    return schedule


@torch.inference_mode()
def collect_classifier_outputs(
    classifier: YangWuBertEncoderClassifier,
    store: YangWuFeatureStore,
    indices: Iterable[int],
    config: dict[str, Any],
    device: torch.device,
    *,
    return_tokens: bool,
) -> dict[str, torch.Tensor]:
    selected = np.asarray(list(indices), dtype=np.int64)
    training = config.get("training", {})
    loader = store.make_loader(
        selected,
        batch_size=int(training.get("eval_batch_size", 400)),
        shuffle=False,
        num_workers=int(training.get("num_workers", 12)),
        pin_memory=bool(training.get("pin_memory", True)),
    )
    values: dict[str, list[torch.Tensor]] = {
        "labels": [],
        "probabilities": [],
        "features": [],
        "indices": [],
    }
    if return_tokens:
        values["modality_tokens"] = []
    classifier.eval()
    for host_batch in loader:
        batch = _move_batch(host_batch, device)
        output = classifier(batch, return_tokens=return_tokens)
        values["labels"].append(batch["labels"].float())
        values["probabilities"].append(output["probabilities"].float())
        values["features"].append(output["features"].float())
        values["indices"].append(batch["index"].long())
        if return_tokens:
            values["modality_tokens"].append(output["modality_tokens"].float())
    result: dict[str, torch.Tensor] = {}
    for name, chunks in values.items():
        if chunks:
            result[name] = torch.cat(chunks, dim=0)
        elif name == "modality_tokens":
            result[name] = torch.empty(
                (0, 3, classifier.feature_dim), dtype=torch.float32, device=device
            )
        else:
            width = store.audit.label_count if name in {"labels", "probabilities"} else 0
            result[name] = torch.empty((0, width), dtype=torch.float32, device=device)
    return result


def _sample_contrastive_labels(
    latent: torch.Tensor, labels: torch.Tensor, maximum_labels: int
) -> tuple[torch.Tensor, torch.Tensor]:
    label_count = int(labels.shape[1])
    maximum = min(max(1, int(maximum_labels)), label_count)
    if maximum == label_count:
        return latent, labels
    positive = torch.nonzero(labels.any(dim=0), as_tuple=False).flatten()
    if positive.numel() >= maximum:
        selected = positive[torch.randperm(positive.numel(), device=labels.device)[:maximum]]
    else:
        mask = torch.ones(label_count, dtype=torch.bool, device=labels.device)
        mask[positive] = False
        negative = torch.nonzero(mask, as_tuple=False).flatten()
        needed = maximum - int(positive.numel())
        negative = negative[
            torch.randperm(negative.numel(), device=labels.device)[:needed]
        ]
        selected = torch.cat((positive, negative))
    selected = selected.sort().values
    return latent.index_select(-2, selected), labels.index_select(-1, selected)


def _contrastive_loss(
    latent: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float,
    anchor_chunk_size: int,
    cross_modal_weight: float,
) -> torch.Tensor:
    if latent.ndim == 3:
        return supervised_contrastive_loss(
            latent,
            labels,
            temperature=temperature,
            anchor_chunk_size=anchor_chunk_size,
        )
    views = int(latent.shape[1])
    within = torch.stack(
        [
            supervised_contrastive_loss(
                latent[:, view],
                labels,
                temperature=temperature,
                anchor_chunk_size=anchor_chunk_size,
            )
            for view in range(views)
        ]
    ).mean()
    if cross_modal_weight <= 0:
        return within
    expanded_labels = labels[:, None, :].expand(-1, views, -1).reshape(
        labels.shape[0] * views, labels.shape[1]
    )
    view_ids = torch.arange(views, device=labels.device)[None, :, None].expand(
        labels.shape[0], -1, labels.shape[1]
    ).reshape(labels.shape[0] * views, labels.shape[1])
    cross = supervised_contrastive_loss(
        latent.reshape(labels.shape[0] * views, labels.shape[1], latent.shape[-1]),
        expanded_labels,
        temperature=temperature,
        anchor_chunk_size=anchor_chunk_size,
        view_ids=view_ids,
    )
    return within + cross_modal_weight * cross


@torch.inference_mode()
def refresh_prototypes(
    comal: CoMALModule,
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    batch_size: int,
) -> None:
    sums = torch.zeros_like(comal.prototypes, dtype=torch.float32)
    counts = torch.zeros_like(comal.prototype_counts, dtype=torch.float32)
    for start in range(0, int(features.shape[0]), batch_size):
        stop = min(start + batch_size, int(features.shape[0]))
        latent = F.normalize(
            comal(
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
    comal.set_prototypes(sums, counts)


@torch.inference_mode()
def attach_comal_outputs(
    comal: CoMALModule,
    outputs: dict[str, torch.Tensor],
    *,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    if comal.num_views > 1:
        features = torch.cat(
            (outputs["modality_tokens"], outputs["features"][:, None, :]), dim=1
        )
    else:
        features = outputs["features"]
    similarities = []
    for start in range(0, int(features.shape[0]), batch_size):
        stop = min(start + batch_size, int(features.shape[0]))
        similarities.append(
            comal(
                features[start:stop],
                compute_similarities="own_bg",
                compute_reconstruction=False,
            )["prototype_similarities"].float()
        )
    result = dict(outputs)
    result["prototype_similarities"] = torch.cat(similarities, dim=0)
    if comal.num_views > 1:
        result["view_own_similarity"] = result["prototype_similarities"][..., 0]
    return result


def _train_comal(
    classifier: YangWuBertEncoderClassifier,
    store: YangWuFeatureStore,
    indices: np.ndarray,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[
    CoMALModule,
    list[float],
    dict[str, torch.Tensor],
    torch.Tensor,
    torch.Tensor | None,
]:
    strategy = str(config["active_learning"]["strategy"]).lower()
    multi_view = strategy == "mm_comal"
    outputs = collect_classifier_outputs(
        classifier, store, indices, config, device, return_tokens=multi_view
    )
    features = (
        torch.cat((outputs["modality_tokens"], outputs["features"][:, None, :]), dim=1)
        if multi_view
        else outputs["features"]
    )
    labels = outputs["labels"]
    cfg = config.get("comal", {})
    training = config.get("training", {})
    comal = CoMALModule(
        classifier.feature_dim,
        int(labels.shape[1]),
        int(cfg.get("label_dim", 8)),
        int(cfg.get("prototype_dim", 8)),
        num_views=4 if multi_view else 1,
    ).to(device)
    optimizer = AdamW(
        comal.parameters(),
        lr=float(cfg.get("learning_rate", 1e-3)),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    reconstruction_weight = float(cfg.get("reconstruction_weight", 0.2))
    classification_weight = float(cfg.get("classification_weight", 0.5))
    batch_size = int(training.get("comal_batch_size", training.get("batch_size", 40)))
    epochs = int(training["comal_epochs"])
    history: list[float] = []
    comal.train()
    for _epoch in range(epochs):
        order = torch.randperm(features.shape[0], device=device)
        losses = []
        for start in range(0, int(order.numel()), batch_size):
            selected = order[start : start + batch_size]
            batch_features = features.index_select(0, selected)
            targets = labels.index_select(0, selected)
            optimizer.zero_grad(set_to_none=True)
            output = comal(batch_features, compute_similarities=False)
            latent, contrastive_targets = _sample_contrastive_labels(
                output["latent_features"],
                targets,
                int(cfg.get("contrastive_label_sample_size", 256)),
            )
            contrastive = _contrastive_loss(
                latent,
                contrastive_targets,
                temperature=float(cfg.get("temperature", 0.07)),
                anchor_chunk_size=int(cfg.get("anchor_chunk_size", 1024)),
                cross_modal_weight=float(cfg.get("cross_modal_weight", 0.15)),
            )
            fused_target = batch_features if batch_features.ndim == 2 else batch_features[:, -1]
            reconstruction = F.mse_loss(output["reconstructed_features"], fused_target)
            reconstruction_bce = F.binary_cross_entropy_with_logits(
                output["reconstructed_logits"], targets
            )
            loss = (
                contrastive
                + reconstruction_weight * reconstruction
                + classification_weight * reconstruction_bce
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(comal.parameters(), float(training["gradient_clip"]))
            optimizer.step()
            losses.append(loss.detach())
        history.append(float(torch.stack(losses).mean().cpu()))
    comal.eval()
    refresh_prototypes(
        comal,
        features,
        labels,
        batch_size=int(training.get("eval_batch_size", 400)),
    )
    attached = attach_comal_outputs(
        comal,
        outputs,
        batch_size=int(training.get("eval_batch_size", 400)),
    )
    similarities = attached["prototype_similarities"]
    if multi_view:
        view_own = similarities[..., 0]
        own = view_own[:, -1]
    else:
        view_own = None
        own = similarities[..., 0]
    return comal, history, outputs, own, view_own


def train_multimodal_round(
    store: YangWuFeatureStore,
    labeled_indices: Iterable[int],
    config: dict[str, Any],
    device: torch.device,
) -> TrainedMultimodalRound:
    indices = np.unique(np.asarray(list(labeled_indices), dtype=np.int64))
    if not indices.size:
        raise ValueError("at least one labeled sample is required")
    training = config.get("training", {})
    model_seed = int(config.get("model", {}).get("seed", 1337))
    torch.manual_seed(model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(model_seed)
    classifier = build_classifier(config, device)
    loader = store.make_loader(
        indices,
        batch_size=int(training.get("batch_size", 40)),
        shuffle=True,
        num_workers=int(training.get("num_workers", 12)),
        pin_memory=bool(training.get("pin_memory", True)),
    )
    epochs = int(training["epochs"])
    total_steps = max(1, epochs * math.ceil(len(indices) / int(training["batch_size"])))
    optimizer = Adam(
        classifier.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = LambdaLR(
        optimizer,
        _linear_warmup_decay(total_steps, float(training["warmup_proportion"])),
    )
    precision = str(training["precision"])
    history: dict[str, list[float]] = {"classifier_loss": [], "comal_loss": []}
    timings: dict[str, float] = {}
    start = time.perf_counter()
    classifier.train()
    for _epoch in range(epochs):
        losses = []
        for host_batch in loader:
            batch = _move_batch(host_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, precision):
                output = classifier(batch)
                loss = F.binary_cross_entropy_with_logits(output["logits"], batch["labels"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), float(training["gradient_clip"]))
            optimizer.step()
            scheduler.step()
            losses.append(loss.detach())
        history["classifier_loss"].append(float(torch.stack(losses).mean().cpu()))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    timings["classifier_training_sec"] = time.perf_counter() - start

    strategy = str(config["active_learning"]["strategy"]).lower()
    comal = None
    labeled_outputs = None
    labeled_own = None
    labeled_view_own = None
    modis_state = None
    if strategy in {"comal", "mm_comal"}:
        start = time.perf_counter()
        comal, comal_history, labeled_outputs, labeled_own, labeled_view_own = _train_comal(
            classifier, store, indices, config, device
        )
        history["comal_loss"] = comal_history
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings["comal_training_sec"] = time.perf_counter() - start
    elif strategy == "modis":
        start = time.perf_counter()
        labeled_outputs = collect_classifier_outputs(
            classifier, store, indices, config, device, return_tokens=True
        )
        from modis.probes import train_modality_probes

        modis_state = train_modality_probes(
            labeled_outputs["modality_tokens"],
            labeled_outputs["labels"],
            labeled_outputs["indices"].detach().cpu().numpy(),
            config,
            seed=int(training.get("seed", 17)),
        )
        history["probe_loss"] = modis_state.history
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings["modis_probe_training_sec"] = time.perf_counter() - start
    elif strategy == "mosaic":
        labeled_outputs = collect_classifier_outputs(
            classifier, store, indices, config, device, return_tokens=True
        )

    return TrainedMultimodalRound(
        classifier=classifier,
        comal=comal,
        history=history,
        timings=timings,
        labeled_outputs=labeled_outputs,
        labeled_own_similarity=labeled_own,
        labeled_view_own_similarity=labeled_view_own,
        modis_state=modis_state,
    )
