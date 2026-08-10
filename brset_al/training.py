"""Fresh-round BRSET classifier, CoMAL, MoDIS, and MoSAIC training state."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam, AdamW

from mimic_comal.model import CoMALModule
from mimic_comal.multimodal_training import (
    _contrastive_loss,
    _sample_contrastive_labels,
    attach_comal_outputs,
    refresh_prototypes,
)

from .data import BrsetFeatureStore
from .model import BrsetMultimodalClassifier, initialize_fusion_layers


@dataclass
class TrainedBrsetRound:
    classifier: BrsetMultimodalClassifier
    comal: CoMALModule | None
    history: dict[str, list[float]]
    timings: dict[str, float]
    labeled_outputs: dict[str, torch.Tensor] | None = None
    labeled_own_similarity: torch.Tensor | None = None
    labeled_view_own_similarity: torch.Tensor | None = None
    modis_state: Any = None


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def build_classifier(
    store: BrsetFeatureStore,
    config: dict[str, Any],
    device: torch.device,
) -> BrsetMultimodalClassifier:
    model_cfg = config.get("model", {})
    model = BrsetMultimodalClassifier(
        store.schema.dimension,
        num_labels=int(model_cfg["output_size"]),
        image_feature_dim=int(model_cfg["image_feature_dim"]),
        metadata_hidden_dim=int(model_cfg["metadata_hidden_dim"]),
        fusion_dim=int(model_cfg["fusion_dim"]),
        dropout=float(model_cfg.get("dropout", 0.2)),
        image_weights=str(model_cfg["image_weights"]),
    )
    initialize_fusion_layers(model)
    return model.to(device)


@torch.inference_mode()
def collect_outputs(
    classifier: BrsetMultimodalClassifier,
    store: BrsetFeatureStore,
    indices: Sequence[int],
    config: dict[str, Any],
    device: torch.device,
    *,
    return_tokens: bool,
) -> dict[str, torch.Tensor]:
    training = config.get("training", {})
    loader = store.make_loader(
        indices,
        train=False,
        batch_size=int(training.get("eval_batch_size", 64)),
        shuffle=False,
    )
    collected: dict[str, list[torch.Tensor]] = {
        "indices": [],
        "labels": [],
        "probabilities": [],
        "features": [],
    }
    if return_tokens:
        collected["modality_tokens"] = []
    classifier.eval()
    for host_batch in loader:
        batch = _move_batch(host_batch, device)
        output = classifier(batch, return_tokens=return_tokens)
        for name in collected:
            source = batch[name] if name in {"indices", "labels"} else output[name]
            collected[name].append(source.detach().float() if name != "indices" else source.detach())
    return {name: torch.cat(values, dim=0) for name, values in collected.items()}


@torch.inference_mode()
def aggregate_patient_outputs(
    classifier: BrsetMultimodalClassifier,
    store: BrsetFeatureStore,
    outputs: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], list[str]]:
    if "modality_tokens" not in outputs:
        raise ValueError("patient aggregation requires modality tokens")
    global_indices = outputs["indices"].detach().cpu().numpy().astype(np.int64)
    row_patients = [str(store.patient_ids_array[index]) for index in global_indices]
    patients = sorted(set(row_patients))
    patient_position = {patient: index for index, patient in enumerate(patients)}
    inverse = torch.as_tensor(
        [patient_position[patient] for patient in row_patients],
        dtype=torch.long,
        device=outputs["modality_tokens"].device,
    )
    tokens = outputs["modality_tokens"].float()
    grouped_tokens = torch.zeros(
        len(patients), tokens.shape[1], tokens.shape[2], dtype=torch.float32, device=tokens.device
    )
    grouped_tokens.index_add_(0, inverse, tokens)
    counts = torch.bincount(inverse, minlength=len(patients)).float().clamp_min(1.0)
    grouped_tokens = grouped_tokens / counts[:, None, None]
    fused = classifier.fuse_from_tokens(grouped_tokens)
    labels = torch.as_tensor(store.patient_targets(patients), dtype=torch.float32, device=tokens.device)
    return {
        "indices": torch.arange(len(patients), dtype=torch.long, device=tokens.device),
        "labels": labels,
        "modality_tokens": grouped_tokens,
        "features": fused,
        "probabilities": classifier.probabilities_from_fused(fused),
    }, patients


def _train_comal(
    classifier: BrsetMultimodalClassifier,
    store: BrsetFeatureStore,
    indices: np.ndarray,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[CoMALModule, list[float], dict[str, torch.Tensor], torch.Tensor, torch.Tensor | None]:
    strategy = str(config["active_learning"]["strategy"]).lower()
    multi_view = strategy == "mm_comal"
    outputs = collect_outputs(classifier, store, indices, config, device, return_tokens=multi_view)
    features = (
        torch.cat((outputs["modality_tokens"], outputs["features"][:, None, :]), dim=1)
        if multi_view
        else outputs["features"]
    )
    labels = outputs["labels"]
    cfg = config.get("comal", {})
    training = config.get("training", {})
    num_views = len(classifier.modality_names) + 1 if multi_view else 1
    comal = CoMALModule(
        classifier.feature_dim,
        labels.shape[1],
        int(cfg.get("label_dim", 8)),
        int(cfg.get("prototype_dim", 8)),
        num_views=num_views,
    ).to(device)
    optimizer = AdamW(
        comal.parameters(),
        lr=float(cfg.get("learning_rate", 1e-3)),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    batch_size = int(training.get("comal_batch_size", training["batch_size"]))
    history: list[float] = []
    comal.train()
    for _epoch in range(int(training["comal_epochs"])):
        losses: list[torch.Tensor] = []
        order = torch.randperm(features.shape[0], device=device)
        for start in range(0, int(order.numel()), batch_size):
            selected = order[start : start + batch_size]
            batch_features = features.index_select(0, selected)
            targets = labels.index_select(0, selected)
            optimizer.zero_grad(set_to_none=True)
            output = comal(batch_features, compute_similarities=False)
            latent, contrastive_targets = _sample_contrastive_labels(
                output["latent_features"],
                targets,
                int(cfg.get("contrastive_label_sample_size", 13)),
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
            classification = F.binary_cross_entropy_with_logits(output["reconstructed_logits"], targets)
            loss = (
                contrastive
                + float(cfg.get("reconstruction_weight", 0.2)) * reconstruction
                + float(cfg.get("classification_weight", 0.5)) * classification
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
        batch_size=int(training.get("eval_batch_size", 64)),
    )
    attached = attach_comal_outputs(
        comal,
        outputs,
        batch_size=int(training.get("eval_batch_size", 64)),
    )
    similarities = attached["prototype_similarities"]
    if multi_view:
        view_own = similarities[..., 0]
        own = view_own[:, -1]
    else:
        view_own = None
        own = similarities[..., 0]
    return comal, history, outputs, own, view_own


def train_round(
    store: BrsetFeatureStore,
    labeled_patients: Iterable[str],
    config: dict[str, Any],
    device: torch.device,
) -> TrainedBrsetRound:
    patients = sorted(set(str(patient) for patient in labeled_patients))
    indices = store.indices_for_patients(patients)
    if not patients or not indices.size:
        raise ValueError("at least one labeled BRSET patient is required")
    training = config.get("training", {})
    model_seed = int(config.get("model", {}).get("seed", 1337))
    torch.manual_seed(model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(model_seed)
    classifier = build_classifier(store, config, device)
    loader = store.make_loader(
        indices,
        train=True,
        batch_size=int(training["batch_size"]),
        shuffle=True,
    )
    optimizer = Adam(
        classifier.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    history: dict[str, list[float]] = {"classifier_loss": [], "comal_loss": []}
    timings: dict[str, float] = {}
    start = time.perf_counter()
    classifier.train()
    for _epoch in range(int(training["epochs"])):
        losses: list[torch.Tensor] = []
        for host_batch in loader:
            batch = _move_batch(host_batch, device)
            optimizer.zero_grad(set_to_none=True)
            output = classifier(batch)
            loss = F.binary_cross_entropy_with_logits(output["logits"], batch["labels"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), float(training["gradient_clip"]))
            optimizer.step()
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
    elif strategy in {"modis", "mosaic"}:
        labeled_outputs = collect_outputs(classifier, store, indices, config, device, return_tokens=True)
        if strategy == "modis":
            start = time.perf_counter()
            from modis.probes import train_modality_probes

            groups = [
                str(store.patient_ids_array[index])
                for index in labeled_outputs["indices"].detach().cpu().numpy().astype(np.int64)
            ]
            modis_state = train_modality_probes(
                labeled_outputs["modality_tokens"],
                labeled_outputs["labels"],
                groups,
                config,
                seed=int(training.get("seed", 17)),
            )
            history["probe_loss"] = modis_state.history
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timings["modis_probe_training_sec"] = time.perf_counter() - start
    return TrainedBrsetRound(
        classifier=classifier,
        comal=comal,
        history=history,
        timings=timings,
        labeled_outputs=labeled_outputs,
        labeled_own_similarity=labeled_own,
        labeled_view_own_similarity=labeled_view_own,
        modis_state=modis_state,
    )
