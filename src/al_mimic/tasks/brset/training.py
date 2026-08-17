"""Image-level BRSET classifier training and patient-level method adapters."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from .data import BrsetFeatureStore
from .model import BrsetMultimodalClassifier, initialize_fusion_layers


@dataclass
class TrainedBrsetRound:
    """Classifier state for one cold-start active-learning round."""

    classifier: BrsetMultimodalClassifier
    history: dict[str, list[float]]
    timings: dict[str, float]
    method_state: Any | None = None


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=device.type == "cuda") for key, value in batch.items()}


def build_classifier(
    store: BrsetFeatureStore,
    config: dict[str, Any],
    device: torch.device,
) -> BrsetMultimodalClassifier:
    model_config = config.get("model", {})
    model = BrsetMultimodalClassifier(
        store.schema.dimension,
        num_labels=int(model_config["output_size"]),
        image_feature_dim=int(model_config["image_feature_dim"]),
        metadata_hidden_dim=int(model_config["metadata_hidden_dim"]),
        fusion_dim=int(model_config["fusion_dim"]),
        dropout=float(model_config.get("dropout", 0.2)),
        image_weights=str(model_config["image_weights"]),
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
    """Collect canonical classifier outputs for image rows."""
    selected = np.asarray(indices, dtype=np.int64)
    training = config.get("training", {})
    loader = store.make_loader(
        selected,
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
            collected[name].append(source.detach() if name == "indices" else source.detach().float())
    if not selected.size:
        result = {
            "indices": torch.empty(0, dtype=torch.long, device=device),
            "labels": torch.empty((0, classifier.num_labels), dtype=torch.float32, device=device),
            "probabilities": torch.empty((0, classifier.num_labels), dtype=torch.float32, device=device),
            "features": torch.empty((0, classifier.feature_dim), dtype=torch.float32, device=device),
        }
        if return_tokens:
            result["modality_tokens"] = torch.empty(
                (0, len(classifier.modality_names), classifier.feature_dim),
                dtype=torch.float32,
                device=device,
            )
        return result
    return {name: torch.cat(values, dim=0) for name, values in collected.items()}


def _patient_order(
    row_patient_ids: Sequence[str],
    patient_ids: Sequence[str] | None,
) -> tuple[tuple[str, ...], torch.Tensor]:
    rows = tuple(str(patient_id) for patient_id in row_patient_ids)
    if not rows:
        raise ValueError("patient aggregation requires at least one image row")
    ordered = (
        tuple(sorted(set(rows)))
        if patient_ids is None
        else tuple(str(patient_id) for patient_id in patient_ids)
    )
    if len(set(ordered)) != len(ordered):
        raise ValueError("patient aggregation order must contain unique patient IDs")
    if set(rows) != set(ordered):
        raise ValueError("patient aggregation order must match the image-row patient IDs exactly")
    positions = {patient_id: position for position, patient_id in enumerate(ordered)}
    inverse = torch.as_tensor([positions[patient_id] for patient_id in rows], dtype=torch.long)
    return ordered, inverse


@torch.inference_mode()
def aggregate_patient_outputs(
    classifier: nn.Module,
    outputs: dict[str, torch.Tensor],
    row_patient_ids: Sequence[str],
    *,
    patient_ids: Sequence[str] | None = None,
) -> tuple[dict[str, torch.Tensor], tuple[str, ...]]:
    """Aggregate image rows into the canonical patient query representation.

    Image and metadata tokens are averaged independently per patient, then fused
    once by the classifier. Disease targets use the union over that patient's
    images. No image-level probabilities, prototypes, or probe rows cross this
    method boundary.
    """
    required = {"labels", "modality_tokens"}
    missing = sorted(required - outputs.keys())
    if missing:
        raise ValueError(f"patient aggregation requires output fields: {missing}")
    tokens = outputs["modality_tokens"].float()
    labels = outputs["labels"].float()
    if tokens.ndim != 3 or labels.ndim != 2 or tokens.shape[0] != labels.shape[0]:
        raise ValueError("patient aggregation expects row-aligned tokens [N,M,D] and labels [N,C]")
    if len(row_patient_ids) != tokens.shape[0]:
        raise ValueError("row_patient_ids must align with image outputs")
    if not callable(getattr(classifier, "fuse_from_tokens", None)) or not callable(
        getattr(classifier, "probabilities_from_fused", None)
    ):
        raise ValueError("patient aggregation requires classifier fusion and probability hooks")

    ordered, host_inverse = _patient_order(row_patient_ids, patient_ids)
    inverse = host_inverse.to(tokens.device)
    counts = torch.bincount(inverse, minlength=len(ordered)).to(dtype=torch.float32)
    grouped_tokens = torch.zeros(
        (len(ordered), tokens.shape[1], tokens.shape[2]),
        dtype=torch.float32,
        device=tokens.device,
    )
    grouped_tokens.index_add_(0, inverse, tokens)
    grouped_tokens /= counts[:, None, None].clamp_min(1.0)

    grouped_labels = torch.zeros((len(ordered), labels.shape[1]), dtype=torch.float32, device=labels.device)
    grouped_labels.index_add_(0, inverse.to(labels.device), labels)
    grouped_labels.clamp_(0.0, 1.0)
    features = classifier.fuse_from_tokens(grouped_tokens)
    probabilities = classifier.probabilities_from_fused(features)
    return {
        "labels": grouped_labels,
        "modality_tokens": grouped_tokens,
        "features": features.float(),
        "probabilities": probabilities.float(),
        "patient_positions": torch.arange(len(ordered), dtype=torch.long, device=tokens.device),
        "image_counts": counts.to(dtype=torch.long),
    }, ordered


@torch.inference_mode()
def collect_patient_outputs(
    classifier: BrsetMultimodalClassifier,
    store: BrsetFeatureStore,
    patient_ids: Sequence[str],
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Collect image outputs and aggregate them in the requested patient order."""
    ordered = tuple(str(patient_id) for patient_id in patient_ids)
    indices = store.indices_for_patients(ordered)
    image_outputs = collect_outputs(
        classifier,
        store,
        indices,
        config,
        device,
        return_tokens=True,
    )
    global_indices = image_outputs["indices"].detach().cpu().numpy().astype(np.int64)
    row_patient_ids = tuple(str(store.patient_ids_array[index]) for index in global_indices)
    patient_outputs, actual_order = aggregate_patient_outputs(
        classifier,
        image_outputs,
        row_patient_ids,
        patient_ids=ordered,
    )
    if actual_order != ordered:
        raise RuntimeError("patient output aggregation changed the requested ordering")
    return patient_outputs


def train_round(
    store: BrsetFeatureStore,
    labeled_patients: Iterable[str],
    config: dict[str, Any],
    device: torch.device,
) -> TrainedBrsetRound:
    """Train a fresh image-level classifier on every image of labeled patients."""
    patients = tuple(sorted(set(str(patient) for patient in labeled_patients)))
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
    history: dict[str, list[float]] = {"classifier_loss": []}
    start = time.perf_counter()
    classifier.train()
    for _epoch in range(int(training["epochs"])):
        losses: list[torch.Tensor] = []
        for host_batch in loader:
            batch = _move_batch(host_batch, device)
            optimizer.zero_grad(set_to_none=True)
            output = classifier(batch)
            loss = F.binary_cross_entropy_with_logits(output["logits"], batch["labels"].float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), float(training["gradient_clip"]))
            optimizer.step()
            losses.append(loss.detach())
        if not losses:
            raise RuntimeError("BRSET classifier training produced no batches")
        history["classifier_loss"].append(float(torch.stack(losses).mean().cpu()))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return TrainedBrsetRound(
        classifier=classifier,
        history=history,
        timings={"classifier_training_sec": time.perf_counter() - start},
    )


__all__ = [
    "TrainedBrsetRound",
    "aggregate_patient_outputs",
    "build_classifier",
    "collect_outputs",
    "collect_patient_outputs",
    "train_round",
]
