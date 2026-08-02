"""Training and inference loops for cached MIMIC-III text features."""

from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Iterable, Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from .data import MIMICRecord
from .metrics import multilabel_metrics
from .model import CoMALModule, TextMLPClassifier, supervised_contrastive_loss


def label_matrix(records: list[MIMICRecord], label_names: tuple[str, ...]) -> np.ndarray:
    positions = {label: index for index, label in enumerate(label_names)}
    values = np.zeros((len(records), len(label_names)), dtype=np.float32)
    for row, record in enumerate(records):
        for label in record.labels:
            if label in positions:
                values[row, positions[label]] = 1.0
    return values


class CachedFeatureDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, features: np.ndarray, labels: np.ndarray, indices: Iterable[int]) -> None:
        self.features = features
        self.labels = labels
        self.indices = np.asarray(list(indices), dtype=np.int64)

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        index = int(self.indices[item])
        # Explicit copies avoid read-only mmap tensors and enable pinned transfer.
        return {
            "features": torch.from_numpy(np.asarray(self.features[index], dtype=np.float32).copy()),
            "labels": torch.from_numpy(self.labels[index].copy()),
            "index": torch.tensor(index, dtype=torch.long),
        }


def build_loader(
    features: np.ndarray,
    labels: np.ndarray,
    indices: Iterable[int],
    *,
    batch_size: int,
    shuffle: bool,
    training: dict[str, Any],
) -> DataLoader[dict[str, torch.Tensor]]:
    workers = int(training.get("num_workers", 0))
    options: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": workers,
        "pin_memory": bool(training.get("pin_memory", torch.cuda.is_available())),
        "drop_last": False,
    }
    if workers > 0:
        options["persistent_workers"] = bool(training.get("persistent_workers", True))
        options["prefetch_factor"] = int(training.get("prefetch_factor", 4))
    return DataLoader(CachedFeatureDataset(features, labels, indices), **options)


def _use_gpu_resident(device: torch.device, training: dict[str, Any]) -> bool:
    return device.type == "cuda" and bool(training.get("gpu_resident_features", True))


_DEVICE_MATRIX_CACHE: dict[tuple[int, int, str, str], torch.Tensor] = {}


def _to_device_matrix(
    values: np.ndarray, device: torch.device, *, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    cache_key = (id(values), values.shape[0], str(device), str(dtype))
    cached = _DEVICE_MATRIX_CACHE.get(cache_key)
    if cached is not None and cached.device == device:
        return cached
    array = np.asarray(values)
    if array.dtype != np.float32:
        array = array.astype(np.float32, copy=False)
    tensor = torch.from_numpy(np.ascontiguousarray(array))
    tensor = tensor.to(device=device, dtype=dtype, non_blocking=True)
    if device.type == "cuda":
        torch.cuda.current_stream().synchronize()
        _DEVICE_MATRIX_CACHE[cache_key] = tensor
    return tensor


def _index_batches(
    indices: torch.Tensor, batch_size: int, *, shuffle: bool, generator: torch.Generator | None = None
) -> Iterator[torch.Tensor]:
    if indices.numel() == 0:
        return
    order = indices
    if shuffle:
        perm = torch.randperm(indices.numel(), device=indices.device, generator=generator)
        order = indices[perm]
    for start in range(0, int(order.numel()), batch_size):
        yield order[start : start + batch_size]


def _autocast(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _pos_weight(labels: np.ndarray | torch.Tensor, indices: np.ndarray | torch.Tensor, maximum: float) -> torch.Tensor:
    if isinstance(labels, torch.Tensor):
        selected = labels.index_select(0, indices if isinstance(indices, torch.Tensor) else torch.as_tensor(indices))
        positives = selected.sum(dim=0)
        negatives = selected.shape[0] - positives
        weights = negatives / positives.clamp_min(1.0)
        return weights.clamp(1.0, maximum).to(dtype=torch.float32)
    selected = labels[np.asarray(indices)]
    positives = selected.sum(axis=0)
    negatives = len(selected) - positives
    return torch.from_numpy(np.clip(negatives / np.maximum(positives, 1.0), 1.0, maximum).astype(np.float32))


@dataclass
class TrainedRound:
    classifier: TextMLPClassifier
    comal: CoMALModule
    history: dict[str, list[float]]
    timings: dict[str, float]


def build_modules(
    input_dim: int, num_labels: int, config: dict[str, Any], device: torch.device
) -> tuple[TextMLPClassifier, CoMALModule]:
    model_cfg = config.get("model", {})
    comal_cfg = config.get("comal", {})
    hidden = tuple(int(value) for value in model_cfg.get("hidden_dims", [512, 256]))
    if len(hidden) != 2:
        raise ValueError("model.hidden_dims must contain exactly two dimensions")
    classifier = TextMLPClassifier(input_dim, num_labels, hidden, float(model_cfg.get("dropout", 0.2))).to(
        device
    )
    comal = CoMALModule(
        classifier.feature_dim,
        num_labels,
        int(comal_cfg.get("label_dim", 64)),
        int(comal_cfg.get("prototype_dim", 64)),
    ).to(device)
    return classifier, comal


def train_round(
    features: np.ndarray,
    labels: np.ndarray,
    labeled_indices: Iterable[int],
    config: dict[str, Any],
    device: torch.device,
    *,
    previous: TrainedRound | None = None,
) -> TrainedRound:
    training = config.get("training", {})
    indices = np.asarray(sorted(set(int(value) for value in labeled_indices)), dtype=np.int64)
    if not indices.size:
        raise ValueError("at least one labeled sample is required")
    classifier, comal = build_modules(features.shape[1], labels.shape[1], config, device)
    if previous is not None and bool(training.get("inherit_across_rounds", False)):
        classifier.load_state_dict(previous.classifier.state_dict())
        comal.load_state_dict(previous.comal.state_dict())
    precision = str(training.get("precision", "bf16" if device.type == "cuda" else "fp32")).lower()
    if precision not in {"bf16", "fp16", "fp32"}:
        raise ValueError("training.precision must be bf16, fp16, or fp32")

    resident = _use_gpu_resident(device, training)
    if resident:
        feature_tensor = _to_device_matrix(features, device)
        label_tensor = _to_device_matrix(labels, device)
        index_tensor = torch.as_tensor(indices, device=device, dtype=torch.long)
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=_pos_weight(label_tensor, index_tensor, float(training.get("maximum_pos_weight", 20)))
        )
    else:
        feature_tensor = None
        label_tensor = None
        index_tensor = None
        criterion = nn.BCEWithLogitsLoss(
            pos_weight=_pos_weight(labels, indices, float(training.get("maximum_pos_weight", 20))).to(device)
        )

    optimizer = AdamW(
        classifier.parameters(),
        lr=float(training.get("learning_rate", 1e-3)),
        weight_decay=float(training.get("weight_decay", 1e-4)),
    )
    history: dict[str, list[float]] = {"classifier_loss": [], "comal_loss": []}
    timings: dict[str, float] = {}
    batch_size = int(training.get("batch_size", 512))
    start = time.perf_counter()
    classifier.train()
    if resident:
        assert feature_tensor is not None and label_tensor is not None and index_tensor is not None
        for _ in range(int(training.get("epochs", 20))):
            losses: list[float] = []
            for batch_index in _index_batches(index_tensor, batch_size, shuffle=True):
                inputs = feature_tensor.index_select(0, batch_index)
                targets = label_tensor.index_select(0, batch_index)
                optimizer.zero_grad(set_to_none=True)
                with _autocast(device, precision):
                    output = classifier(inputs)
                    loss = criterion(output["logits"], targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), float(training.get("gradient_clip", 5.0)))
                optimizer.step()
                losses.append(float(loss.detach()))
            history["classifier_loss"].append(float(np.mean(losses)))
    else:
        classifier_loader = build_loader(
            features,
            labels,
            indices,
            batch_size=batch_size,
            shuffle=True,
            training=training,
        )
        for _ in range(int(training.get("epochs", 20))):
            losses = []
            for batch in classifier_loader:
                inputs = batch["features"].to(device, non_blocking=True)
                targets = batch["labels"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with _autocast(device, precision):
                    output = classifier(inputs)
                    loss = criterion(output["logits"], targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), float(training.get("gradient_clip", 5.0)))
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            history["classifier_loss"].append(float(np.mean(losses)))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    timings["classifier_training_sec"] = time.perf_counter() - start

    # The original implementation freezes the classifier before training CoMAL.
    classifier.eval()
    for parameter in classifier.parameters():
        parameter.requires_grad_(False)
    comal_cfg = config.get("comal", {})
    optimizer_comal = AdamW(
        comal.parameters(),
        lr=float(comal_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(training.get("weight_decay", 1e-4)),
    )
    comal_batch_size = int(training.get("comal_batch_size", 32))
    start = time.perf_counter()
    comal.train()
    if resident:
        assert feature_tensor is not None and label_tensor is not None and index_tensor is not None
        for _ in range(int(training.get("comal_epochs", 10))):
            losses = []
            for batch_index in _index_batches(index_tensor, comal_batch_size, shuffle=True):
                inputs = feature_tensor.index_select(0, batch_index)
                targets = label_tensor.index_select(0, batch_index)
                optimizer_comal.zero_grad(set_to_none=True)
                with torch.no_grad():
                    classifier_output = classifier(inputs)
                with _autocast(device, precision):
                    output = comal(classifier_output["features"].detach())
                    contrastive = supervised_contrastive_loss(
                        output["latent_features"],
                        targets,
                        temperature=float(comal_cfg.get("temperature", 0.07)),
                        anchor_chunk_size=int(comal_cfg.get("anchor_chunk_size", 1024)),
                    )
                    reconstruction = F.mse_loss(
                        output["reconstructed_features"], classifier_output["features"].detach()
                    )
                    reconstruction_bce = criterion(output["reconstructed_logits"], targets)
                    loss = (
                        contrastive
                        + float(comal_cfg.get("reconstruction_weight", 0.2)) * reconstruction
                        + float(comal_cfg.get("classification_weight", 0.5)) * reconstruction_bce
                    )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(comal.parameters(), float(training.get("gradient_clip", 5.0)))
                optimizer_comal.step()
                losses.append(float(loss.detach()))
            history["comal_loss"].append(float(np.mean(losses)))
    else:
        comal_loader = build_loader(
            features,
            labels,
            indices,
            batch_size=comal_batch_size,
            shuffle=True,
            training=training,
        )
        for _ in range(int(training.get("comal_epochs", 10))):
            losses = []
            for batch in comal_loader:
                inputs = batch["features"].to(device, non_blocking=True)
                targets = batch["labels"].to(device, non_blocking=True)
                optimizer_comal.zero_grad(set_to_none=True)
                with torch.no_grad():
                    classifier_output = classifier(inputs)
                with _autocast(device, precision):
                    output = comal(classifier_output["features"].detach())
                    contrastive = supervised_contrastive_loss(
                        output["latent_features"],
                        targets,
                        temperature=float(comal_cfg.get("temperature", 0.07)),
                        anchor_chunk_size=int(comal_cfg.get("anchor_chunk_size", 1024)),
                    )
                    reconstruction = F.mse_loss(
                        output["reconstructed_features"], classifier_output["features"].detach()
                    )
                    reconstruction_bce = criterion(output["reconstructed_logits"], targets)
                    loss = (
                        contrastive
                        + float(comal_cfg.get("reconstruction_weight", 0.2)) * reconstruction
                        + float(comal_cfg.get("classification_weight", 0.5)) * reconstruction_bce
                    )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(comal.parameters(), float(training.get("gradient_clip", 5.0)))
                optimizer_comal.step()
                losses.append(float(loss.detach().cpu()))
            history["comal_loss"].append(float(np.mean(losses)))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    timings["comal_training_sec"] = time.perf_counter() - start
    if resident:
        assert feature_tensor is not None and label_tensor is not None
        _refresh_prototypes_tensors(
            classifier,
            comal,
            feature_tensor,
            label_tensor,
            torch.as_tensor(indices, device=device, dtype=torch.long),
            int(training.get("eval_batch_size", 1024)),
        )
    else:
        refresh_prototypes(classifier, comal, features, labels, indices, config, device)
    return TrainedRound(classifier, comal, history, timings)


@torch.inference_mode()
def _refresh_prototypes_tensors(
    classifier: TextMLPClassifier,
    comal: CoMALModule,
    feature_tensor: torch.Tensor,
    label_tensor: torch.Tensor,
    index_tensor: torch.Tensor,
    eval_batch_size: int,
) -> None:
    sums = torch.zeros_like(comal.prototypes, dtype=torch.float32, device=feature_tensor.device)
    counts = torch.zeros_like(comal.prototype_counts, dtype=torch.float32, device=feature_tensor.device)
    classifier.eval()
    comal.eval()
    for batch_index in _index_batches(index_tensor, eval_batch_size, shuffle=False):
        inputs = feature_tensor.index_select(0, batch_index)
        targets = label_tensor.index_select(0, batch_index)
        latent = F.normalize(comal(classifier(inputs)["features"])["latent_features"].float(), dim=-1)
        sums[:-1] += torch.einsum("bl,bld->ld", targets, latent)
        counts[:-1] += targets.sum(dim=0)
        negative = 1.0 - targets
        sums[-1] += (latent * negative[..., None]).sum(dim=(0, 1))
        counts[-1] += negative.sum()
    comal.set_prototypes(sums, counts)


@torch.inference_mode()
def refresh_prototypes(
    classifier: TextMLPClassifier,
    comal: CoMALModule,
    features: np.ndarray,
    labels: np.ndarray,
    indices: Iterable[int],
    config: dict[str, Any],
    device: torch.device,
) -> None:
    training = config.get("training", {})
    index_array = np.asarray(list(indices), dtype=np.int64)
    if _use_gpu_resident(device, training):
        _refresh_prototypes_tensors(
            classifier,
            comal,
            _to_device_matrix(features, device),
            _to_device_matrix(labels, device),
            torch.as_tensor(index_array, device=device, dtype=torch.long),
            int(training.get("eval_batch_size", 1024)),
        )
        return
    sums = torch.zeros_like(comal.prototypes, dtype=torch.float32, device=device)
    counts = torch.zeros_like(comal.prototype_counts, dtype=torch.float32, device=device)
    classifier.eval()
    comal.eval()
    loader = build_loader(
        features,
        labels,
        index_array,
        batch_size=int(training.get("eval_batch_size", 1024)),
        shuffle=False,
        training=training,
    )
    for batch in loader:
        inputs = batch["features"].to(device, non_blocking=True)
        targets = batch["labels"].to(device, non_blocking=True)
        latent = F.normalize(comal(classifier(inputs)["features"])["latent_features"].float(), dim=-1)
        sums[:-1] += torch.einsum("bl,bld->ld", targets, latent)
        counts[:-1] += targets.sum(dim=0)
        negative = 1.0 - targets
        sums[-1] += (latent * negative[..., None]).sum(dim=(0, 1))
        counts[-1] += negative.sum()
    comal.set_prototypes(sums, counts)


@torch.inference_mode()
def predict(
    trained: TrainedRound,
    features: np.ndarray,
    labels: np.ndarray,
    indices: Iterable[int],
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, np.ndarray]:
    training = config.get("training", {})
    index_array = np.asarray(list(indices), dtype=np.int64)
    trained.classifier.eval()
    trained.comal.eval()
    values: dict[str, list[np.ndarray]] = {"indices": [], "labels": [], "probabilities": [], "latents": []}
    if _use_gpu_resident(device, training):
        feature_tensor = _to_device_matrix(features, device)
        label_tensor = _to_device_matrix(labels, device)
        index_tensor = torch.as_tensor(index_array, device=device, dtype=torch.long)
        for batch_index in _index_batches(
            index_tensor, int(training.get("eval_batch_size", 1024)), shuffle=False
        ):
            inputs = feature_tensor.index_select(0, batch_index)
            output = trained.classifier(inputs)
            comal_output = trained.comal(output["features"])
            values["indices"].append(batch_index.detach().cpu().numpy())
            values["labels"].append(label_tensor.index_select(0, batch_index).detach().cpu().numpy())
            values["probabilities"].append(torch.sigmoid(output["logits"]).detach().cpu().numpy())
            values["latents"].append(comal_output["latent_features"].float().detach().cpu().numpy())
    else:
        loader = build_loader(
            features,
            labels,
            index_array,
            batch_size=int(training.get("eval_batch_size", 1024)),
            shuffle=False,
            training=training,
        )
        for batch in loader:
            inputs = batch["features"].to(device, non_blocking=True)
            output = trained.classifier(inputs)
            comal_output = trained.comal(output["features"])
            values["indices"].append(batch["index"].numpy())
            values["labels"].append(batch["labels"].numpy())
            values["probabilities"].append(torch.sigmoid(output["logits"]).cpu().numpy())
            values["latents"].append(comal_output["latent_features"].float().cpu().numpy())
    return {name: np.concatenate(parts, axis=0) if parts else np.empty(0) for name, parts in values.items()}


def evaluate(
    trained: TrainedRound,
    features: np.ndarray,
    labels: np.ndarray,
    indices: Iterable[int],
    config: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    predictions = predict(trained, features, labels, indices, config, device)
    metrics = multilabel_metrics(
        predictions["labels"],
        predictions["probabilities"],
        float(config.get("training", {}).get("threshold", 0.5)),
    )
    return metrics, predictions
