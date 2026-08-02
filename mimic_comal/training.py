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
    rows: list[int] = []
    cols: list[int] = []
    for row, record in enumerate(records):
        for label in record.labels:
            column = positions.get(label)
            if column is not None:
                rows.append(row)
                cols.append(column)
    if rows:
        values[np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)] = 1.0
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


def clear_device_matrix_cache() -> None:
    _DEVICE_MATRIX_CACHE.clear()


def warm_resident_matrices(
    features: np.ndarray, labels: np.ndarray, device: torch.device, training: dict[str, Any]
) -> None:
    """Upload feature/label tables once before the first timed AL round."""
    if not _use_gpu_resident(device, training):
        return
    _to_device_matrix(features, device)
    _to_device_matrix(labels, device)


def _to_device_matrix(
    values: np.ndarray, device: torch.device, *, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    cache_key = (id(values), int(values.shape[0]), str(device), str(dtype))
    cached = _DEVICE_MATRIX_CACHE.get(cache_key)
    if cached is not None and cached.device == device and cached.shape[0] == values.shape[0]:
        return cached
    array = np.asarray(values)
    # Keep native dtype on host (e.g. float16 BERT caches); cast on the device copy.
    if not array.flags["C_CONTIGUOUS"]:
        array = np.ascontiguousarray(array)
    host = torch.from_numpy(array)
    if device.type == "cuda":
        if not host.is_pinned():
            host = host.pin_memory()
        tensor = host.to(device=device, dtype=dtype, non_blocking=True)
        torch.cuda.current_stream().synchronize()
        _DEVICE_MATRIX_CACHE[cache_key] = tensor
        return tensor
    return host.to(dtype=dtype)


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


def _shuffled_slices(
    features: torch.Tensor,
    labels: torch.Tensor,
    batch_size: int,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Shuffle via index batches without cloning the full feature/label tables."""
    count = int(features.shape[0])
    if count == 0:
        return
    if batch_size >= count:
        yield features, labels
        return
    perm = torch.randperm(count, device=features.device)
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        batch_index = perm[start:stop]
        yield features.index_select(0, batch_index), labels.index_select(0, batch_index)


def _autocast(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _pos_weight(labels: np.ndarray | torch.Tensor, indices: np.ndarray | torch.Tensor, maximum: float) -> torch.Tensor:
    if isinstance(labels, torch.Tensor):
        index_tensor = indices if isinstance(indices, torch.Tensor) else torch.as_tensor(indices, device=labels.device)
        selected = labels.index_select(0, index_tensor)
        positives = selected.sum(dim=0)
        negatives = selected.shape[0] - positives
        weights = negatives / positives.clamp_min(1.0)
        return weights.clamp(1.0, maximum).to(dtype=torch.float32)
    selected = labels[np.asarray(indices)]
    positives = selected.sum(axis=0)
    negatives = len(selected) - positives
    return torch.from_numpy(np.clip(negatives / np.maximum(positives, 1.0), 1.0, maximum).astype(np.float32))


def _maybe_compile(module: nn.Module, enabled: bool) -> nn.Module:
    if not enabled or not hasattr(torch, "compile"):
        return module
    # Prefer dynamic shapes: AL labeled/candidate widths change every round.
    for kwargs in (
        {"mode": "max-autotune-no-cudagraphs", "dynamic": True},
        {"mode": "default", "dynamic": True},
        {"mode": "reduce-overhead"},
    ):
        try:
            return torch.compile(module, **kwargs)  # type: ignore[return-value]
        except Exception:
            continue
    return module


@dataclass
class TrainedRound:
    classifier: TextMLPClassifier
    comal: CoMALModule
    history: dict[str, list[float]]
    timings: dict[str, float]
    # Optional resident labeled-pool caches for paper acquisition (avoids a re-encode).
    labeled_labels: torch.Tensor | None = None
    labeled_own_similarity: torch.Tensor | None = None


def build_modules(
    input_dim: int, num_labels: int, config: dict[str, Any], device: torch.device
) -> tuple[TextMLPClassifier, CoMALModule]:
    model_cfg = config.get("model", {})
    comal_cfg = config.get("comal", {})
    training = config.get("training", {})
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
    if device.type == "cuda" and bool(training.get("torch_compile", False)):
        classifier = _maybe_compile(classifier, True)  # type: ignore[assignment]
        comal = _maybe_compile(comal, True)  # type: ignore[assignment]
    return classifier, comal


@torch.inference_mode()
def _cache_classifier_features(
    classifier: TextMLPClassifier,
    feature_tensor: torch.Tensor,
    index_tensor: torch.Tensor | None,
    eval_batch_size: int,
) -> torch.Tensor:
    """Frozen-classifier features for the labeled pool; avoids re-encoding each CoMAL step."""
    classifier.eval()
    if index_tensor is None:
        count = int(feature_tensor.shape[0])
        if count == 0:
            return feature_tensor.new_zeros((0, classifier.feature_dim))
        if count <= eval_batch_size:
            return classifier(feature_tensor)["features"]
        out = feature_tensor.new_empty((count, classifier.feature_dim))
        for start in range(0, count, eval_batch_size):
            stop = min(start + eval_batch_size, count)
            out[start:stop] = classifier(feature_tensor[start:stop])["features"]
        return out
    count = int(index_tensor.numel())
    if count == 0:
        return feature_tensor.new_zeros((0, classifier.feature_dim))
    if count <= eval_batch_size:
        return classifier(feature_tensor.index_select(0, index_tensor))["features"]
    out = feature_tensor.new_empty((count, classifier.feature_dim))
    cursor = 0
    for batch_index in _index_batches(index_tensor, eval_batch_size, shuffle=False):
        width = int(batch_index.numel())
        out[cursor : cursor + width] = classifier(feature_tensor.index_select(0, batch_index))["features"]
        cursor += width
    return out


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
    indices = np.unique(np.asarray(list(labeled_indices), dtype=np.int64))
    if not indices.size:
        raise ValueError("at least one labeled sample is required")
    inherit = previous is not None and bool(training.get("inherit_across_rounds", False))
    if inherit:
        # Reuse live modules: avoids rebuild + state_dict clone on every AL round.
        assert previous is not None
        classifier = previous.classifier
        comal = previous.comal
        for parameter in classifier.parameters():
            parameter.requires_grad_(True)
        for parameter in comal.parameters():
            parameter.requires_grad_(True)
    else:
        classifier, comal = build_modules(features.shape[1], labels.shape[1], config, device)
    precision = str(training.get("precision", "bf16" if device.type == "cuda" else "fp32")).lower()
    if precision not in {"bf16", "fp16", "fp32"}:
        raise ValueError("training.precision must be bf16, fp16, or fp32")

    resident = _use_gpu_resident(device, training)
    labeled_features: torch.Tensor | None = None
    labeled_targets: torch.Tensor | None = None
    if resident:
        feature_tensor = _to_device_matrix(features, device)
        label_tensor = _to_device_matrix(labels, device)
        index_tensor = torch.as_tensor(indices, device=device, dtype=torch.long)
        labeled_features = feature_tensor.index_select(0, index_tensor).contiguous()
        labeled_targets = label_tensor.index_select(0, index_tensor).contiguous()
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

    optimizer_kwargs: dict[str, Any] = {
        "lr": float(training.get("learning_rate", 1e-3)),
        "weight_decay": float(training.get("weight_decay", 1e-4)),
    }
    if device.type == "cuda" and bool(training.get("fused_optimizer", True)):
        optimizer_kwargs["fused"] = True
    else:
        optimizer_kwargs["foreach"] = True
    try:
        optimizer = AdamW(classifier.parameters(), **optimizer_kwargs)
    except (TypeError, RuntimeError):
        optimizer_kwargs.pop("fused", None)
        optimizer_kwargs.pop("foreach", None)
        optimizer = AdamW(classifier.parameters(), **optimizer_kwargs)
    history: dict[str, list[float]] = {"classifier_loss": [], "comal_loss": []}
    timings: dict[str, float] = {}
    batch_size = int(training.get("batch_size", 512))
    grad_clip = float(training.get("gradient_clip", 5.0))
    # Clipping every step forces a device sync; interval>1 amortizes that cost.
    grad_clip_interval = max(1, int(training.get("gradient_clip_interval", 1)))
    start = time.perf_counter()
    classifier.train()
    classifier_epoch_losses: list[torch.Tensor] = []
    global_step = 0
    if resident:
        assert labeled_features is not None and labeled_targets is not None
        for _ in range(int(training.get("epochs", 20))):
            epoch_loss = labeled_features.new_zeros(())
            steps = 0
            for inputs, targets in _shuffled_slices(labeled_features, labeled_targets, batch_size):
                optimizer.zero_grad(set_to_none=True)
                with _autocast(device, precision):
                    output = classifier(inputs)
                    loss = criterion(output["logits"], targets)
                loss.backward()
                global_step += 1
                if grad_clip > 0 and global_step % grad_clip_interval == 0:
                    torch.nn.utils.clip_grad_norm_(classifier.parameters(), grad_clip)
                optimizer.step()
                epoch_loss = epoch_loss + loss.detach()
                steps += 1
            # Defer .item() sync until after the phase barrier.
            classifier_epoch_losses.append(epoch_loss / max(steps, 1))
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
                global_step += 1
                if grad_clip > 0 and global_step % grad_clip_interval == 0:
                    torch.nn.utils.clip_grad_norm_(classifier.parameters(), grad_clip)
                optimizer.step()
                losses.append(loss.detach())
            classifier_epoch_losses.append(torch.stack(losses).mean() if losses else torch.zeros(()))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    timings["classifier_training_sec"] = time.perf_counter() - start
    if classifier_epoch_losses:
        history["classifier_loss"] = torch.stack(classifier_epoch_losses).detach().cpu().tolist()

    # The original implementation freezes the classifier before training CoMAL.
    classifier.eval()
    for parameter in classifier.parameters():
        parameter.requires_grad_(False)
    comal_cfg = config.get("comal", {})
    try:
        optimizer_comal = AdamW(
            comal.parameters(),
            lr=float(comal_cfg.get("learning_rate", 1e-3)),
            weight_decay=float(training.get("weight_decay", 1e-4)),
            fused=bool(device.type == "cuda" and training.get("fused_optimizer", True)),
        )
    except TypeError:
        optimizer_comal = AdamW(
            comal.parameters(),
            lr=float(comal_cfg.get("learning_rate", 1e-3)),
            weight_decay=float(training.get("weight_decay", 1e-4)),
        )
    comal_batch_size = int(training.get("comal_batch_size", 32))
    start = time.perf_counter()
    comal.train()
    prototypes_from_cache = False
    if resident:
        assert labeled_features is not None and labeled_targets is not None
        # Classifier is frozen: encode labeled pool once, then train CoMAL on cached fused features.
        cached_features = _cache_classifier_features(
            classifier,
            labeled_features,
            None,
            int(training.get("eval_batch_size", 1024)),
        )
        cached_labels = labeled_targets
        recon_w = float(comal_cfg.get("reconstruction_weight", 0.2))
        clf_w = float(comal_cfg.get("classification_weight", 0.5))
        temperature = float(comal_cfg.get("temperature", 0.07))
        anchor_chunk = int(comal_cfg.get("anchor_chunk_size", 1024))
        comal_epoch_losses: list[torch.Tensor] = []
        comal_step = 0
        for _ in range(int(training.get("comal_epochs", 10))):
            epoch_loss = cached_features.new_zeros(())
            steps = 0
            for fused, targets in _shuffled_slices(cached_features, cached_labels, comal_batch_size):
                optimizer_comal.zero_grad(set_to_none=True)
                with _autocast(device, precision):
                    # Skip similarity matmul in the train loss path only; eval always computes it.
                    output = comal(fused, compute_similarities=False)
                    contrastive = supervised_contrastive_loss(
                        output["latent_features"],
                        targets,
                        temperature=temperature,
                        anchor_chunk_size=anchor_chunk,
                    )
                    reconstruction = F.mse_loss(output["reconstructed_features"], fused)
                    reconstruction_bce = criterion(output["reconstructed_logits"], targets)
                    loss = contrastive + recon_w * reconstruction + clf_w * reconstruction_bce
                loss.backward()
                comal_step += 1
                if grad_clip > 0 and comal_step % grad_clip_interval == 0:
                    torch.nn.utils.clip_grad_norm_(comal.parameters(), grad_clip)
                optimizer_comal.step()
                epoch_loss = epoch_loss + loss.detach()
                steps += 1
            comal_epoch_losses.append(epoch_loss / max(steps, 1))
        # Reuse frozen classifier features instead of re-encoding the labeled pool.
        prototypes_from_cache = True
        eval_batch = int(training.get("eval_batch_size", 1024))
        # One labeled pass: refresh prototypes and cache own-sims for paper acquisition.
        labeled_own_cache = _refresh_prototypes_from_cached(
            comal,
            cached_features,
            cached_labels,
            eval_batch,
            return_own_similarity=True,
        )
        labeled_labels_cache = cached_labels
    else:
        labeled_labels_cache = None
        labeled_own_cache = None
        comal_loader = build_loader(
            features,
            labels,
            indices,
            batch_size=comal_batch_size,
            shuffle=True,
            training=training,
        )
        comal_epoch_losses = []
        recon_w = float(comal_cfg.get("reconstruction_weight", 0.2))
        clf_w = float(comal_cfg.get("classification_weight", 0.5))
        temperature = float(comal_cfg.get("temperature", 0.07))
        anchor_chunk = int(comal_cfg.get("anchor_chunk_size", 1024))
        comal_step = 0
        for _ in range(int(training.get("comal_epochs", 10))):
            losses = []
            for batch in comal_loader:
                inputs = batch["features"].to(device, non_blocking=True)
                targets = batch["labels"].to(device, non_blocking=True)
                optimizer_comal.zero_grad(set_to_none=True)
                with torch.no_grad():
                    classifier_output = classifier(inputs)
                fused = classifier_output["features"].detach()
                with _autocast(device, precision):
                    output = comal(fused, compute_similarities=False)
                    contrastive = supervised_contrastive_loss(
                        output["latent_features"],
                        targets,
                        temperature=temperature,
                        anchor_chunk_size=anchor_chunk,
                    )
                    reconstruction = F.mse_loss(output["reconstructed_features"], fused)
                    reconstruction_bce = criterion(output["reconstructed_logits"], targets)
                    loss = contrastive + recon_w * reconstruction + clf_w * reconstruction_bce
                loss.backward()
                comal_step += 1
                if grad_clip > 0 and comal_step % grad_clip_interval == 0:
                    torch.nn.utils.clip_grad_norm_(comal.parameters(), grad_clip)
                optimizer_comal.step()
                losses.append(loss.detach())
            comal_epoch_losses.append(torch.stack(losses).mean() if losses else torch.zeros(()))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    timings["comal_training_sec"] = time.perf_counter() - start
    if comal_epoch_losses:
        history["comal_loss"] = torch.stack(comal_epoch_losses).detach().cpu().tolist()
    if not prototypes_from_cache:
        if resident:
            assert feature_tensor is not None and label_tensor is not None
            _refresh_prototypes_tensors(
                classifier,
                comal,
                feature_tensor,
                label_tensor,
                index_tensor if index_tensor is not None else torch.as_tensor(indices, device=device, dtype=torch.long),
                int(training.get("eval_batch_size", 1024)),
            )
        else:
            refresh_prototypes(classifier, comal, features, labels, indices, config, device)
    return TrainedRound(
        classifier,
        comal,
        history,
        timings,
        labeled_labels=labeled_labels_cache,
        labeled_own_similarity=labeled_own_cache,
    )


@torch.inference_mode()
def _refresh_prototypes_from_cached(
    comal: CoMALModule,
    cached_features: torch.Tensor,
    cached_labels: torch.Tensor,
    eval_batch_size: int,
    *,
    return_own_similarity: bool = False,
) -> torch.Tensor | None:
    """Refresh prototypes from frozen classifier features (no classifier re-encode)."""
    sums = torch.zeros_like(comal.prototypes, dtype=torch.float32, device=cached_features.device)
    counts = torch.zeros_like(comal.prototype_counts, dtype=torch.float32, device=cached_features.device)
    comal.eval()
    count = int(cached_features.shape[0])
    num_labels = int(cached_labels.shape[1])
    latents: torch.Tensor | None = None
    if return_own_similarity and count > 0:
        latents = cached_features.new_empty((count, num_labels, int(comal.prototype_dim)))
    for start in range(0, max(count, 1), eval_batch_size):
        if count == 0:
            break
        stop = min(start + eval_batch_size, count)
        fused = cached_features[start:stop]
        targets = cached_labels[start:stop]
        latent = F.normalize(
            comal(fused, compute_similarities=False, compute_reconstruction=False)[
                "latent_features"
            ].float(),
            dim=-1,
        )
        if latents is not None:
            latents[start:stop] = latent
        negative = 1.0 - targets
        sums[:-1] += torch.einsum("bl,bld->ld", targets, latent)
        counts[:-1] += targets.sum(dim=0)
        sums[-1] += torch.einsum("bl,bld->d", negative, latent)
        counts[-1] += negative.sum()
    comal.set_prototypes(sums, counts)
    if not return_own_similarity:
        return None
    if latents is None:
        return cached_features.new_zeros((0, num_labels))
    # Own-sims must use the refreshed prototypes, not the pre-update buffers.
    return torch.einsum("nld,ld->nl", latents, comal.prototypes[:-1].float())


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
    selected_features = feature_tensor.index_select(0, index_tensor)
    selected_labels = label_tensor.index_select(0, index_tensor)
    count = int(selected_features.shape[0])
    for start in range(0, max(count, 1), eval_batch_size):
        if count == 0:
            break
        stop = min(start + eval_batch_size, count)
        inputs = selected_features[start:stop]
        targets = selected_labels[start:stop]
        fused = classifier(inputs)["features"]
        latent = F.normalize(
            comal(fused, compute_similarities=False, compute_reconstruction=False)[
                "latent_features"
            ].float(),
            dim=-1,
        )
        negative = 1.0 - targets
        sums[:-1] += torch.einsum("bl,bld->ld", targets, latent)
        counts[:-1] += targets.sum(dim=0)
        sums[-1] += torch.einsum("bl,bld->d", negative, latent)
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
        fused = classifier(inputs)["features"]
        latent = F.normalize(
            comal(fused, compute_similarities=False, compute_reconstruction=False)[
                "latent_features"
            ].float(),
            dim=-1,
        )
        negative = 1.0 - targets
        sums[:-1] += torch.einsum("bl,bld->ld", targets, latent)
        counts[:-1] += targets.sum(dim=0)
        sums[-1] += torch.einsum("bl,bld->d", negative, latent)
        counts[-1] += negative.sum()
    comal.set_prototypes(sums, counts)


@torch.inference_mode()
def predict_tensors(
    trained: TrainedRound,
    features: np.ndarray,
    labels: np.ndarray,
    indices: Iterable[int],
    config: dict[str, Any],
    device: torch.device,
    *,
    return_latents: bool = True,
    similarity_mode: str = "full",
) -> dict[str, torch.Tensor]:
    """GPU-resident prediction path used by acquisition and evaluation.

    ``similarity_mode`` is ``full`` ([N,L,L+1]) for eval metrics or ``own_bg``
    ([N,L,2]) for cheaper acquisition-only scans.
    """
    if similarity_mode not in {"full", "own_bg"}:
        raise ValueError("similarity_mode must be 'full' or 'own_bg'")
    training = config.get("training", {})
    index_array = np.asarray(indices, dtype=np.int64)
    trained.classifier.eval()
    trained.comal.eval()
    num_labels = int(labels.shape[1])
    prototype_dim = int(trained.comal.prototype_dim)
    eval_batch = int(training.get("eval_batch_size", 1024))
    sim_width = num_labels + 1 if similarity_mode == "full" else 2
    if _use_gpu_resident(device, training) or device.type == "cuda":
        feature_tensor = _to_device_matrix(features, device)
        label_tensor = _to_device_matrix(labels, device)
        index_tensor = torch.as_tensor(index_array, device=device, dtype=torch.long)
        count = int(index_tensor.numel())
        # One gather makes subsequent batching contiguous (far cheaper than scattered selects).
        selected_features = feature_tensor.index_select(0, index_tensor)
        selected_labels = label_tensor.index_select(0, index_tensor)
        out_probs = torch.empty(count, num_labels, dtype=torch.float32, device=device)
        out_latents = (
            torch.empty(count, num_labels, prototype_dim, dtype=torch.float32, device=device)
            if return_latents
            else None
        )
        out_similarities = torch.empty(count, num_labels, sim_width, dtype=torch.float32, device=device)

        def _write(start: int, stop: int, output: dict[str, torch.Tensor], comal_output: dict[str, torch.Tensor]) -> None:
            # Write through into preallocated buffers to avoid temporary activations.
            torch.sigmoid(output["logits"], out=out_probs[start:stop])
            sims = comal_output["prototype_similarities"]
            if sims.dtype == out_similarities.dtype:
                out_similarities[start:stop].copy_(sims)
            else:
                out_similarities[start:stop] = sims.to(dtype=out_similarities.dtype)
            if out_latents is not None:
                latents = comal_output["latent_features"]
                if latents.dtype == out_latents.dtype:
                    out_latents[start:stop].copy_(latents)
                else:
                    out_latents[start:stop] = latents.to(dtype=out_latents.dtype)

        if count <= eval_batch:
            output = trained.classifier(selected_features)
            comal_output = trained.comal(
                output["features"],
                compute_reconstruction=False,
                compute_similarities=similarity_mode,
            )
            _write(0, count, output, comal_output)
        else:
            for start in range(0, count, eval_batch):
                stop = min(start + eval_batch, count)
                output = trained.classifier(selected_features[start:stop])
                comal_output = trained.comal(
                    output["features"],
                    compute_reconstruction=False,
                    compute_similarities=similarity_mode,
                )
                _write(start, stop, output, comal_output)
        result = {
            "indices": index_tensor,
            "labels": selected_labels,
            "probabilities": out_probs,
            "prototype_similarities": out_similarities,
        }
        if out_latents is not None:
            result["latents"] = out_latents
        return result
    loader = build_loader(
        features,
        labels,
        index_array,
        batch_size=eval_batch,
        shuffle=False,
        training=training,
    )
    index_parts = []
    label_parts = []
    prob_parts = []
    latent_parts = []
    similarity_parts = []
    for batch in loader:
        inputs = batch["features"].to(device, non_blocking=True)
        output = trained.classifier(inputs)
        comal_output = trained.comal(
            output["features"],
            compute_reconstruction=False,
            compute_similarities=similarity_mode,
        )
        index_parts.append(batch["index"].to(device, non_blocking=True))
        label_parts.append(batch["labels"].to(device, non_blocking=True))
        prob_parts.append(torch.sigmoid(output["logits"]))
        if return_latents:
            latent_parts.append(comal_output["latent_features"].float())
        similarity_parts.append(comal_output["prototype_similarities"].float())
    result = {
        "indices": torch.cat(index_parts) if index_parts else torch.empty(0, dtype=torch.long, device=device),
        "labels": torch.cat(label_parts) if label_parts else torch.empty(0, num_labels, device=device),
        "probabilities": torch.cat(prob_parts) if prob_parts else torch.empty(0, num_labels, device=device),
        "prototype_similarities": torch.cat(similarity_parts)
        if similarity_parts
        else torch.empty(0, num_labels, sim_width, device=device),
    }
    if return_latents:
        result["latents"] = (
            torch.cat(latent_parts)
            if latent_parts
            else torch.empty(0, num_labels, prototype_dim, device=device)
        )
    return result


@torch.inference_mode()
def predict(
    trained: TrainedRound,
    features: np.ndarray,
    labels: np.ndarray,
    indices: Iterable[int],
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, np.ndarray]:
    tensors = predict_tensors(trained, features, labels, indices, config, device)
    return {name: value.detach().cpu().numpy() for name, value in tensors.items()}


def _empty_prototype_similarity_metrics() -> dict[str, float | None]:
    return {
        "prototype_positive_own_similarity": None,
        "prototype_negative_own_similarity": None,
        "prototype_background_similarity": None,
        "prototype_positive_vs_background_margin": None,
    }


@torch.inference_mode()
def prototype_similarity_metrics_torch(
    labels: torch.Tensor, prototype_similarities: torch.Tensor
) -> dict[str, float | None]:
    """GPU reductions for prototype similarity metrics; one host sync for scalars."""
    sims = prototype_similarities
    if sims.ndim != 3 or int(sims.shape[0]) == 0:
        return _empty_prototype_similarity_metrics()
    if sims.shape[-1] == 2:
        own = sims[..., 0]
        background = sims[..., 1]
    else:
        index = torch.arange(sims.shape[1], device=sims.device)
        own = sims[:, index, index]
        background = sims[:, :, -1]
    pos_mask = labels >= 0.5
    neg_mask = ~pos_mask
    pos_count = pos_mask.sum()
    neg_count = neg_mask.sum()
    pos_own = own.masked_select(pos_mask).sum()
    neg_own = own.masked_select(neg_mask).sum()
    pos_margin = (own - background).masked_select(pos_mask).sum()
    bg_mean = background.mean()
    packed = torch.stack(
        (
            bg_mean,
            pos_count.to(dtype=bg_mean.dtype),
            neg_count.to(dtype=bg_mean.dtype),
            pos_own,
            neg_own,
            pos_margin,
        )
    )
    bg_v, pos_n, neg_n, pos_s, neg_s, margin_s = packed.detach().cpu().tolist()
    return {
        "prototype_background_similarity": float(bg_v),
        "prototype_positive_own_similarity": float(pos_s / pos_n) if pos_n > 0 else None,
        "prototype_negative_own_similarity": float(neg_s / neg_n) if neg_n > 0 else None,
        "prototype_positive_vs_background_margin": float(margin_s / pos_n) if pos_n > 0 else None,
    }


def prototype_similarity_metrics(
    labels: np.ndarray, prototype_similarities: np.ndarray
) -> dict[str, float | None]:
    """Evaluation metrics derived from CoMAL prototype similarities [N, L, L+1]."""
    sims = np.asarray(prototype_similarities)
    if sims.ndim != 3 or sims.shape[0] == 0:
        return _empty_prototype_similarity_metrics()
    if sims.shape[-1] == 2:
        own = sims[..., 0]
        background = sims[..., 1]
    else:
        index = np.arange(sims.shape[1])
        own = sims[:, index, index]
        background = sims[:, :, -1]
    pos_mask = np.asarray(labels) >= 0.5
    neg_mask = ~pos_mask
    return {
        "prototype_positive_own_similarity": float(own[pos_mask].mean()) if pos_mask.any() else None,
        "prototype_negative_own_similarity": float(own[neg_mask].mean()) if neg_mask.any() else None,
        "prototype_background_similarity": float(background.mean()),
        "prototype_positive_vs_background_margin": float((own - background)[pos_mask].mean())
        if pos_mask.any()
        else None,
    }


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
    if "prototype_similarities" in predictions:
        metrics.update(
            prototype_similarity_metrics(predictions["labels"], predictions["prototype_similarities"])
        )
    return metrics, predictions
