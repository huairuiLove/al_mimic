"""Formal round training for multimodal MIMIC-III classification tasks."""

from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from .config import resolve_path
from .mixup import MixupConfig, label_space_mixup
from .model import CoMALModule, YangWuBertEncoderClassifier, supervised_contrastive_loss
from .multimodal_data import YangWuFeatureStore


@dataclass
class TrainedMultimodalRound:
    classifier: YangWuBertEncoderClassifier
    comal: CoMALModule | None
    history: dict[str, list[float]]
    timings: dict[str, float]
    training_summary: dict[str, Any]
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
    checkpoint = resolve_path(
        config, config.get("dataset", {}).get("clinicalbert_checkpoint", "")
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing ClinicalBERT checkpoint: {checkpoint}")
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
        time_series_pooling=str(model.get("time_series_pooling", "first")),
    )
    return classifier.to(device)


def _linear_warmup_decay(total_steps: int, warmup_proportion: float):
    warmup_steps = int(total_steps * warmup_proportion)

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        return max(0.0, float(total_steps - step) / max(1, total_steps - warmup_steps))

    return schedule


def _add_optimizer_groups(
    groups: list[dict[str, Any]],
    named_parameters: Iterable[tuple[str, torch.nn.Parameter]],
    *,
    group_name: str,
    learning_rate: float,
    weight_decay: float,
) -> None:
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for _name, parameter in named_parameters:
        if not parameter.requires_grad:
            continue
        # Biases and normalization scales are one-dimensional and should not decay.
        target = decay if parameter.ndim > 1 else no_decay
        target.append(parameter)
    if decay:
        groups.append(
            {
                "params": decay,
                "lr": learning_rate,
                "weight_decay": weight_decay,
                "group_name": f"{group_name}_decay",
            }
        )
    if no_decay:
        groups.append(
            {
                "params": no_decay,
                "lr": learning_rate,
                "weight_decay": 0.0,
                "group_name": f"{group_name}_no_decay",
            }
        )


def _classifier_parameter_groups(
    classifier: YangWuBertEncoderClassifier,
    training: dict[str, Any],
) -> list[dict[str, Any]]:
    base_learning_rate = float(training["learning_rate"])
    bert_learning_rate = float(training["bert_learning_rate"])
    layerwise_decay = float(training["bert_layerwise_lr_decay"])
    weight_decay = float(training["weight_decay"])
    if base_learning_rate <= 0 or bert_learning_rate <= 0:
        raise ValueError("classifier and BERT learning rates must be positive")
    if not 0 < layerwise_decay <= 1:
        raise ValueError("bert_layerwise_lr_decay must be in (0, 1]")
    if weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")

    all_named = list(classifier.named_parameters())
    names_by_id = {id(parameter): name for name, parameter in all_named}
    assigned: set[int] = set()
    groups: list[dict[str, Any]] = []

    def add_module(module: torch.nn.Module, group_name: str, learning_rate: float) -> None:
        selected = [
            (names_by_id[id(parameter)], parameter)
            for parameter in module.parameters()
            if parameter.requires_grad
            and id(parameter) in names_by_id
            and id(parameter) not in assigned
        ]
        assigned.update(id(parameter) for _name, parameter in selected)
        _add_optimizer_groups(
            groups,
            selected,
            group_name=group_name,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
        )

    text_encoder = classifier.text_encoder
    encoder = getattr(text_encoder, "encoder", None)
    encoder_layers = list(getattr(encoder, "layer", ()))
    embeddings = getattr(text_encoder, "embeddings", None)
    if isinstance(embeddings, torch.nn.Module):
        add_module(
            embeddings,
            "bert_embeddings",
            bert_learning_rate * layerwise_decay ** len(encoder_layers),
        )
    for layer_index, layer in enumerate(encoder_layers):
        depth_from_top = len(encoder_layers) - layer_index - 1
        add_module(
            layer,
            f"bert_layer_{layer_index}",
            bert_learning_rate * layerwise_decay**depth_from_top,
        )
    add_module(text_encoder, "bert_other", bert_learning_rate)

    non_bert = [
        (name, parameter)
        for name, parameter in all_named
        if parameter.requires_grad and id(parameter) not in assigned
    ]
    assigned.update(id(parameter) for _name, parameter in non_bert)
    _add_optimizer_groups(
        groups,
        non_bert,
        group_name="multimodal",
        learning_rate=base_learning_rate,
        weight_decay=weight_decay,
    )
    trainable = {id(parameter) for _name, parameter in all_named if parameter.requires_grad}
    if assigned != trainable:
        raise RuntimeError("optimizer parameter grouping did not cover every trainable parameter")
    return groups


def _build_classifier_optimizer(
    classifier: YangWuBertEncoderClassifier,
    training: dict[str, Any],
) -> AdamW:
    groups = _classifier_parameter_groups(classifier, training)
    return AdamW(groups, lr=float(training["learning_rate"]))


def _clone_module_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in module.state_dict().items()
    }


@torch.inference_mode()
def _validation_loss(
    classifier: YangWuBertEncoderClassifier,
    loader: Iterable[dict[str, torch.Tensor]],
    device: torch.device,
    precision: str,
) -> float:
    was_training = classifier.training
    classifier.eval()
    loss_sum = 0.0
    element_count = 0
    for host_batch in loader:
        batch = _move_batch(host_batch, device)
        with _autocast(device, precision):
            logits = classifier(batch)["logits"]
        targets = batch["labels"].float()
        loss_sum += float(
            F.binary_cross_entropy_with_logits(
                logits.float(), targets, reduction="sum"
            ).cpu()
        )
        element_count += targets.numel()
    classifier.train(was_training)
    return loss_sum / max(1, element_count)


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
        "subject_ids": [],
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
        values["subject_ids"].append(batch["subject_id"].long())
        values["indices"].append(batch["index"].long())
        if return_tokens:
            values["modality_tokens"].append(output["modality_tokens"].float())
    result: dict[str, torch.Tensor] = {}
    for name, chunks in values.items():
        if chunks:
            result[name] = torch.cat(chunks, dim=0)
        elif name == "modality_tokens":
            result[name] = torch.empty(
                (0, len(classifier.modality_names), classifier.feature_dim),
                dtype=torch.float32,
                device=device,
            )
        else:
            if name in {"labels", "probabilities"}:
                result[name] = torch.empty(
                    (0, store.audit.label_count), dtype=torch.float32, device=device
                )
            elif name in {"subject_ids", "indices"}:
                result[name] = torch.empty((0,), dtype=torch.long, device=device)
            else:
                result[name] = torch.empty((0, 0), dtype=torch.float32, device=device)
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


def _build_comal(
    classifier: YangWuBertEncoderClassifier,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[CoMALModule, AdamW]:
    strategy = str(config["active_learning"]["strategy"]).lower()
    multi_view = strategy == "mm_comal"
    cfg = config.get("comal", {})
    training = config.get("training", {})
    comal = CoMALModule(
        classifier.feature_dim,
        classifier.num_labels,
        int(cfg.get("label_dim", 8)),
        int(cfg.get("prototype_dim", 8)),
        num_views=len(classifier.modality_names) + 1 if multi_view else 1,
    ).to(device)
    comal_groups: list[dict[str, Any]] = []
    _add_optimizer_groups(
        comal_groups,
        comal.named_parameters(),
        group_name="comal",
        learning_rate=float(cfg.get("learning_rate", 1e-3)),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    optimizer = AdamW(
        comal_groups,
        lr=float(cfg.get("learning_rate", 1e-3)),
    )
    return comal, optimizer


def _comal_loss(
    comal: CoMALModule,
    classifier_output: dict[str, torch.Tensor],
    targets: torch.Tensor,
    config: dict[str, Any],
) -> torch.Tensor:
    cfg = config.get("comal", {})
    multi_view = comal.num_views > 1
    # The auxiliary branch is optimized in this same step, while its input is
    # detached so its loss cannot update the classifier or any encoder path.
    if multi_view:
        features = torch.cat(
            (
                classifier_output["modality_tokens"].detach(),
                classifier_output["features"].detach()[:, None, :],
            ),
            dim=1,
        )
    else:
        features = classifier_output["features"].detach()
    output = comal(features, compute_similarities=False)
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
    fused_target = features if features.ndim == 2 else features[:, -1]
    reconstruction = F.mse_loss(output["reconstructed_features"], fused_target)
    reconstruction_bce = F.binary_cross_entropy_with_logits(
        output["reconstructed_logits"], targets
    )
    return (
        contrastive
        + float(cfg.get("reconstruction_weight", 0.2)) * reconstruction
        + float(cfg.get("classification_weight", 0.5)) * reconstruction_bce
    )


@torch.inference_mode()
def _finalize_comal(
    classifier: YangWuBertEncoderClassifier,
    comal: CoMALModule,
    store: YangWuFeatureStore,
    indices: np.ndarray,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor | None]:
    multi_view = comal.num_views > 1
    outputs = collect_classifier_outputs(
        classifier, store, indices, config, device, return_tokens=multi_view
    )
    features = (
        torch.cat((outputs["modality_tokens"], outputs["features"][:, None, :]), dim=1)
        if multi_view
        else outputs["features"]
    )
    labels = outputs["labels"]
    training = config.get("training", {})
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
    return outputs, own, view_own


def train_multimodal_round(
    store: YangWuFeatureStore,
    labeled_indices: Iterable[int],
    config: dict[str, Any],
    device: torch.device,
    *,
    validation_indices: Iterable[int] | None = None,
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
    max_epochs = int(training["epochs"])
    optimizer_step_budget = int(training["optimizer_steps_per_round"])
    early_stopping_patience = int(training["early_stopping_patience"])
    early_stopping_min_delta = float(training["early_stopping_min_delta"])
    if max_epochs < 1 or optimizer_step_budget < 1:
        raise ValueError("epochs and optimizer_steps_per_round must be positive")
    if early_stopping_patience < 0 or early_stopping_min_delta < 0:
        raise ValueError("early stopping patience and min_delta must be non-negative")
    if max_epochs * len(loader) < optimizer_step_budget:
        raise ValueError(
            "epochs is too small to reach optimizer_steps_per_round for this labeled set"
        )
    optimizer = _build_classifier_optimizer(classifier, training)
    scheduler = LambdaLR(
        optimizer,
        _linear_warmup_decay(
            optimizer_step_budget, float(training["warmup_proportion"])
        ),
    )
    strategy = str(config["active_learning"]["strategy"]).lower()
    comal: CoMALModule | None = None
    comal_optimizer: AdamW | None = None
    if strategy in {"comal", "mm_comal"}:
        comal, comal_optimizer = _build_comal(classifier, config, device)
    precision = str(training["precision"])
    history: dict[str, list[float]] = {"classifier_loss": [], "comal_loss": []}
    timings: dict[str, float] = {}
    mixup = MixupConfig.from_config(config)
    mixup_generator: np.random.Generator | None = None
    if mixup.enabled:
        mixup_generator = np.random.default_rng(model_seed)
        history["mixup_loss"] = []
        history["mixup_anchor_positive_mean"] = []
        history["mixup_mixed_positive_mean"] = []
    start = time.perf_counter()
    classifier.train()
    if comal is not None:
        comal.train()
    optimizer_steps = 0
    best_validation_loss = float("inf")
    best_epoch = 0
    best_optimizer_step = 0
    stale_validation_checks = 0
    best_classifier_state: dict[str, torch.Tensor] | None = None
    best_comal_state: dict[str, torch.Tensor] | None = None
    stop_reason = "max_epochs"
    validation = (
        None
        if validation_indices is None
        else np.unique(np.asarray(list(validation_indices), dtype=np.int64))
    )
    validation_loader = None
    if validation is not None:
        if not validation.size:
            raise ValueError("early stopping requires at least one validation sample")
        validation_loader = store.make_loader(
            validation,
            batch_size=int(training.get("eval_batch_size", 400)),
            shuffle=False,
            num_workers=int(training.get("num_workers", 12)),
            pin_memory=bool(training.get("pin_memory", True)),
        )
        history["validation_loss"] = []

    for epoch_index in range(max_epochs):
        classifier_losses: list[torch.Tensor] = []
        comal_losses: list[torch.Tensor] = []
        mixup_losses: list[torch.Tensor] = []
        anchor_positives: list[float] = []
        mixed_positives: list[float] = []
        for host_batch in loader:
            if optimizer_steps >= optimizer_step_budget:
                break
            batch = _move_batch(host_batch, device)
            optimizer.zero_grad(set_to_none=True)
            if comal_optimizer is not None:
                comal_optimizer.zero_grad(set_to_none=True)
            with _autocast(device, precision):
                output = classifier(batch, return_tokens=comal is not None and comal.num_views > 1)
                classifier_loss = F.binary_cross_entropy_with_logits(
                    output["logits"], batch["labels"]
                )
                auxiliary_loss = (
                    _comal_loss(comal, output, batch["labels"], config)
                    if comal is not None
                    else classifier_loss.detach() * 0.0
                )
                loss = classifier_loss + auxiliary_loss
                if mixup.enabled:
                    assert mixup_generator is not None
                    mixed = label_space_mixup(
                        output["features"], batch["labels"], mixup, mixup_generator
                    )
                    if mixed is not None:
                        mixup_loss = F.binary_cross_entropy_with_logits(
                            classifier.classifier(mixed.features), mixed.labels
                        )
                        loss = loss + mixup.weight * mixup_loss
                        mixup_losses.append(mixup_loss.detach())
                        anchor_positives.append(mixed.diagnostics["anchor_positive_mean"])
                        mixed_positives.append(mixed.diagnostics["mixed_positive_mean"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), float(training["gradient_clip"]))
            if comal is not None:
                torch.nn.utils.clip_grad_norm_(comal.parameters(), float(training["gradient_clip"]))
            optimizer.step()
            if comal_optimizer is not None:
                comal_optimizer.step()
            scheduler.step()
            optimizer_steps += 1
            classifier_losses.append(classifier_loss.detach())
            if comal is not None:
                comal_losses.append(auxiliary_loss.detach())
        history["classifier_loss"].append(float(torch.stack(classifier_losses).mean().cpu()))
        if comal is not None:
            history["comal_loss"].append(float(torch.stack(comal_losses).mean().cpu()))
        if mixup.enabled:
            history["mixup_loss"].append(
                float(torch.stack(mixup_losses).mean().cpu()) if mixup_losses else 0.0
            )
            history["mixup_anchor_positive_mean"].append(
                float(np.mean(anchor_positives)) if anchor_positives else 0.0
            )
            history["mixup_mixed_positive_mean"].append(
                float(np.mean(mixed_positives)) if mixed_positives else 0.0
            )
        if validation is not None:
            assert validation_loader is not None
            current_validation_loss = _validation_loss(
                classifier,
                validation_loader,
                device,
                precision,
            )
            if not np.isfinite(current_validation_loss):
                raise FloatingPointError("validation BCE became non-finite")
            history["validation_loss"].append(current_validation_loss)
            if current_validation_loss < best_validation_loss - early_stopping_min_delta:
                best_validation_loss = current_validation_loss
                best_epoch = epoch_index + 1
                best_optimizer_step = optimizer_steps
                stale_validation_checks = 0
                best_classifier_state = _clone_module_state(classifier)
                best_comal_state = (
                    _clone_module_state(comal) if comal is not None else None
                )
            else:
                stale_validation_checks += 1
                if (
                    early_stopping_patience > 0
                    and stale_validation_checks >= early_stopping_patience
                ):
                    stop_reason = "early_stopping"
                    break
        if optimizer_steps >= optimizer_step_budget:
            stop_reason = "optimizer_step_budget"
            break

    if best_classifier_state is not None:
        classifier.load_state_dict(best_classifier_state)
        if comal is not None and best_comal_state is not None:
            comal.load_state_dict(best_comal_state)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    timings["joint_training_sec"] = time.perf_counter() - start
    timings["classifier_training_sec"] = timings["joint_training_sec"]
    learning_rates = {
        str(group["group_name"]): float(group["initial_lr"])
        for group in optimizer.param_groups
    }
    training_summary: dict[str, Any] = {
        "max_epochs": max_epochs,
        "epochs_ran": len(history["classifier_loss"]),
        "optimizer_step_budget": optimizer_step_budget,
        "optimizer_steps": optimizer_steps,
        "stop_reason": stop_reason,
        "best_epoch": best_epoch if validation is not None else None,
        "best_optimizer_step": best_optimizer_step if validation is not None else None,
        "best_validation_loss": (
            best_validation_loss if validation is not None else None
        ),
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
        "optimizer": "adamw",
        "weight_decay": float(training["weight_decay"]),
        "learning_rates": learning_rates,
    }

    labeled_outputs = None
    labeled_own = None
    labeled_view_own = None
    modis_state = None
    if strategy in {"comal", "mm_comal"}:
        if comal is None:
            raise RuntimeError("joint CoMAL module was not initialized")
        labeled_outputs, labeled_own, labeled_view_own = _finalize_comal(
            classifier, comal, store, indices, config, device
        )
    elif strategy == "modis":
        start = time.perf_counter()
        labeled_outputs = collect_classifier_outputs(
            classifier, store, indices, config, device, return_tokens=True
        )
        from modis.probes import train_modality_probes

        modis_state = train_modality_probes(
            labeled_outputs["modality_tokens"],
            labeled_outputs["labels"],
            labeled_outputs["subject_ids"].detach().cpu().numpy(),
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
        training_summary=training_summary,
        labeled_outputs=labeled_outputs,
        labeled_own_similarity=labeled_own,
        labeled_view_own_similarity=labeled_view_own,
        modis_state=modis_state,
    )
