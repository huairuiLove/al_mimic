"""Yang-Wu classifier training and method-plugin hooks for MIMIC-III."""

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

from .config import require_paths
from .data import YangWuFeatureStore
from .mixup import MixupConfig, label_space_mixup, modality_space_mixup
from .model import YangWuBertEncoderClassifier


@dataclass
class TrainedMultimodalRound:
    """Outputs owned by the task trainer; method state belongs to a plugin."""

    classifier: YangWuBertEncoderClassifier
    history: dict[str, list[float]]
    timings: dict[str, float]
    training_summary: dict[str, Any]
    method_state: Any | None = None


def _autocast(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    if precision not in {"bf16", "fp16"}:
        raise ValueError("training.precision must be 'fp32', 'bf16', or 'fp16'")
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=device.type == "cuda") for key, value in batch.items()}


def build_classifier(
    store: YangWuFeatureStore, config: dict[str, Any], device: torch.device
) -> YangWuBertEncoderClassifier:
    model_config = config.get("model", {})
    preprocessing = config.get("preprocessing", {})
    checkpoint = require_paths(config)["clinicalbert_checkpoint"]
    declared = int(model_config["output_size"])
    if store.scenario.label_columns is None and declared != store.label_count:
        raise ValueError(
            f"model.output_size={declared} does not match the {store.label_count} "
            "labels in the split artifact"
        )
    classifier = YangWuBertEncoderClassifier(
        str(checkpoint),
        num_labels=store.label_count,
        time_invariant_dim=int(preprocessing["time_invariant_dim"]),
        time_invariant_hidden_dim=int(model_config["time_invariant_hidden_dim"]),
        time_series_dim=int(preprocessing["time_series_dim"]),
        time_series_hidden_dim=int(model_config["time_series_hidden_dim"]),
        time_series_layers=int(model_config["time_series_layers"]),
        time_series_heads=int(model_config["time_series_heads"]),
        text_hidden_dim=int(model_config["text_hidden_dim"]),
        dropout=float(model_config["dropout"]),
        time_series_pooling=str(model_config.get("time_series_pooling", "first")),
    )
    return classifier.to(device)


def _linear_warmup_decay(total_steps: int, warmup_proportion: float):
    if total_steps < 1:
        raise ValueError("optimizer step budget must be positive")
    if not 0.0 <= warmup_proportion < 1.0:
        raise ValueError("training.warmup_proportion must be in [0, 1)")
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
        (decay if parameter.ndim > 1 else no_decay).append(parameter)
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
            if parameter.requires_grad and id(parameter) in names_by_id and id(parameter) not in assigned
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
    return {name: value.detach().cpu().clone() for name, value in module.state_dict().items()}


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
        loss_sum += float(F.binary_cross_entropy_with_logits(logits.float(), targets, reduction="sum").cpu())
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
            continue
        if name == "modality_tokens":
            result[name] = torch.empty(
                (0, len(classifier.modality_names), classifier.feature_dim),
                dtype=torch.float32,
                device=device,
            )
        elif name in {"labels", "probabilities"}:
            result[name] = torch.empty((0, store.audit.label_count), dtype=torch.float32, device=device)
        elif name in {"subject_ids", "indices"}:
            result[name] = torch.empty((0,), dtype=torch.long, device=device)
        else:
            result[name] = torch.empty((0, 0), dtype=torch.float32, device=device)
    return result


def train_multimodal_round(
    store: YangWuFeatureStore,
    labeled_indices: Iterable[int],
    config: dict[str, Any],
    device: torch.device,
    *,
    validation_indices: Iterable[int] | None = None,
) -> TrainedMultimodalRound:
    """Train one fresh Yang-Wu classifier round.

    Acquisition methods are deliberately absent from this loop. A method plugin
    may consume the returned classifier and outputs through the runner context.
    """
    indices = np.unique(np.asarray(list(labeled_indices), dtype=np.int64))
    if not indices.size:
        raise ValueError("at least one labeled sample is required")
    training = config.get("training", {})
    model_seed = int(config.get("model", {}).get("seed", 1337))
    torch.manual_seed(model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(model_seed)
    classifier = build_classifier(store, config, device)
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
        raise ValueError("epochs is too small to reach optimizer_steps_per_round")

    optimizer = _build_classifier_optimizer(classifier, training)
    scheduler = LambdaLR(
        optimizer,
        _linear_warmup_decay(optimizer_step_budget, float(training["warmup_proportion"])),
    )
    precision = str(training["precision"])
    history: dict[str, list[float]] = {"classifier_loss": []}
    mixup = MixupConfig.from_config(config)
    mixup_generator: np.random.Generator | None = None
    if mixup.enabled:
        mixup_generator = np.random.default_rng(model_seed)
        history.update(
            {
                "mixup_loss": [],
                "mixup_anchor_positive_mean": [],
                "mixup_mixed_positive_mean": [],
                "mixup_virtual_samples": [],
            }
        )

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

    start = time.perf_counter()
    classifier.train()
    optimizer_steps = 0
    best_validation_loss = float("inf")
    best_epoch = 0
    best_optimizer_step = 0
    stale_validation_checks = 0
    best_classifier_state: dict[str, torch.Tensor] | None = None
    stop_reason = "max_epochs"

    for epoch_index in range(max_epochs):
        classifier_losses: list[torch.Tensor] = []
        mixup_losses: list[torch.Tensor] = []
        anchor_positives: list[float] = []
        mixed_positives: list[float] = []
        virtual_samples = 0
        for host_batch in loader:
            if optimizer_steps >= optimizer_step_budget:
                break
            batch = _move_batch(host_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(device, precision):
                if mixup.enabled and mixup.space == "modalities":
                    text, series, static = classifier.encode_modalities(batch)
                    output = classifier.forward_from_modalities(text, series, static)
                else:
                    text = series = static = None
                    output = classifier(batch)
                classifier_loss = F.binary_cross_entropy_with_logits(output["logits"], batch["labels"])
                loss = classifier_loss
                if mixup.enabled:
                    assert mixup_generator is not None
                    if mixup.space == "modalities":
                        assert text is not None and series is not None
                        mixed = modality_space_mixup(
                            (text, series, static), batch["labels"], mixup, mixup_generator
                        )
                        mixed_output = (
                            None if mixed is None else classifier.forward_from_modalities(*mixed.modalities)
                        )
                    else:
                        mixed = label_space_mixup(output["features"], batch["labels"], mixup, mixup_generator)
                        mixed_output = (
                            None if mixed is None else {"logits": classifier.classifier(mixed.features)}
                        )
                    if mixed is not None and mixed_output is not None:
                        mixup_loss = F.binary_cross_entropy_with_logits(mixed_output["logits"], mixed.labels)
                        loss = loss + mixup.weight * mixup_loss
                        mixup_losses.append(mixup_loss.detach())
                        anchor_positives.append(float(mixed.diagnostics["anchor_positive_mean"]))
                        mixed_positives.append(float(mixed.diagnostics["mixed_positive_mean"]))
                        virtual_samples += int(mixed.diagnostics["virtual_samples"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), float(training["gradient_clip"]))
            optimizer.step()
            scheduler.step()
            optimizer_steps += 1
            classifier_losses.append(classifier_loss.detach())

        if not classifier_losses:
            raise RuntimeError("training produced no optimizer steps in an epoch")
        history["classifier_loss"].append(float(torch.stack(classifier_losses).mean().cpu()))
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
            history["mixup_virtual_samples"].append(float(virtual_samples))
        if validation is not None:
            assert validation_loader is not None
            current_validation_loss = _validation_loss(classifier, validation_loader, device, precision)
            if not np.isfinite(current_validation_loss):
                raise FloatingPointError("validation BCE became non-finite")
            history["validation_loss"].append(current_validation_loss)
            if current_validation_loss < best_validation_loss - early_stopping_min_delta:
                best_validation_loss = current_validation_loss
                best_epoch = epoch_index + 1
                best_optimizer_step = optimizer_steps
                stale_validation_checks = 0
                best_classifier_state = _clone_module_state(classifier)
            else:
                stale_validation_checks += 1
                if early_stopping_patience > 0 and stale_validation_checks >= early_stopping_patience:
                    stop_reason = "early_stopping"
                    break
        if optimizer_steps >= optimizer_step_budget:
            stop_reason = "optimizer_step_budget"
            break

    if best_classifier_state is not None:
        classifier.load_state_dict(best_classifier_state)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    learning_rates = {
        str(group["group_name"]): float(group["initial_lr"]) for group in optimizer.param_groups
    }
    summary: dict[str, Any] = {
        "max_epochs": max_epochs,
        "epochs_ran": len(history["classifier_loss"]),
        "optimizer_step_budget": optimizer_step_budget,
        "optimizer_steps": optimizer_steps,
        "stop_reason": stop_reason,
        "best_epoch": best_epoch if validation is not None else None,
        "best_optimizer_step": best_optimizer_step if validation is not None else None,
        "best_validation_loss": best_validation_loss if validation is not None else None,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_min_delta": early_stopping_min_delta,
        "optimizer": "adamw",
        "weight_decay": float(training["weight_decay"]),
        "learning_rates": learning_rates,
        "mixup_space": mixup.space if mixup.enabled else None,
        "mixup_virtual_samples": int(sum(history.get("mixup_virtual_samples", []))),
    }
    return TrainedMultimodalRound(
        classifier=classifier,
        history=history,
        timings={"classifier_training_sec": elapsed},
        training_summary=summary,
    )


__all__ = [
    "TrainedMultimodalRound",
    "build_classifier",
    "collect_classifier_outputs",
    "train_multimodal_round",
]
