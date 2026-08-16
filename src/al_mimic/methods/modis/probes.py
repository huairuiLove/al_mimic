"""Stop-gradient modality probes and out-of-fold reliability estimation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from .intervene import modality_prototypes


class ModalityProbes(nn.Module):
    """Capacity-matched linear readouts over individual modality tokens."""

    def __init__(self, num_modalities: int, token_dim: int, num_labels: int) -> None:
        super().__init__()
        if min(num_modalities, token_dim, num_labels) < 1:
            raise ValueError("probe dimensions must be positive")
        self.probes = nn.ModuleList(nn.Linear(token_dim, num_labels) for _ in range(num_modalities))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[1] != len(self.probes):
            raise ValueError("tokens must have shape [N, M, D]")
        detached = tokens.detach()
        return torch.stack(
            [probe(detached[:, modality]) for modality, probe in enumerate(self.probes)],
            dim=1,
        )


@dataclass(frozen=True)
class ReliabilityStatistics:
    information_gain: torch.Tensor
    skill_scores: torch.Tensor
    skill_standard_errors: torch.Tensor
    shrunk_skill_scores: torch.Tensor
    pooled_skill_scores: torch.Tensor
    label_weights: torch.Tensor
    modality_weights: torch.Tensor


@dataclass(frozen=True)
class MoDISProbeState:
    probes: ModalityProbes
    statistics: ReliabilityStatistics
    prototypes: torch.Tensor
    labeled_prevalence: torch.Tensor
    labeled_cardinality: float
    diagnostics: dict[str, Any]
    history: list[float]


def _positive_weights(labels: torch.Tensor, maximum: float) -> torch.Tensor:
    positives = labels.sum(dim=0)
    negatives = labels.shape[0] - positives
    return (negatives / positives.clamp_min(1.0)).clamp(1.0, float(maximum))


def _fit_probes(
    tokens: torch.Tensor,
    labels: torch.Tensor,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    maximum_pos_weight: float,
    seed: int,
) -> tuple[ModalityProbes, list[float]]:
    probes = ModalityProbes(tokens.shape[1], tokens.shape[2], labels.shape[1]).to(tokens.device)
    optimizer = AdamW(probes.parameters(), lr=learning_rate, weight_decay=weight_decay)
    pos_weight = _positive_weights(labels, maximum_pos_weight)
    generator = torch.Generator(device=tokens.device)
    generator.manual_seed(int(seed))
    history: list[float] = []
    count = int(tokens.shape[0])
    step = min(max(1, int(batch_size)), count)
    probes.train()
    for _ in range(max(1, int(epochs))):
        order = torch.randperm(count, device=tokens.device, generator=generator)
        epoch_loss = tokens.new_zeros((), dtype=torch.float32)
        batches = 0
        for start in range(0, count, step):
            selected = order[start : start + step]
            batch_tokens = tokens.index_select(0, selected)
            batch_labels = labels.index_select(0, selected)
            optimizer.zero_grad(set_to_none=True)
            logits = probes(batch_tokens).float()
            targets = batch_labels[:, None, :].expand_as(logits)
            loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)
            loss.backward()
            optimizer.step()
            epoch_loss = epoch_loss + loss.detach()
            batches += 1
        history.append(float(epoch_loss / max(batches, 1)))
    probes.eval()
    return probes, history


@torch.inference_mode()
def probe_probabilities(
    probes: ModalityProbes,
    tokens: torch.Tensor,
    *,
    batch_size: int,
) -> torch.Tensor:
    count = int(tokens.shape[0])
    output = torch.empty(
        count,
        tokens.shape[1],
        probes.probes[0].out_features,
        dtype=torch.float32,
        device=tokens.device,
    )
    step = max(1, int(batch_size))
    probes.eval()
    for start in range(0, count, step):
        stop = min(start + step, count)
        output[start:stop] = torch.sigmoid(probes(tokens[start:stop]).float())
    return output


def _group_folds(groups: Iterable[object], requested_folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    group_array = np.asarray(list(groups), dtype=object)
    if group_array.ndim != 1:
        raise ValueError("OOF groups must be a one-dimensional row-aligned SUBJECT_ID array")
    if group_array.size == 0:
        return []
    if any(group is None for group in group_array):
        raise ValueError("OOF groups must contain true SUBJECT_ID values, not missing identifiers")
    unique_count = int(np.unique(group_array).size)
    fold_count = min(max(2, int(requested_folds)), unique_count)
    if unique_count < 2:
        return []
    from sklearn.model_selection import GroupKFold

    splitter = GroupKFold(n_splits=fold_count)
    placeholder = np.zeros(group_array.shape[0], dtype=np.float32)
    return [
        (train.astype(np.int64, copy=False), validation.astype(np.int64, copy=False))
        for train, validation in splitter.split(placeholder, groups=group_array)
    ]


@torch.inference_mode()
def estimate_reliability_weights(
    labels: torch.Tensor,
    oof_probabilities: torch.Tensor,
    *,
    epsilon: float = 1e-7,
) -> ReliabilityStatistics:
    """Estimate equations (10)-(11) with variance-derived empirical-Bayes shrinkage."""
    if labels.ndim != 2 or oof_probabilities.ndim != 3:
        raise ValueError("labels and OOF probabilities must have shapes [N,C] and [N,M,C]")
    if labels.shape[0] != oof_probabilities.shape[0] or labels.shape[1] != oof_probabilities.shape[2]:
        raise ValueError("labels and OOF probabilities have incompatible shapes")
    values = labels.float()
    probabilities = oof_probabilities.float().clamp(epsilon, 1.0 - epsilon)
    prevalence = values.mean(dim=0).clamp(epsilon, 1.0 - epsilon)
    baseline_loss = -(values * prevalence.log() + (1.0 - values) * (1.0 - prevalence).log())
    probe_loss = -(
        values[:, None, :] * probabilities.log() + (1.0 - values[:, None, :]) * (1.0 - probabilities).log()
    )
    improvements = baseline_loss[:, None, :] - probe_loss
    information_gain = improvements.mean(dim=0)
    entropy = -(prevalence * prevalence.log() + (1.0 - prevalence) * (1.0 - prevalence).log())
    entropy = entropy.clamp_min(epsilon)
    skill_scores = information_gain / entropy[None, :]
    if labels.shape[0] > 1:
        standard_error_gain = improvements.std(dim=0, unbiased=True) / labels.shape[0] ** 0.5
    else:
        standard_error_gain = torch.zeros_like(information_gain)
    standard_error_skill = standard_error_gain / entropy[None, :]

    positive_counts = values.sum(dim=0)
    count_denominator = positive_counts.sum().clamp_min(1.0)
    pooled = (skill_scores * positive_counts[None, :]).sum(dim=1) / count_denominator
    observed_variance = skill_scores.var(dim=1, unbiased=skill_scores.shape[1] > 1)
    sampling_variance = standard_error_skill.square().mean(dim=1)
    true_variance = (observed_variance - sampling_variance).clamp_min(0.0)
    posterior_weight = true_variance[:, None] / (
        true_variance[:, None] + standard_error_skill.square()
    ).clamp_min(epsilon)
    shrunk = posterior_weight * skill_scores + (1.0 - posterior_weight) * pooled[:, None]
    shrunk = torch.where(true_variance[:, None] <= epsilon, pooled[:, None], shrunk)

    positive_skill = shrunk.clamp_min(0.0)
    denominator = positive_skill.sum(dim=0, keepdim=True)
    uniform = torch.full_like(positive_skill, 1.0 / positive_skill.shape[0])
    label_weights = torch.where(denominator > epsilon, positive_skill / denominator, uniform)
    modality_weights = (label_weights * positive_counts[None, :]).sum(dim=1) / count_denominator
    if float(positive_counts.sum()) <= epsilon:
        modality_weights = torch.full_like(modality_weights, 1.0 / modality_weights.numel())
    return ReliabilityStatistics(
        information_gain=information_gain,
        skill_scores=skill_scores,
        skill_standard_errors=standard_error_skill,
        shrunk_skill_scores=shrunk,
        pooled_skill_scores=pooled,
        label_weights=label_weights,
        modality_weights=modality_weights,
    )


def train_modality_probes(
    tokens: torch.Tensor,
    labels: torch.Tensor,
    groups: Iterable[object],
    config: dict[str, Any],
    *,
    seed: int,
) -> MoDISProbeState:
    """Fit probes and SUBJECT_ID-grouped OOF copies without perturbing caller RNG."""
    if tokens.ndim != 3 or labels.ndim != 2 or tokens.shape[0] != labels.shape[0]:
        raise ValueError("expected tokens [N,M,D] and labels [N,C]")
    group_array = np.asarray(list(groups), dtype=object)
    if group_array.ndim != 1 or group_array.shape[0] != tokens.shape[0]:
        raise ValueError("OOF SUBJECT_ID groups must be row-aligned with tokens and labels")
    modis_cfg = config.get("modis", {})
    training_cfg = config.get("training", {})
    epochs = int(modis_cfg.get("probe_epochs", training_cfg.get("epochs", 20)))
    batch_size = int(modis_cfg.get("probe_batch_size", training_cfg.get("batch_size", 512)))
    prediction_batch_size = int(
        modis_cfg.get("probe_eval_batch_size", training_cfg.get("eval_batch_size", 1024))
    )
    fit_options = {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": float(modis_cfg.get("probe_learning_rate", training_cfg.get("learning_rate", 1e-3))),
        "weight_decay": float(modis_cfg.get("probe_weight_decay", training_cfg.get("weight_decay", 1e-4))),
        "maximum_pos_weight": float(training_cfg.get("maximum_pos_weight", 20.0)),
    }
    detached_tokens = tokens.detach().float()
    detached_labels = labels.detach().float()
    folds = _group_folds(group_array, int(modis_cfg.get("oof_folds", 5)))
    cuda_devices = []
    if tokens.device.type == "cuda":
        cuda_devices = [
            tokens.device.index if tokens.device.index is not None else torch.cuda.current_device()
        ]
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(int(seed))
        full_probes, history = _fit_probes(
            detached_tokens,
            detached_labels,
            seed=seed,
            **fit_options,
        )
        if folds:
            oof_probabilities = torch.empty(
                labels.shape[0],
                tokens.shape[1],
                labels.shape[1],
                dtype=torch.float32,
                device=tokens.device,
            )
            for fold_index, (train_indices, validation_indices) in enumerate(folds):
                train_tensor = torch.as_tensor(train_indices, dtype=torch.long, device=tokens.device)
                validation_tensor = torch.as_tensor(
                    validation_indices, dtype=torch.long, device=tokens.device
                )
                fold_probes, _fold_history = _fit_probes(
                    detached_tokens.index_select(0, train_tensor),
                    detached_labels.index_select(0, train_tensor),
                    seed=seed + fold_index + 1,
                    **fit_options,
                )
                oof_probabilities[validation_tensor] = probe_probabilities(
                    fold_probes,
                    detached_tokens.index_select(0, validation_tensor),
                    batch_size=prediction_batch_size,
                )
        else:
            prevalence = detached_labels.mean(dim=0)
            oof_probabilities = prevalence[None, None, :].expand(labels.shape[0], tokens.shape[1], -1).clone()

    statistics = estimate_reliability_weights(detached_labels, oof_probabilities)
    prototypes, prototype_diagnostics = modality_prototypes(
        detached_tokens,
        method=str(modis_cfg.get("prototype", "mean")),
        pairwise_sample_size=int(modis_cfg.get("prototype_diagnostic_sample_size", 512)),
    )
    diagnostics = {
        "oof_folds": len(folds),
        "oof_group_count": int(np.unique(group_array).size),
        "information_gain": statistics.information_gain.detach().cpu().tolist(),
        "skill_scores": statistics.skill_scores.detach().cpu().tolist(),
        "skill_standard_errors": statistics.skill_standard_errors.detach().cpu().tolist(),
        "shrunk_skill_scores": statistics.shrunk_skill_scores.detach().cpu().tolist(),
        "pooled_skill_scores": statistics.pooled_skill_scores.detach().cpu().tolist(),
        "label_weights": statistics.label_weights.detach().cpu().tolist(),
        "modality_weights": statistics.modality_weights.detach().cpu().tolist(),
        "unreliable_modalities": [
            index for index, value in enumerate(statistics.pooled_skill_scores) if float(value) <= 0.0
        ],
        "prototype": str(modis_cfg.get("prototype", "mean")).lower(),
        "prototype_diagnostics": asdict(prototype_diagnostics),
    }
    return MoDISProbeState(
        probes=full_probes,
        statistics=statistics,
        prototypes=prototypes,
        labeled_prevalence=detached_labels.mean(dim=0),
        labeled_cardinality=float(detached_labels.sum(dim=1).mean()),
        diagnostics=diagnostics,
        history=history,
    )
