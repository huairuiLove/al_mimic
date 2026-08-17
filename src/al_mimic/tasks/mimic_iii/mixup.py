"""Label-space mixup for the multi-label diagnosis head.

Motivation: on MIMIC-III Diagnoses the acquisition strategies systematically
prefer visits with fewer positive labels (13.8-14.9 positives versus 16.3 for
never-selected visits). Under multi-label BCE the positive terms carry most of
the learning signal, so those visits contribute weaker gradients. Mixing a
sparse anchor with a label-rich partner raises the positive mass of the target
without spending extra annotation budget.

Mixing happens on the fused representation rather than the raw inputs: clinical
notes enter the model as discrete token ids, which cannot be interpolated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class MixupConfig:
    enabled: bool = False
    space: str = "fused"
    alpha: float = 0.4
    weight: float = 1.0
    pairing: str = "targeted"
    anchor_quantile: float = 0.5
    keep_anchor: bool = True

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "MixupConfig":
        section = config.get("mixup") or {}
        if not isinstance(section, dict):
            raise ValueError("mixup section must be a mapping")
        parsed = cls(
            enabled=bool(section.get("enabled", False)),
            space=str(section.get("space", "fused")).lower(),
            alpha=float(section.get("alpha", 0.4)),
            weight=float(section.get("weight", 1.0)),
            pairing=str(section.get("pairing", "targeted")).lower(),
            anchor_quantile=float(section.get("anchor_quantile", 0.5)),
            keep_anchor=bool(section.get("keep_anchor", True)),
        )
        if parsed.alpha <= 0.0:
            raise ValueError("mixup.alpha must be positive")
        if parsed.space not in {"fused", "modalities"}:
            raise ValueError("mixup.space must be 'fused' or 'modalities'")
        if parsed.weight < 0.0:
            raise ValueError("mixup.weight must be non-negative")
        if parsed.pairing not in {"targeted", "random"}:
            raise ValueError("mixup.pairing must be 'targeted' or 'random'")
        if not 0.0 < parsed.anchor_quantile <= 1.0:
            raise ValueError("mixup.anchor_quantile must lie in (0, 1]")
        return parsed


@dataclass
class MixupBatch:
    features: torch.Tensor
    labels: torch.Tensor
    diagnostics: dict[str, float | int]


@dataclass
class ModalityMixupBatch:
    modalities: tuple[torch.Tensor | None, ...]
    labels: torch.Tensor
    diagnostics: dict[str, float | int]


@dataclass(frozen=True)
class _MixupPlan:
    anchors: torch.Tensor
    partners: torch.Tensor
    weights: torch.Tensor
    labels: torch.Tensor
    diagnostics: dict[str, float | int]


def _anchor_mask(positives: torch.Tensor, quantile: float) -> torch.Tensor:
    if quantile >= 1.0:
        return torch.ones_like(positives, dtype=torch.bool)
    cutoff = torch.quantile(positives, quantile)
    return positives <= cutoff


def _partner_positions(
    positives: torch.Tensor,
    anchors: torch.Tensor,
    pairing: str,
    generator: "np.random.Generator",
) -> torch.Tensor:
    count = int(positives.numel())
    if pairing == "random":
        offsets = generator.integers(1, count, size=int(anchors.numel()))
        drawn = (anchors.detach().cpu().numpy() + offsets) % count
    else:
        # Targeted: sample partners in proportion to their positive-label mass so
        # the mixed target gains positives the sparse anchor is missing.
        weights = positives.detach().cpu().numpy().clip(min=1.0)
        drawn = generator.choice(count, size=int(anchors.numel()), p=weights / weights.sum())
    return torch.as_tensor(drawn, dtype=torch.long, device=positives.device)


def _mixup_plan(
    labels: torch.Tensor,
    config: MixupConfig,
    generator: "np.random.Generator",
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> _MixupPlan | None:
    if not config.enabled or config.weight == 0.0:
        return None
    if labels.ndim != 2:
        raise ValueError("labels must be 2-D")
    if labels.shape[0] < 2:
        return None

    positives = labels.detach().float().sum(dim=1)
    anchors = _anchor_mask(positives, config.anchor_quantile).nonzero(as_tuple=True)[0]
    if anchors.numel() == 0:
        return None
    partners = _partner_positions(positives, anchors, config.pairing, generator)
    distinct = partners != anchors
    if not bool(distinct.any()):
        return None
    anchors = anchors[distinct]
    partners = partners[distinct]

    drawn = generator.beta(config.alpha, config.alpha, size=int(anchors.numel()))
    if config.keep_anchor:
        drawn = np.maximum(drawn, 1.0 - drawn)
    weights = torch.as_tensor(drawn, dtype=dtype, device=device).unsqueeze(1)
    anchor_labels = labels.index_select(0, anchors).to(device=device, dtype=dtype)
    partner_labels = labels.index_select(0, partners).to(device=device, dtype=dtype)
    mixed_labels = weights * anchor_labels + (1.0 - weights) * partner_labels
    diagnostics: dict[str, float | int] = {
        "anchor_fraction": anchors.numel() / labels.shape[0],
        "mean_lambda": float(weights.mean()),
        "anchor_positive_mean": float(anchor_labels.sum(dim=1).mean()),
        "mixed_positive_mean": float(mixed_labels.sum(dim=1).mean()),
        "virtual_samples": int(anchors.numel()),
    }
    return _MixupPlan(anchors, partners, weights, mixed_labels, diagnostics)


def label_space_mixup(
    features: torch.Tensor,
    labels: torch.Tensor,
    config: MixupConfig,
    generator: "np.random.Generator",
) -> MixupBatch | None:
    """Interpolate fused features and multi-hot labels of low-positive anchors.

    Returns ``None`` when the batch yields no usable anchor/partner pair.
    """
    if not config.enabled or config.weight == 0.0:
        return None
    if features.ndim != 2 or labels.ndim != 2:
        raise ValueError("features and labels must be 2-D")
    if features.shape[0] != labels.shape[0]:
        raise ValueError("features and labels must describe the same batch")
    plan = _mixup_plan(
        labels,
        config,
        generator,
        dtype=features.dtype,
        device=features.device,
    )
    if plan is None:
        return None
    anchor_features = features.index_select(0, plan.anchors)
    partner_features = features.index_select(0, plan.partners)
    mixed_features = plan.weights * anchor_features + (1.0 - plan.weights) * partner_features
    return MixupBatch(mixed_features, plan.labels, plan.diagnostics)


def modality_space_mixup(
    modalities: tuple[torch.Tensor | None, ...],
    labels: torch.Tensor,
    config: MixupConfig,
    generator: "np.random.Generator",
) -> ModalityMixupBatch | None:
    """Interpolate continuous modality representations with one shared mixup plan."""
    if not config.enabled or config.weight == 0.0:
        return None
    present = tuple(value for value in modalities if value is not None)
    if not present:
        raise ValueError("at least one modality representation is required")
    reference = present[0]
    if reference.ndim != 2:
        raise ValueError("modality representations must be 2-D")
    for value in present[1:]:
        if value.ndim != 2 or value.shape[0] != reference.shape[0]:
            raise ValueError("modality representations must have the same batch dimension")
        if value.device != reference.device:
            raise ValueError("modality representations must be on the same device")
    if labels.ndim != 2 or labels.shape[0] != reference.shape[0]:
        raise ValueError("labels and modalities must describe the same batch")
    if labels.device != reference.device:
        raise ValueError("labels and modality representations must be on the same device")

    plan = _mixup_plan(
        labels,
        config,
        generator,
        dtype=reference.dtype,
        device=reference.device,
    )
    if plan is None:
        return None
    mixed_modalities = tuple(
        None
        if value is None
        else plan.weights * value.index_select(0, plan.anchors)
        + (1.0 - plan.weights) * value.index_select(0, plan.partners)
        for value in modalities
    )
    return ModalityMixupBatch(mixed_modalities, plan.labels, plan.diagnostics)
