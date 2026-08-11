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
            alpha=float(section.get("alpha", 0.4)),
            weight=float(section.get("weight", 1.0)),
            pairing=str(section.get("pairing", "targeted")).lower(),
            anchor_quantile=float(section.get("anchor_quantile", 0.5)),
            keep_anchor=bool(section.get("keep_anchor", True)),
        )
        if parsed.alpha <= 0.0:
            raise ValueError("mixup.alpha must be positive")
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
    diagnostics: dict[str, float]


def _anchor_mask(positives: torch.Tensor, quantile: float) -> torch.Tensor:
    if quantile >= 1.0:
        return torch.ones_like(positives, dtype=torch.bool)
    cutoff = torch.quantile(positives, quantile)
    return positives <= cutoff


def _partner_positions(
    positives: torch.Tensor,
    anchor_count: int,
    pairing: str,
    generator: "np.random.Generator",
) -> torch.Tensor:
    count = int(positives.numel())
    if pairing == "random":
        drawn = generator.integers(count, size=anchor_count)
    else:
        # Targeted: sample partners in proportion to their positive-label mass so
        # the mixed target gains positives the sparse anchor is missing.
        weights = positives.detach().cpu().numpy().clip(min=1.0)
        drawn = generator.choice(count, size=anchor_count, p=weights / weights.sum())
    return torch.as_tensor(drawn, dtype=torch.long, device=positives.device)


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
    if features.shape[0] < 2:
        return None

    positives = labels.detach().float().sum(dim=1)
    anchors = _anchor_mask(positives, config.anchor_quantile).nonzero(as_tuple=True)[0]
    if anchors.numel() == 0:
        return None
    partners = _partner_positions(
        positives, int(anchors.numel()), config.pairing, generator
    )
    # A sample mixed with itself contributes nothing beyond the clean term.
    distinct = partners != anchors
    if not bool(distinct.any()):
        return None
    anchors = anchors[distinct]
    partners = partners[distinct]

    drawn = generator.beta(config.alpha, config.alpha, size=int(anchors.numel()))
    if config.keep_anchor:
        # Keep the labelled anchor dominant so the synthetic gradient stays
        # anchored to a real annotation rather than drifting to the partner.
        drawn = np.maximum(drawn, 1.0 - drawn)
    weight = torch.as_tensor(
        drawn, dtype=features.dtype, device=features.device
    ).unsqueeze(1)

    anchor_features = features.index_select(0, anchors)
    partner_features = features.index_select(0, partners)
    anchor_labels = labels.index_select(0, anchors).to(features.dtype)
    partner_labels = labels.index_select(0, partners).to(features.dtype)

    mixed_features = weight * anchor_features + (1.0 - weight) * partner_features
    mixed_labels = weight * anchor_labels + (1.0 - weight) * partner_labels
    diagnostics = {
        "anchor_fraction": anchors.numel() / features.shape[0],
        "mean_lambda": float(weight.mean()),
        "anchor_positive_mean": float(anchor_labels.sum(dim=1).mean()),
        "mixed_positive_mean": float(mixed_labels.sum(dim=1).mean()),
    }
    return MixupBatch(mixed_features, mixed_labels, diagnostics)
