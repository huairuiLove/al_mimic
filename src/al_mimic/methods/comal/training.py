"""Task-independent CoMAL module and auxiliary training helpers."""

from __future__ import annotations

from typing import Any

import torch

from al_mimic.utils.contrastive import supervised_contrastive_loss
from al_mimic.utils.prototypes import (
    LabelPrototypeAutoencoder,
    attach_prototype_outputs,
    finalize_prototype_outputs,
    prototype_training_loss,
    refresh_prototypes,
)


class CoMALModule(LabelPrototypeAutoencoder):
    """Named compatibility surface for the migrated CoMAL prototype module."""


def comal_training_loss(
    module: CoMALModule,
    classifier_outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    config: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Compute the detached CoMAL contrastive and reconstruction objective."""
    settings = (config or {}).get("comal", config or {})
    return prototype_training_loss(
        module,
        classifier_outputs,
        labels,
        maximum_labels=int(settings.get("contrastive_label_sample_size", 256)),
        temperature=float(settings.get("temperature", 0.07)),
        anchor_chunk_size=int(settings.get("anchor_chunk_size", 1024)),
        cross_view_weight=float(settings.get("cross_modal_weight", 0.15)),
        reconstruction_weight=float(settings.get("reconstruction_weight", 0.2)),
        classification_weight=float(settings.get("classification_weight", 0.5)),
    )


attach_comal_outputs = attach_prototype_outputs
finalize_comal_outputs = finalize_prototype_outputs
refresh_comal_prototypes = refresh_prototypes


__all__ = [
    "CoMALModule",
    "attach_comal_outputs",
    "comal_training_loss",
    "finalize_comal_outputs",
    "refresh_comal_prototypes",
    "supervised_contrastive_loss",
]
