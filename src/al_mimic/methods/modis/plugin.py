"""Task-independent MoDIS plugin."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from al_mimic.methods.api import (
    AcquisitionResult,
    acquisition_result,
    context_field,
    validate_query_size,
)

from .acquire import acquire_modis
from .probes import MoDISProbeState, train_modality_probes


def _candidate_outputs(context: Any, fields: Mapping[str, Any]) -> dict[str, Any]:
    outputs = context_field(context, fields, "candidate_outputs", None)
    if outputs is None:
        outputs = context_field(context, fields, "candidates")
    if not isinstance(outputs, Mapping):
        raise ValueError("MoDIS candidate_outputs must be a tensor mapping")
    return dict(outputs)


@dataclass(frozen=True)
class MoDISPlugin:
    """Acquire by modality disagreement and intervention instability."""

    method_id: str = "modis"
    display_name: str = "MoDIS"
    required_capabilities: tuple[str, ...] = (
        "multilabel_probabilities",
        "modality_tokens",
        "token_fusion",
    )
    required_context_fields: tuple[str, ...] = (
        "candidate_ids",
        "query_size",
        "classifier",
        "probe_state",
        "candidate_outputs",
    )

    def fit(self, context: Any = None, **fields: Any) -> MoDISProbeState:
        """Fit the method-specific modality probes from labeled outputs."""
        labeled = context_field(context, fields, "labeled_outputs")
        groups = context_field(context, fields, "groups", labeled.get("subject_ids"))
        if groups is None:
            raise ValueError("MoDIS probe fitting requires row-aligned groups")
        if isinstance(groups, torch.Tensor):
            groups = groups.detach().cpu().reshape(-1).tolist()
        return train_modality_probes(
            labeled["modality_tokens"],
            labeled["labels"],
            groups,
            context_field(context, fields, "config", {}),
            seed=int(context_field(context, fields, "seed", 0)),
        )

    def acquire(self, context: Any = None, **fields: Any) -> AcquisitionResult:
        candidates: Sequence[Any] = context_field(context, fields, "candidate_ids")
        query_size = validate_query_size(context_field(context, fields, "query_size"), len(candidates))
        result = acquire_modis(
            context_field(context, fields, "classifier"),
            context_field(context, fields, "probe_state"),
            _candidate_outputs(context, fields),
            query_size=query_size,
            config=context_field(context, fields, "config", {}),
            initial_prevalence=context_field(context, fields, "initial_prevalence", None),
            candidate_metadata=context_field(context, fields, "candidate_metadata", None),
            comparison_scores=context_field(context, fields, "comparison_scores", None),
        )
        scores = {
            "disagreement": result.disagreement,
            "instability": result.instability,
            "dominance": result.dominance,
            "sufficiency_penalty": result.sufficiency_penalty,
            "combined": result.combined,
        }
        return acquisition_result(
            self.method_id,
            candidates,
            result.selected_positions,
            scores=scores,
            diagnostics=result.diagnostics,
        )

    __call__ = acquire


PLUGIN = MoDISPlugin()


__all__ = ["MoDISPlugin", "PLUGIN"]
