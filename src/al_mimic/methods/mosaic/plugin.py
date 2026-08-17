"""Task-independent MoSAIC plugin."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from al_mimic.methods.api import (
    AcquisitionResult,
    acquisition_result,
    context_field,
    validate_query_size,
)

from .acquire import acquire_mosaic


def _tensor_dictionary(context: Any, fields: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = context_field(context, fields, name)
    if not isinstance(value, Mapping):
        raise ValueError(f"MoSAIC {name} must be a tensor mapping")
    return dict(value)


@dataclass(frozen=True)
class MoSAICPlugin:
    """Acquire by Fisher design gain and modality-lattice synergy."""

    method_id: str = "mosaic"
    display_name: str = "MoSAIC"
    required_capabilities: tuple[str, ...] = (
        "multilabel_probabilities",
        "modality_tokens",
        "token_fusion",
        "reference_labels",
    )
    required_context_fields: tuple[str, ...] = (
        "candidate_ids",
        "query_size",
        "classifier",
        "labeled_outputs",
        "reference_outputs",
        "candidate_outputs",
    )

    def acquire(self, context: Any = None, **fields: Any) -> AcquisitionResult:
        candidates: Sequence[Any] = context_field(context, fields, "candidate_ids")
        query_size = validate_query_size(context_field(context, fields, "query_size"), len(candidates))
        seed = int(context_field(context, fields, "seed", 0)) + int(
            context_field(context, fields, "round_index", 0)
        )
        result = acquire_mosaic(
            context_field(context, fields, "classifier"),
            _tensor_dictionary(context, fields, "labeled_outputs"),
            _tensor_dictionary(context, fields, "reference_outputs"),
            _tensor_dictionary(context, fields, "candidate_outputs"),
            query_size=query_size,
            config=context_field(context, fields, "config", {}),
            seed=seed,
        )
        scores = {
            "additive": result.additive,
            "synergy": result.synergy,
            "total_gain": result.total_gain,
            "combined": result.combined,
        }
        diagnostics = dict(result.diagnostics)
        diagnostics["seed"] = seed
        return acquisition_result(
            self.method_id,
            candidates,
            result.selected_positions,
            scores=scores,
            diagnostics=diagnostics,
        )

    __call__ = acquire


PLUGIN = MoSAICPlugin()


__all__ = ["MoSAICPlugin", "PLUGIN"]
