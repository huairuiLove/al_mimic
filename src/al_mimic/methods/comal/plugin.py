"""Task-independent CoMAL plugin."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from al_mimic.methods.api import (
    AcquisitionResult,
    PreparedContext,
    acquisition_result,
    context_field,
    validate_query_size,
)
from al_mimic.utils.prototypes import (
    PrototypeFitOptions,
    PrototypeFitState,
    attach_prototype_outputs,
    build_and_fit_prototype_module,
    positive_similarity_thresholds,
)

from .scoring import paper_comal_acquisition_scores
from .training import CoMALModule, comal_training_loss

CoMALState = PrototypeFitState


def _tensor_outputs(context: Any, fields: Mapping[str, Any], name: str) -> Mapping[str, torch.Tensor]:
    outputs = context_field(context, fields, name)
    if not isinstance(outputs, Mapping):
        raise ValueError(f"CoMAL {name} must be a tensor mapping")
    return outputs


def _method_config(context: Any, fields: Mapping[str, Any]) -> Mapping[str, Any]:
    config = context_field(context, fields, "config", {})
    if not isinstance(config, Mapping):
        raise ValueError("CoMAL config must be a mapping")
    return config


def _output_field(context: Any, fields: Mapping[str, Any], name: str) -> Any:
    direct = context_field(context, fields, name, None)
    if direct is not None:
        return direct
    outputs = context_field(context, fields, "candidate_outputs", None)
    if isinstance(outputs, Mapping):
        if name in outputs:
            return outputs[name]
        if name == "own_similarity" and "prototype_similarities" in outputs:
            return outputs["prototype_similarities"][..., 0]
    raise ValueError(f"CoMAL acquisition context requires {name!r}")


@dataclass(frozen=True)
class CoMALPlugin:
    """Acquire by the released CoMAL positive-prototype score."""

    method_id: str = "comal"
    display_name: str = "CoMAL"
    required_capabilities: tuple[str, ...] = (
        "multilabel_probabilities",
        "label_prototypes",
    )
    required_context_fields: tuple[str, ...] = (
        "candidate_ids",
        "query_size",
        "probabilities",
        "own_similarity",
        "labeled_labels",
        "labeled_own_similarity",
    )
    module_class: type[CoMALModule] = CoMALModule

    def training_loss(
        self,
        module: CoMALModule,
        classifier_outputs: dict[str, torch.Tensor],
        labels: torch.Tensor,
        config: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        return comal_training_loss(module, classifier_outputs, labels, config)

    def fit(self, context: Any = None, **fields: Any) -> CoMALState:
        """Fit a single-view CoMAL module from canonical labeled outputs."""
        config = _method_config(context, fields)
        settings = config.get("comal", {})
        if not isinstance(settings, Mapping):
            settings = {}
        return build_and_fit_prototype_module(
            self.module_class,
            _tensor_outputs(context, fields, "labeled_outputs"),
            PrototypeFitOptions.from_config(config),
            num_views=1,
            label_dim=int(settings.get("label_dim", 8)),
            prototype_dim=int(settings.get("prototype_dim", 8)),
        )

    def prepare_context(
        self,
        context: Any,
        state: CoMALState,
        **fields: Any,
    ) -> PreparedContext:
        """Attach prototype evidence needed by CoMAL acquisition."""
        candidate_outputs = attach_prototype_outputs(
            state.module,
            dict(_tensor_outputs(context, fields, "candidate_outputs")),
            batch_size=state.eval_batch_size,
        )
        labeled_outputs = state.labeled_outputs
        own_similarity = candidate_outputs["prototype_similarities"][..., 0]
        return PreparedContext(
            context,
            state=state,
            module=state.module,
            prototypes=state.prototypes,
            candidate_outputs=candidate_outputs,
            labeled_outputs=labeled_outputs,
            probabilities=candidate_outputs["probabilities"],
            own_similarity=own_similarity,
            labeled_labels=labeled_outputs["labels"],
            labeled_own_similarity=state.labeled_own_similarity,
        )

    def acquire(self, context: Any = None, **fields: Any) -> AcquisitionResult:
        candidates: Sequence[Any] = context_field(context, fields, "candidate_ids")
        query_size = validate_query_size(context_field(context, fields, "query_size"), len(candidates))
        probabilities = _output_field(context, fields, "probabilities")
        own_similarity = _output_field(context, fields, "own_similarity")
        labeled_labels = context_field(context, fields, "labeled_labels")
        labeled_own = context_field(context, fields, "labeled_own_similarity")
        thresholds = context_field(context, fields, "positive_thresholds", None)
        if thresholds is None:
            thresholds = positive_similarity_thresholds(labeled_labels, own_similarity=labeled_own)
        expected_cardinality = context_field(
            context,
            fields,
            "expected_cardinality",
            labeled_labels.float().sum(dim=1).mean(),
        )
        prototypes = context_field(context, fields, "prototypes", None)
        if prototypes is None:
            prototypes = torch.empty(0, device=probabilities.device)
        parts = paper_comal_acquisition_scores(
            probabilities,
            None,
            prototypes,
            thresholds,
            expected_cardinality=expected_cardinality,
            own_similarity=own_similarity,
        )
        positions = torch.argsort(parts.combined, descending=True, stable=True)[:query_size]
        scores = {
            "inverse_positive_evidence": parts.inverse_positive_evidence,
            "cardinality_mismatch": parts.cardinality_mismatch,
            "prototype_positive_count": parts.prototype_positive_count,
            "combined": parts.combined,
        }
        return acquisition_result(
            self.method_id,
            candidates,
            positions,
            scores=scores,
            diagnostics={"positive_thresholds": thresholds},
        )

    __call__ = acquire


PLUGIN = CoMALPlugin()


__all__ = ["CoMALPlugin", "CoMALState", "PLUGIN"]
