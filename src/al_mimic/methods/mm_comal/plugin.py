"""Task-independent MM-CoMAL plugin."""

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
    LabelPrototypeAutoencoder,
    PrototypeFitOptions,
    PrototypeFitState,
    attach_prototype_outputs,
    build_and_fit_prototype_module,
    prototype_training_loss,
)

from .scoring import estimate_mm_comal_statistics, mm_comal_acquisition_scores

MMCoMALState = PrototypeFitState


def _tensor_outputs(context: Any, fields: Mapping[str, Any], name: str) -> Mapping[str, torch.Tensor]:
    outputs = context_field(context, fields, name)
    if not isinstance(outputs, Mapping):
        raise ValueError(f"MM-CoMAL {name} must be a tensor mapping")
    return outputs


def _method_config(context: Any, fields: Mapping[str, Any]) -> Mapping[str, Any]:
    config = context_field(context, fields, "config", {})
    if not isinstance(config, Mapping):
        raise ValueError("MM-CoMAL config must be a mapping")
    return config


def _output_field(context: Any, fields: Mapping[str, Any], name: str) -> Any:
    direct = context_field(context, fields, name, None)
    if direct is not None:
        return direct
    outputs = context_field(context, fields, "candidate_outputs", None)
    if isinstance(outputs, Mapping):
        if name in outputs:
            return outputs[name]
        if name == "view_own_similarity" and "prototype_similarities" in outputs:
            return outputs["prototype_similarities"][..., 0]
    raise ValueError(f"MM-CoMAL acquisition context requires {name!r}")


def _settings(config: Mapping[str, Any]) -> Mapping[str, Any]:
    acquisition = config.get("acquisition", {})
    if isinstance(acquisition, Mapping):
        nested = acquisition.get("mm", {})
        if isinstance(nested, Mapping):
            return nested
    nested = config.get("mm_comal", {})
    return nested if isinstance(nested, Mapping) else {}


@dataclass(frozen=True)
class MMCoMALPlugin:
    """Acquire by reliability-weighted multimodal prototype evidence."""

    method_id: str = "mm_comal"
    display_name: str = "MM-CoMAL"
    required_capabilities: tuple[str, ...] = (
        "multilabel_probabilities",
        "modality_tokens",
        "label_prototypes",
    )
    required_context_fields: tuple[str, ...] = (
        "candidate_ids",
        "query_size",
        "probabilities",
        "view_own_similarity",
        "labeled_labels",
        "labeled_view_own_similarity",
    )
    module_class: type[LabelPrototypeAutoencoder] = LabelPrototypeAutoencoder

    def training_loss(
        self,
        module: LabelPrototypeAutoencoder,
        classifier_outputs: dict[str, torch.Tensor],
        labels: torch.Tensor,
        config: dict[str, Any] | None = None,
    ) -> torch.Tensor:
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

    def fit(self, context: Any = None, **fields: Any) -> MMCoMALState:
        """Fit a multi-view prototype module from canonical labeled outputs."""
        config = _method_config(context, fields)
        settings = config.get("comal", {})
        if not isinstance(settings, Mapping):
            settings = {}
        labeled_outputs = _tensor_outputs(context, fields, "labeled_outputs")
        tokens = labeled_outputs.get("modality_tokens")
        if tokens is None or tokens.ndim != 3:
            raise ValueError("MM-CoMAL labeled_outputs require modality_tokens [N,M,D]")
        return build_and_fit_prototype_module(
            self.module_class,
            labeled_outputs,
            PrototypeFitOptions.from_config(config),
            num_views=int(tokens.shape[1]) + 1,
            label_dim=int(settings.get("label_dim", 8)),
            prototype_dim=int(settings.get("prototype_dim", 8)),
        )

    def prepare_context(
        self,
        context: Any,
        state: MMCoMALState,
        **fields: Any,
    ) -> PreparedContext:
        """Attach view-resolved prototype evidence needed by MM-CoMAL."""
        candidate_outputs = attach_prototype_outputs(
            state.module,
            dict(_tensor_outputs(context, fields, "candidate_outputs")),
            batch_size=state.eval_batch_size,
        )
        labeled_outputs = state.labeled_outputs
        labeled_view_own = state.labeled_view_own_similarity
        if labeled_view_own is None:
            raise ValueError("MM-CoMAL state does not contain view-resolved similarities")
        return PreparedContext(
            context,
            state=state,
            module=state.module,
            prototypes=state.prototypes,
            candidate_outputs=candidate_outputs,
            labeled_outputs=labeled_outputs,
            probabilities=candidate_outputs["probabilities"],
            view_own_similarity=candidate_outputs["view_own_similarity"],
            own_similarity=candidate_outputs["view_own_similarity"][:, -1],
            labeled_labels=labeled_outputs["labels"],
            labeled_view_own_similarity=labeled_view_own,
            labeled_own_similarity=state.labeled_own_similarity,
        )

    def acquire(self, context: Any = None, **fields: Any) -> AcquisitionResult:
        candidates: Sequence[Any] = context_field(context, fields, "candidate_ids")
        query_size = validate_query_size(context_field(context, fields, "query_size"), len(candidates))
        probabilities = _output_field(context, fields, "probabilities")
        similarities = _output_field(context, fields, "view_own_similarity")
        labels = context_field(context, fields, "labeled_labels")
        labeled_similarities = context_field(context, fields, "labeled_view_own_similarity")
        config = context_field(context, fields, "config", {})
        settings = _settings(config)
        statistics = estimate_mm_comal_statistics(
            labeled_similarities,
            labels,
            reliability_shrinkage=float(settings.get("reliability_shrinkage", 10.0)),
            threshold_shrinkage=float(settings.get("threshold_shrinkage", 10.0)),
            threshold_estimator=str(settings.get("threshold_estimator", "shrunk")),
            include_fused_in_weights=bool(settings.get("include_fused_in_weights", False)),
            equal_weights=bool(settings.get("equal_weights", False)),
        )
        expected_cardinality = context_field(
            context,
            fields,
            "expected_cardinality",
            labels.float().sum(dim=1).mean(),
        )
        parts = mm_comal_acquisition_scores(
            probabilities,
            similarities,
            statistics,
            expected_cardinality=expected_cardinality,
            alpha=float(settings.get("alpha", 1.0)),
            dispersion=str(settings.get("dispersion", "weighted_mad")),
        )
        positions = torch.argsort(parts.combined, descending=True, stable=True)[:query_size]
        scores = {
            "inverse_positive_evidence": parts.inverse_positive_evidence,
            "cardinality_mismatch": parts.cardinality_mismatch,
            "prototype_positive_count": parts.prototype_positive_count,
            "dispersion": parts.dispersion,
            "base_score": parts.base_score,
            "combined": parts.combined,
        }
        return acquisition_result(
            self.method_id,
            candidates,
            positions,
            scores=scores,
            diagnostics={
                "reliability": statistics.reliability,
                "view_weights": statistics.weights,
                "positive_thresholds": statistics.thresholds,
            },
        )

    __call__ = acquire


PLUGIN = MMCoMALPlugin()


__all__ = ["MMCoMALPlugin", "MMCoMALState", "PLUGIN"]
