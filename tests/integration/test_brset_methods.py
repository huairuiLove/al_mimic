from __future__ import annotations

from typing import Any

import pytest
import torch
import torch.nn as nn

from al_mimic.core.methods import fit_method, prepare_method_context
from al_mimic.methods import get_method
from al_mimic.tasks.brset.runner import acquire_method, build_method_context


class SyntheticBrsetClassifier(nn.Module):
    modality_names = ("fundus_image", "clinical_metadata")
    feature_dim = 3

    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Linear(3, 2, bias=False)
        with torch.no_grad():
            self.classifier.weight.copy_(torch.tensor([[0.7, -0.2, 0.4], [-0.3, 0.8, 0.2]]))

    def fuse_from_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens.sum(dim=-2)

    def probabilities_from_fused(self, fused: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.classifier(fused))


def _patient_outputs(
    classifier: SyntheticBrsetClassifier,
    tokens: torch.Tensor,
    labels: torch.Tensor,
) -> dict[str, torch.Tensor]:
    features = classifier.fuse_from_tokens(tokens)
    return {
        "labels": labels,
        "modality_tokens": tokens,
        "features": features,
        "probabilities": classifier.probabilities_from_fused(features),
        "image_counts": torch.ones(tokens.shape[0], dtype=torch.long),
    }


def _fixture() -> tuple[
    SyntheticBrsetClassifier,
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
]:
    classifier = SyntheticBrsetClassifier()
    labeled_ids = tuple(f"labeled-{index}" for index in range(6))
    candidate_ids = tuple(f"candidate-{index}" for index in range(6))
    reference_ids = tuple(f"validation-{index}" for index in range(4))
    labeled_tokens = torch.tensor(
        [
            [[1.0, 0.2, -0.1], [0.2, 0.1, 0.3]],
            [[0.8, -0.3, 0.4], [0.1, 0.5, -0.2]],
            [[-0.7, 0.8, 0.2], [0.4, -0.2, 0.3]],
            [[-0.9, 0.2, -0.3], [-0.1, 0.7, 0.4]],
            [[0.4, 0.9, -0.5], [0.6, -0.4, 0.1]],
            [[-0.2, -0.8, 0.6], [0.3, 0.2, -0.5]],
        ]
    )
    candidate_tokens = torch.tensor(
        [
            [[1.2, -0.1, 0.2], [-0.2, 0.7, 0.1]],
            [[-1.0, 0.4, 0.5], [0.5, -0.3, 0.2]],
            [[0.3, 1.1, -0.4], [0.7, -0.5, 0.3]],
            [[-0.4, -0.9, 0.7], [0.2, 0.4, -0.6]],
            [[0.9, 0.6, -0.2], [-0.5, 0.2, 0.8]],
            [[-0.8, 0.1, -0.7], [0.6, 0.9, 0.2]],
        ]
    )
    reference_tokens = torch.tensor(
        [
            [[0.7, 0.3, -0.2], [0.1, 0.4, 0.2]],
            [[-0.6, 0.7, 0.1], [0.3, -0.1, 0.5]],
            [[0.2, -0.7, 0.8], [0.4, 0.6, -0.3]],
            [[-0.5, -0.2, -0.4], [-0.2, 0.8, 0.7]],
        ]
    )
    labeled_labels = torch.tensor([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    candidate_labels = torch.zeros(6, 2)
    reference_labels = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]])
    return (
        classifier,
        labeled_ids,
        candidate_ids,
        reference_ids,
        _patient_outputs(classifier, labeled_tokens, labeled_labels),
        _patient_outputs(classifier, candidate_tokens, candidate_labels),
        _patient_outputs(classifier, reference_tokens, reference_labels),
    )


def _config() -> dict[str, Any]:
    return {
        "training": {
            "epochs": 1,
            "comal_epochs": 1,
            "batch_size": 4,
            "comal_batch_size": 4,
            "eval_batch_size": 16,
            "learning_rate": 0.01,
            "weight_decay": 0.0,
            "gradient_clip": 1.0,
            "maximum_pos_weight": 10.0,
        },
        "comal": {
            "label_dim": 3,
            "prototype_dim": 2,
            "learning_rate": 0.01,
            "contrastive_label_sample_size": 2,
            "temperature": 0.2,
            "reconstruction_weight": 0.1,
            "classification_weight": 0.2,
        },
        "acquisition": {"mm": {"alpha": 0.5, "dispersion": "weighted_mad"}},
        "modis": {
            "probe_epochs": 1,
            "probe_batch_size": 3,
            "probe_eval_batch_size": 16,
            "probe_learning_rate": 0.01,
            "oof_folds": 2,
            "workset_size": 6,
            "grid_k": 2,
            "bisect_steps": 0,
            "fusion_batch_size": 32,
        },
        "mosaic": {
            "eta": 0.25,
            "partners": 1,
            "workset_size": 6,
            "synergy_workset_size": 6,
            "damping": 0.1,
            "deflation_steps": 1,
            "fusion_batch_size": 32,
            "value_batch_size": 32,
        },
    }


@pytest.mark.parametrize("method_name", ["random", "comal", "modis", "mosaic"])
def test_all_methods_select_exact_unique_patient_budget(method_name: str) -> None:
    (
        classifier,
        labeled_ids,
        candidate_ids,
        reference_ids,
        labeled_outputs,
        candidate_outputs,
        reference_outputs,
    ) = _fixture()
    plugin = get_method(method_name)
    context = build_method_context(
        classifier=classifier,
        candidate_ids=candidate_ids,
        labeled_ids=labeled_ids,
        reference_ids=reference_ids,
        candidate_outputs=candidate_outputs,
        labeled_outputs=labeled_outputs,
        reference_outputs=reference_outputs,
        query_size=2,
        config=_config(),
        seed=11,
        round_index=1,
        initial_prevalence=labeled_outputs["labels"].mean(dim=0),
    )
    state = fit_method(plugin, context)
    context = prepare_method_context(plugin, context, state)
    selected, diagnostics = acquire_method(plugin, context)

    assert len(selected) == 2
    assert len(set(selected)) == 2
    assert set(selected).issubset(candidate_ids)
    assert not set(selected) & set(labeled_ids)
    assert diagnostics["query_unit"] == "patient"
    assert diagnostics["candidate_patients"] == len(candidate_ids)
    assert diagnostics["selected_patients"] == 2
    if method_name == "comal":
        assert state is not None
        assert state.labeled_outputs["labels"].shape[0] == len(labeled_ids)
        assert state.labeled_outputs["prototype_similarities"].shape[0] == len(labeled_ids)
    if method_name == "modis":
        assert context["probe_state"].diagnostics["oof_group_count"] == len(labeled_ids)
    if method_name == "mosaic":
        assert context["reference_outputs"] is reference_outputs
        assert set(context["reference_patient_ids"]) == set(reference_ids)
        assert not set(context["reference_patient_ids"]) & set(labeled_ids)


def test_method_context_rejects_non_validation_reference_reuse() -> None:
    (
        classifier,
        labeled_ids,
        candidate_ids,
        _reference_ids,
        labeled_outputs,
        candidate_outputs,
        _reference_outputs,
    ) = _fixture()

    with pytest.raises(ValueError, match="must be disjoint"):
        build_method_context(
            classifier=classifier,
            candidate_ids=candidate_ids,
            labeled_ids=labeled_ids,
            reference_ids=labeled_ids,
            candidate_outputs=candidate_outputs,
            labeled_outputs=labeled_outputs,
            reference_outputs=labeled_outputs,
            query_size=2,
            config=_config(),
            seed=11,
            round_index=0,
            initial_prevalence=labeled_outputs["labels"].mean(dim=0),
        )
