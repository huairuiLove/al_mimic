from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from al_mimic.methods.comal import (
    CoMALModule,
    comal_training_loss,
    paper_comal_acquisition_scores,
)
from al_mimic.methods.comal import (
    positive_similarity_thresholds as legacy_positive_similarity_thresholds,
)
from al_mimic.methods.comal.plugin import CoMALPlugin
from al_mimic.methods.mm_comal import (
    MMCoMALPlugin,
    estimate_mm_comal_statistics,
    mm_comal_acquisition_scores,
)
from al_mimic.utils.prototypes import (
    attach_prototype_outputs,
    positive_similarity_thresholds,
    refresh_prototypes,
)


def test_comal_training_is_finite_and_stops_classifier_gradient() -> None:
    torch.manual_seed(2)
    module = CoMALModule(input_dim=3, num_labels=2, label_dim=3, prototype_dim=2)
    features = torch.randn(4, 3, requires_grad=True)
    labels = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])

    loss = comal_training_loss(
        module,
        {"features": features},
        labels,
        {
            "comal": {
                "contrastive_label_sample_size": 2,
                "temperature": 0.2,
                "reconstruction_weight": 0.1,
                "classification_weight": 0.2,
            }
        },
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert features.grad is None
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in module.parameters()
    )


def test_prototype_refresh_and_attachment_support_multiview_outputs() -> None:
    torch.manual_seed(3)
    module = CoMALModule(
        input_dim=2,
        num_labels=2,
        label_dim=2,
        prototype_dim=2,
        num_views=3,
    )
    outputs = {
        "features": torch.randn(4, 2),
        "modality_tokens": torch.randn(4, 2, 2),
    }
    labels = torch.tensor([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]])
    views = torch.cat((outputs["modality_tokens"], outputs["features"][:, None]), dim=1)

    refresh_prototypes(module, views, labels, batch_size=2)
    attached = attach_prototype_outputs(module, outputs, batch_size=3)

    assert module.prototype_counts.shape == (3, 3)
    assert torch.equal(module.prototype_counts[:, :2], labels.sum(dim=0)[None].expand(3, -1))
    assert attached["prototype_similarities"].shape == (4, 3, 2, 2)
    assert attached["view_own_similarity"].shape == (4, 3, 2)
    nonempty = module.prototype_counts > 0
    assert torch.allclose(
        torch.linalg.vector_norm(module.prototypes[nonempty], dim=-1),
        torch.ones(int(nonempty.sum())),
        atol=1e-5,
    )


def test_legacy_positive_threshold_signature_is_preserved() -> None:
    labels = torch.tensor([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    similarity = torch.tensor([[0.8, -0.4], [0.6, 0.7], [-0.2, 0.5]])

    thresholds = legacy_positive_similarity_thresholds(
        None,
        labels,
        torch.empty(0),
        own_similarity=similarity,
    )

    assert torch.allclose(thresholds, torch.tensor([0.7, 0.6]))


def test_paper_comal_score_matches_released_formula() -> None:
    probabilities = torch.tensor([[0.9, 0.2], [0.6, 0.8], [0.2, 0.1]])
    own_similarity = torch.tensor([[0.8, -0.2], [0.0, 0.4], [-0.4, -0.5]])
    thresholds = torch.tensor([0.1, 0.0])

    parts = paper_comal_acquisition_scores(
        probabilities,
        None,
        torch.empty(0),
        thresholds,
        expected_cardinality=1.0,
        own_similarity=own_similarity,
    )

    evidence = ((own_similarity + 1.0) * 0.5).clamp_min(1e-10)
    expected_inverse = (((probabilities >= 0.5).float() * evidence).sum(dim=1) + 2e-10).reciprocal()
    expected_counts = (own_similarity > thresholds).sum(dim=1).float()
    expected_mismatch = (expected_counts - 1.0).abs()
    expected_combined = expected_inverse.sqrt() * expected_mismatch.sqrt()
    assert torch.allclose(parts.inverse_positive_evidence, expected_inverse)
    assert torch.equal(parts.prototype_positive_count, expected_counts)
    assert torch.allclose(parts.combined, expected_combined)


def test_comal_plugin_ranks_candidates_and_returns_uniform_result() -> None:
    labeled_labels = torch.tensor([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    labeled_similarity = torch.tensor([[0.8, -0.4], [0.6, 0.7], [-0.2, 0.5]])
    candidate_similarity = torch.tensor([[0.7, -0.2], [0.1, 0.1], [-0.5, -0.5]])
    probabilities = torch.tensor([[0.9, 0.1], [0.6, 0.7], [0.1, 0.1]])

    result = CoMALPlugin().acquire(
        candidate_ids=("a", "b", "c"),
        query_size=2,
        probabilities=probabilities,
        own_similarity=candidate_similarity,
        labeled_labels=labeled_labels,
        labeled_own_similarity=labeled_similarity,
    )

    expected = torch.argsort(result.scores["combined"], descending=True, stable=True)[:2]
    assert result.selected_positions == tuple(expected.tolist())
    assert result.selected_ids == tuple(("a", "b", "c")[index] for index in expected)
    assert result.diagnostics["method"] == "comal"


def test_mm_comal_alpha_zero_equals_fused_paper_comal() -> None:
    labels = torch.tensor([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]])
    labeled_similarity = torch.tensor(
        [
            [[0.8, -0.4], [0.7, -0.2], [0.9, -0.3]],
            [[0.6, 0.5], [0.4, 0.7], [0.8, 0.6]],
            [[-0.3, 0.8], [-0.1, 0.6], [-0.2, 0.9]],
            [[-0.6, -0.4], [-0.5, -0.3], [-0.7, -0.5]],
        ]
    )
    candidate_similarity = labeled_similarity[:3] * 0.9
    probabilities = torch.tensor([[0.8, 0.2], [0.7, 0.8], [0.2, 0.9]])
    statistics = estimate_mm_comal_statistics(
        labeled_similarity,
        labels,
        threshold_estimator="midpoint",
    )

    mm_parts = mm_comal_acquisition_scores(
        probabilities,
        candidate_similarity,
        statistics,
        expected_cardinality=1.0,
        alpha=0.0,
    )
    fused_thresholds = positive_similarity_thresholds(labels, own_similarity=labeled_similarity[:, -1])
    paper_parts = paper_comal_acquisition_scores(
        probabilities,
        None,
        torch.empty(0),
        fused_thresholds,
        expected_cardinality=1.0,
        own_similarity=candidate_similarity[:, -1],
    )

    assert torch.allclose(statistics.thresholds[-1], fused_thresholds)
    assert torch.allclose(mm_parts.base_score, paper_parts.combined)
    assert torch.allclose(mm_parts.combined, paper_parts.combined)
    assert torch.equal(mm_parts.dispersion, torch.zeros_like(mm_parts.dispersion)) is False


def _task_outputs(
    *,
    rows: int,
    feature_dim: int,
    labels: int,
    modalities: int | None,
    seed: int,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(rows, feature_dim, generator=generator)
    targets = torch.randint(0, 2, (rows, labels), generator=generator).float()
    targets[:labels] = torch.eye(labels)
    outputs = {
        "features": features,
        "probabilities": torch.sigmoid(torch.randn(rows, labels, generator=generator)),
        "labels": targets,
    }
    if modalities is not None:
        outputs["modality_tokens"] = torch.randn(rows, modalities, feature_dim, generator=generator)
    return outputs


def _fit_config() -> dict[str, object]:
    return {
        "comal": {
            "label_dim": 3,
            "prototype_dim": 2,
            "epochs": 2,
            "batch_size": 3,
            "eval_batch_size": 4,
            "learning_rate": 0.01,
            "contrastive_label_sample_size": 3,
            "temperature": 0.2,
            "seed": 19,
        },
        "training": {"weight_decay": 0.0, "gradient_clip": 1.0},
    }


@pytest.mark.parametrize("context_style", ["mimic", "brset"])
def test_comal_fit_prepare_and_acquire_need_no_task_adapter(context_style: str) -> None:
    plugin = CoMALPlugin()
    labeled = _task_outputs(rows=6, feature_dim=4, labels=3, modalities=None, seed=7)
    candidates = _task_outputs(rows=4, feature_dim=4, labels=3, modalities=None, seed=8)
    candidate_ids = (101, 102, 103, 104) if context_style == "mimic" else ("p1", "p2", "p3", "p4")
    values = {
        "candidate_ids": candidate_ids,
        "query_size": 2,
        "labeled_outputs": labeled,
        "candidate_outputs": candidates,
        "config": _fit_config(),
    }
    context = SimpleNamespace(**values) if context_style == "mimic" else values

    state = plugin.fit(context)
    prepared = plugin.prepare_context(context, state)
    result = plugin.acquire(prepared)

    assert state.module.num_views == 1
    assert len(state.history) == 2
    assert state.labeled_outputs["prototype_similarities"].shape == (6, 3, 2)
    assert state.labeled_outputs["own_similarity"].shape == (6, 3)
    assert torch.equal(prepared.prototypes, state.module.prototypes)
    assert prepared.candidate_outputs["prototype_similarities"].shape == (4, 3, 2)
    assert prepared.own_similarity.shape == (4, 3)
    assert prepared.labeled_own_similarity.shape == (6, 3)
    assert len(result.selected_ids) == 2
    assert "own_similarity" not in values


@pytest.mark.parametrize("context_style", ["mimic", "brset"])
def test_mm_comal_fit_prepare_and_acquire_need_no_task_adapter(context_style: str) -> None:
    plugin = MMCoMALPlugin()
    labeled = _task_outputs(rows=6, feature_dim=4, labels=3, modalities=2, seed=9)
    candidates = _task_outputs(rows=4, feature_dim=4, labels=3, modalities=2, seed=10)
    candidate_ids = (201, 202, 203, 204) if context_style == "mimic" else ("a", "b", "c", "d")
    values = {
        "candidate_ids": candidate_ids,
        "query_size": 2,
        "labeled_outputs": labeled,
        "candidate_outputs": candidates,
        "config": _fit_config(),
    }
    context = SimpleNamespace(**values) if context_style == "mimic" else values

    state = plugin.fit(context)
    prepared = plugin.prepare_context(context, state)
    result = plugin.acquire(prepared)

    assert state.module.num_views == 3
    assert len(state.history) == 2
    assert state.labeled_outputs["view_own_similarity"].shape == (6, 3, 3)
    assert state.labeled_outputs["own_similarity"].shape == (6, 3)
    assert prepared.candidate_outputs["view_own_similarity"].shape == (4, 3, 3)
    assert prepared.view_own_similarity.shape == (4, 3, 3)
    assert prepared.own_similarity.shape == (4, 3)
    assert prepared.labeled_view_own_similarity.shape == (6, 3, 3)
    assert torch.equal(prepared.prototypes, state.module.prototypes)
    assert len(result.selected_ids) == 2
    assert "view_own_similarity" not in values


def test_mm_comal_plugin_normalizes_view_weights_and_selects_exact_budget() -> None:
    labels = torch.tensor([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]])
    labeled_similarity = torch.tensor(
        [
            [[0.9, -0.5], [0.7, -0.3], [0.8, -0.4]],
            [[0.8, 0.7], [0.6, 0.8], [0.7, 0.6]],
            [[-0.4, 0.9], [-0.2, 0.7], [-0.3, 0.8]],
            [[-0.6, -0.5], [-0.4, -0.6], [-0.5, -0.4]],
        ]
    )
    result = MMCoMALPlugin().acquire(
        candidate_ids=(10, 11, 12),
        query_size=2,
        probabilities=torch.tensor([[0.9, 0.1], [0.6, 0.8], [0.2, 0.7]]),
        view_own_similarity=labeled_similarity[:3],
        labeled_labels=labels,
        labeled_view_own_similarity=labeled_similarity,
        config={"acquisition": {"mm": {"alpha": 0.5}}},
    )

    weights = result.diagnostics["view_weights"]
    assert torch.allclose(weights[:-1].sum(dim=0), torch.ones(2))
    assert torch.equal(weights[-1], torch.zeros(2))
    assert len(result.selected_positions) == 2
    assert len(set(result.selected_ids)) == 2
