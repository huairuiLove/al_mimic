from __future__ import annotations

import torch
import torch.nn as nn

from al_mimic.methods.modis import (
    ModalityProbes,
    MoDISPlugin,
    MoDISProbeState,
    ReliabilityStatistics,
    estimate_reliability_weights,
    generalized_js_disagreement,
)
from al_mimic.methods.mosaic import MoSAICPlugin, decompose_lattice, mobius_inversion


class SumFusionClassifier(nn.Module):
    modality_names = ("left", "right")
    feature_dim = 2

    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.classifier.weight.copy_(torch.eye(2))

    def fuse_from_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens.sum(dim=-2)

    def probabilities_from_fused(self, fused: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.classifier(fused))


def _probe_state() -> MoDISProbeState:
    probes = ModalityProbes(2, 2, 2)
    with torch.no_grad():
        probes.probes[0].weight.copy_(torch.eye(2))
        probes.probes[0].bias.zero_()
        probes.probes[1].weight.copy_(-torch.eye(2))
        probes.probes[1].bias.zero_()
    half = torch.full((2, 2), 0.5)
    statistics = ReliabilityStatistics(
        information_gain=torch.zeros(2, 2),
        skill_scores=torch.zeros(2, 2),
        skill_standard_errors=torch.zeros(2, 2),
        shrunk_skill_scores=torch.zeros(2, 2),
        pooled_skill_scores=torch.zeros(2),
        label_weights=half,
        modality_weights=torch.tensor([0.5, 0.5]),
    )
    return MoDISProbeState(
        probes=probes,
        statistics=statistics,
        prototypes=torch.zeros(2, 2),
        labeled_prevalence=torch.tensor([0.5, 0.5]),
        labeled_cardinality=1.0,
        diagnostics={"fixture": True},
        history=[],
    )


def test_generalized_js_disagreement_is_zero_for_identical_views() -> None:
    probabilities = torch.tensor([[[0.8, 0.2], [0.8, 0.2]], [[0.6, 0.4], [0.6, 0.4]]])
    support = torch.ones(2, 2, dtype=torch.bool)
    weights = torch.full((2, 2), 0.5)

    disagreement, per_label = generalized_js_disagreement(probabilities, weights, support)
    opposite, _ = generalized_js_disagreement(
        torch.tensor([[[0.9, 0.1], [0.1, 0.9]]]),
        weights,
        torch.ones(1, 2, dtype=torch.bool),
    )

    assert torch.allclose(per_label, torch.zeros_like(per_label), atol=1e-6)
    assert torch.allclose(disagreement, torch.zeros_like(disagreement), atol=1e-6)
    assert float(opposite[0]) > 0.3


def test_reliability_weights_favor_the_informative_modality() -> None:
    labels = torch.tensor([[1.0], [1.0], [0.0], [0.0]])
    oof = torch.tensor(
        [
            [[0.9], [0.5]],
            [[0.8], [0.5]],
            [[0.1], [0.5]],
            [[0.2], [0.5]],
        ]
    )

    statistics = estimate_reliability_weights(labels, oof)

    assert statistics.label_weights.shape == (2, 1)
    assert statistics.label_weights[0, 0] > statistics.label_weights[1, 0]
    assert torch.allclose(statistics.label_weights.sum(dim=0), torch.ones(1))
    assert torch.allclose(statistics.modality_weights.sum(), torch.tensor(1.0))


def test_modis_plugin_runs_without_mosaic_and_selects_exact_budget() -> None:
    classifier = SumFusionClassifier()
    tokens = torch.tensor(
        [
            [[2.0, -1.0], [-1.0, 1.5]],
            [[1.0, 1.0], [-1.0, -1.0]],
            [[-2.0, 1.0], [1.0, -1.5]],
            [[0.2, 0.1], [0.1, 0.2]],
        ]
    )
    probabilities = classifier.probabilities_from_fused(classifier.fuse_from_tokens(tokens))

    result = MoDISPlugin().acquire(
        candidate_ids=("a", "b", "c", "d"),
        query_size=2,
        classifier=classifier,
        probe_state=_probe_state(),
        candidate_outputs={
            "probabilities": probabilities,
            "modality_tokens": tokens,
        },
        config={
            "modis": {
                "workset_size": 4,
                "grid_k": 2,
                "bisect_steps": 1,
                "fusion_batch_size": 4,
                "probe_eval_batch_size": 4,
            }
        },
    )

    assert len(result.selected_ids) == 2
    assert len(set(result.selected_positions)) == 2
    assert set(result.scores) == {
        "disagreement",
        "instability",
        "dominance",
        "sufficiency_penalty",
        "combined",
    }
    assert torch.isfinite(result.scores["combined"]).all()
    assert result.diagnostics["method"] == "modis"


def test_mobius_inversion_recovers_additive_and_synergy_terms() -> None:
    values = torch.tensor([[1.0, 3.0, 4.0, 10.0]])

    interactions = mobius_inversion(values, num_modalities=2)
    decomposition = decompose_lattice(values, num_modalities=2, eta=0.25)

    assert torch.equal(interactions, torch.tensor([[1.0, 2.0, 3.0, 4.0]]))
    assert torch.equal(decomposition.additive, torch.tensor([5.0]))
    assert torch.equal(decomposition.synergy, torch.tensor([4.0]))
    assert torch.equal(decomposition.score, torch.tensor([5.25]))


def _outputs(classifier: SumFusionClassifier, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
    features = classifier.fuse_from_tokens(tokens)
    return {
        "features": features,
        "probabilities": classifier.probabilities_from_fused(features),
        "modality_tokens": tokens,
    }


def test_mosaic_plugin_runs_fisher_lattice_selection() -> None:
    classifier = SumFusionClassifier()
    labeled_tokens = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 0.5]],
            [[0.0, 1.0], [0.5, 0.0]],
            [[-1.0, 0.0], [0.0, -0.5]],
            [[0.0, -1.0], [-0.5, 0.0]],
        ]
    )
    reference_tokens = torch.tensor(
        [
            [[0.8, 0.1], [0.1, 0.4]],
            [[-0.7, 0.2], [-0.2, -0.4]],
            [[0.1, 0.8], [0.4, 0.1]],
        ]
    )
    candidate_tokens = torch.tensor(
        [
            [[1.2, -0.2], [-0.1, 0.7]],
            [[-1.1, 0.3], [0.2, -0.8]],
            [[0.2, 1.1], [0.8, -0.1]],
        ]
    )
    labeled = _outputs(classifier, labeled_tokens)
    reference = _outputs(classifier, reference_tokens)
    reference["labels"] = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    candidates = _outputs(classifier, candidate_tokens)

    result = MoSAICPlugin().acquire(
        candidate_ids=(101, 102, 103),
        query_size=2,
        classifier=classifier,
        labeled_outputs=labeled,
        reference_outputs=reference,
        candidate_outputs=candidates,
        seed=7,
        config={
            "mosaic": {
                "eta": 0.25,
                "partners": 2,
                "workset_size": 3,
                "synergy_workset_size": 3,
                "damping": 0.1,
                "deflation_steps": 1,
                "fusion_batch_size": 16,
                "value_batch_size": 16,
            }
        },
    )

    assert len(result.selected_ids) == 2
    assert len(set(result.selected_positions)) == 2
    assert set(result.scores) == {"additive", "synergy", "total_gain", "combined"}
    assert torch.isfinite(result.scores["combined"]).all()
    assert result.diagnostics["coalitions"] == 4
    assert result.diagnostics["method"] == "mosaic"
