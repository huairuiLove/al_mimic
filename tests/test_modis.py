from __future__ import annotations

import math

import torch
import torch.nn as nn

from modis.acquire import (
    critical_instability,
    decision_support,
    generalized_js_disagreement,
    modality_sufficiency,
    modality_thresholds,
    quantile_thresholds,
)
from modis.probes import ModalityProbes, estimate_reliability_weights


def test_modality_probes_stop_gradient_at_tokens() -> None:
    probes = ModalityProbes(num_modalities=3, token_dim=4, num_labels=2)
    tokens = torch.randn(5, 3, 4, requires_grad=True)

    probes(tokens).sum().backward()

    assert tokens.grad is None
    assert all(parameter.grad is not None for parameter in probes.parameters())


def test_generalized_js_is_bounded_and_modality_permutation_invariant() -> None:
    generator = torch.Generator().manual_seed(5)
    probabilities = torch.rand(17, 3, 6, generator=generator)
    raw_weights = torch.rand(3, 6, generator=generator)
    weights = raw_weights / raw_weights.sum(dim=0, keepdim=True)
    support = torch.ones(17, 6, dtype=torch.bool)

    disagreement, per_label = generalized_js_disagreement(probabilities, weights, support)
    permutation = torch.tensor([2, 0, 1])
    permuted, permuted_per_label = generalized_js_disagreement(
        probabilities[:, permutation], weights[permutation], support
    )

    assert torch.all(per_label >= 0)
    assert float(per_label.max()) <= math.log(3) + 1e-6
    assert torch.allclose(disagreement, permuted, atol=1e-6)
    assert torch.allclose(per_label, permuted_per_label, atol=1e-6)


def test_empty_decision_support_falls_back_to_fused_argmax() -> None:
    fused = torch.tensor([[0.1, 0.3, 0.2]])
    probes = torch.full((1, 3, 3), 0.1)
    fused_positive, probe_positive, support = decision_support(
        fused,
        probes,
        torch.full((3,), 0.9),
        torch.full((3, 3), 0.9),
    )

    assert not fused_positive.any()
    assert not probe_positive.any()
    assert support.tolist() == [[False, True, False]]


def test_all_positive_probe_is_not_fully_sufficient_for_subset_fusion_decision() -> None:
    fused_positive = torch.tensor([[True, False, False, False]])
    probe_positive = torch.ones(1, 3, 4, dtype=torch.bool)

    dominance, jaccard = modality_sufficiency(fused_positive, probe_positive)

    assert torch.allclose(jaccard, torch.full((1, 3), 0.25))
    assert dominance.item() == 0.25


def test_quantile_thresholds_match_positive_rate_across_fusion_and_probes() -> None:
    probabilities = torch.arange(1, 41, dtype=torch.float32).reshape(10, 4) / 41
    probe_probabilities = torch.stack(
        (probabilities, probabilities.flip(0), probabilities.roll(2, dims=0)), dim=1
    )
    prevalence = torch.full((4,), 0.3)
    fused_thresholds = quantile_thresholds(probabilities, prevalence)
    probe_thresholds = modality_thresholds(probe_probabilities, prevalence)

    fused_rate = (probabilities >= fused_thresholds).float().mean(dim=0)
    probe_rate = (probe_probabilities >= probe_thresholds[None]).float().mean(dim=0)

    assert torch.allclose(fused_rate, torch.full((4,), 0.3))
    assert torch.allclose(probe_rate, torch.full((3, 4), 0.3))


def test_empirical_bayes_zero_between_label_variance_fully_pools() -> None:
    label_column = torch.tensor([0, 1, 0, 1, 1, 0, 1, 0], dtype=torch.float32)
    labels = label_column[:, None].expand(-1, 4).clone()
    first = torch.where(labels > 0, torch.full_like(labels, 0.8), torch.full_like(labels, 0.2))
    second = torch.where(labels > 0, torch.full_like(labels, 0.7), torch.full_like(labels, 0.3))
    oof = torch.stack((first, second, torch.full_like(labels, 0.5)), dim=1)

    statistics = estimate_reliability_weights(labels, oof)

    expected = statistics.pooled_skill_scores[:, None].expand_as(statistics.shrunk_skill_scores)
    assert torch.allclose(statistics.shrunk_skill_scores, expected, atol=1e-6)
    assert torch.allclose(statistics.label_weights.sum(dim=0), torch.ones(4))


def test_reliability_estimation_stays_finite_for_saturated_probabilities() -> None:
    labels = torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
    saturated = labels[:, None, :].expand(-1, 3, -1).clone()

    statistics = estimate_reliability_weights(labels, saturated)

    assert torch.isfinite(statistics.skill_scores).all()
    assert torch.isfinite(statistics.label_weights).all()


class _LinearFusion(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature_dim = 1
        self.classifier = nn.Linear(1, 1)
        with torch.no_grad():
            self.classifier.weight.fill_(1.0)
            self.classifier.bias.zero_()

    def fuse_from_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens.mean(dim=1)


def test_grid_instability_is_conservative_against_dense_scan() -> None:
    classifier = _LinearFusion()
    tokens = torch.tensor([[[1.0], [-0.2], [-0.2]]])
    prototypes = torch.zeros(3, 1)
    base_decision = torch.sigmoid(classifier.classifier(classifier.fuse_from_tokens(tokens))) >= 0.5
    result = critical_instability(
        classifier,  # type: ignore[arg-type]
        tokens,
        prototypes,
        base_decision,
        torch.tensor([0.5]),
        torch.tensor([1.0, 0.0, 0.0]),
        grid_k=8,
        bisect_steps=0,
        fusion_batch_size=32,
    )
    dense_alphas = torch.linspace(0.999, 0.0, 1000)
    changed_alphas = []
    for alpha in dense_alphas:
        intervened = tokens.clone()
        intervened[:, 0] = alpha * tokens[:, 0]
        decision = (
            torch.sigmoid(classifier.classifier(classifier.fuse_from_tokens(intervened))) >= 0.5
        )
        if bool((decision != base_decision).any()):
            changed_alphas.append(float(alpha))
    dense_supremum = max(changed_alphas)

    assert result.per_modality[0, 0].item() <= dense_supremum + 1e-6
    assert dense_supremum - result.per_modality[0, 0].item() < 1 / 8
