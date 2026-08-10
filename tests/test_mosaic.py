from __future__ import annotations

import torch
import torch.nn as nn

from mosaic.acquire import acquire_mosaic
from mosaic.design import FisherDesign
from mosaic.lattice import decompose_lattice, mobius_inversion


def test_mobius_inversion_reconstructs_coalition_values() -> None:
    torch.manual_seed(4)
    values = torch.randn(6, 8)
    interactions = mobius_inversion(values, num_modalities=3)
    reconstructed = torch.zeros_like(values)
    for coalition in range(8):
        subsets = [subset for subset in range(8) if subset & coalition == subset]
        reconstructed[:, coalition] = interactions[:, subsets].sum(dim=1)
    decomposition = decompose_lattice(values, num_modalities=3, eta=0.25)
    assert torch.allclose(reconstructed, values, atol=1e-6)
    assert decomposition.score.shape == (6,)


def test_fisher_design_gain_and_deflation_are_finite() -> None:
    torch.manual_seed(5)
    labeled_features = torch.randn(12, 5)
    labeled_probabilities = torch.sigmoid(torch.randn(12, 3))
    reference_features = torch.randn(7, 5)
    reference_probabilities = torch.sigmoid(torch.randn(7, 3))
    reference_labels = (torch.rand(7, 3) > 0.5).float()
    design = FisherDesign.build(
        labeled_features,
        labeled_probabilities,
        reference_features,
        reference_probabilities,
        reference_labels,
        damping=0.1,
    )
    candidate = torch.randn(4, 5)
    probabilities = torch.sigmoid(torch.randn(4, 3))
    before = design.marginal_gain(candidate, probabilities)
    design.deflate(candidate[0], probabilities[0])
    after = design.marginal_gain(candidate, probabilities)
    assert torch.isfinite(before).all()
    assert torch.isfinite(after).all()
    assert after[0] <= before[0] + 1e-6


class _TwoModalityFusion(nn.Module):
    modality_names = ("image", "metadata")

    def __init__(self) -> None:
        super().__init__()
        self.feature_dim = 4
        self.classifier = nn.Linear(4, 3)

    def fuse_from_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens.sum(dim=1)

    def probabilities_from_fused(self, fused: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.classifier(fused))


def test_mosaic_supports_brset_two_modality_lattice() -> None:
    torch.manual_seed(12)
    classifier = _TwoModalityFusion().eval()

    def outputs(count: int) -> dict[str, torch.Tensor]:
        tokens = torch.randn(count, 2, 4)
        features = classifier.fuse_from_tokens(tokens)
        return {
            "modality_tokens": tokens,
            "features": features,
            "probabilities": classifier.probabilities_from_fused(features),
            "labels": (torch.rand(count, 3) > 0.6).float(),
        }

    labeled = outputs(10)
    candidates = outputs(7)
    result = acquire_mosaic(
        classifier,
        labeled,
        labeled,
        candidates,
        query_size=2,
        config={
            "mosaic": {
                "partners": 2,
                "mixup_closure_samples": 0,
                "workset_size": 7,
                "synergy_workset_size": 7,
                "deflation_steps": 1,
                "damping": 0.1,
                "fusion_batch_size": 32,
                "value_batch_size": 32,
            }
        },
        seed=4,
    )
    assert result.selected_positions.shape == (2,)
    assert result.diagnostics["modalities"] == 2
    assert result.diagnostics["coalitions"] == 4
