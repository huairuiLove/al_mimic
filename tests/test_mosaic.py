from __future__ import annotations

import torch

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
