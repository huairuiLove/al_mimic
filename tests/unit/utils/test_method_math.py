from __future__ import annotations

import torch
import torch.nn as nn

from al_mimic.utils.contrastive import (
    multiview_contrastive_loss,
    supervised_contrastive_loss,
)
from al_mimic.utils.fusion import fuse_token_batches, probabilities_from_fused
from al_mimic.utils.linear import augment_bias
from al_mimic.utils.prototypes import positive_similarity_thresholds


class FusionModel(nn.Module):
    modality_names = ("a", "b")
    feature_dim = 3

    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Linear(3, 2, bias=False)
        with torch.no_grad():
            self.classifier.weight.copy_(torch.tensor([[1.0, 0.0, -1.0], [0.0, 1.0, 1.0]]))

    def fuse_from_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens.sum(dim=-2)


def test_augment_bias_preserves_leading_dimensions_and_dtype() -> None:
    features = torch.arange(12, dtype=torch.float64).reshape(2, 2, 3)
    augmented = augment_bias(features)

    assert augmented.shape == (2, 2, 4)
    assert augmented.dtype == torch.float64
    assert torch.equal(augmented[..., :-1], features)
    assert torch.equal(augmented[..., -1], torch.ones(2, 2, dtype=torch.float64))


def test_fusion_helpers_support_nested_token_batches() -> None:
    model = FusionModel()
    model.train()
    tokens = torch.arange(36, dtype=torch.float32).reshape(2, 3, 2, 3) / 10

    fused = fuse_token_batches(model, tokens, batch_size=2)
    probabilities = probabilities_from_fused(model, fused)

    assert fused.shape == (2, 3, 3)
    assert model.training
    assert torch.allclose(fused, tokens.sum(dim=-2))
    assert probabilities.shape == (2, 3, 2)
    assert torch.all((0.0 <= probabilities) & (probabilities <= 1.0))


def test_supervised_contrastive_loss_is_finite_and_differentiable() -> None:
    torch.manual_seed(4)
    features = torch.randn(4, 2, 3, requires_grad=True)
    labels = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])

    loss = supervised_contrastive_loss(features, labels, temperature=0.2)
    loss.backward()

    assert torch.isfinite(loss)
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()


def test_cross_view_loss_without_eligible_pair_returns_finite_zero() -> None:
    features = torch.randn(1, 2, 3, requires_grad=True)
    labels = torch.tensor([[1.0, 0.0]])
    view_ids = torch.zeros_like(labels, dtype=torch.long)

    loss = supervised_contrastive_loss(
        features,
        labels,
        view_ids=view_ids,
    )
    loss.backward()

    assert torch.equal(loss, torch.tensor(0.0))
    assert features.grad is not None
    assert torch.equal(features.grad, torch.zeros_like(features.grad))


def test_multiview_contrastive_loss_is_finite_for_cross_view_training() -> None:
    torch.manual_seed(5)
    latent = torch.randn(3, 2, 2, 4)
    labels = torch.tensor([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])

    loss = multiview_contrastive_loss(
        latent,
        labels,
        temperature=0.2,
        cross_view_weight=0.5,
    )

    assert torch.isfinite(loss)
    assert float(loss) > 0.0


def test_positive_similarity_thresholds_use_positive_midpoints() -> None:
    labels = torch.tensor([[1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]])
    similarity = torch.tensor([[0.3, -0.4], [0.9, 0.2], [-0.3, 0.8], [-0.8, -0.5]])

    thresholds = positive_similarity_thresholds(labels, own_similarity=similarity)

    assert torch.allclose(thresholds, torch.tensor([0.6, 0.5]))
