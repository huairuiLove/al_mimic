from __future__ import annotations

import torch

from mimic_comal.model import (
    CoMALModule,
    comal_acquisition_scores,
    paper_comal_acquisition_scores,
    positive_similarity_thresholds,
    supervised_contrastive_loss,
)


def test_comal_shapes_and_loss_are_finite() -> None:
    torch.manual_seed(1)
    module = CoMALModule(input_dim=12, num_labels=4, label_dim=6, prototype_dim=5)
    output = module(torch.randn(8, 12))
    labels = (torch.rand(8, 4) > 0.7).float()
    loss = supervised_contrastive_loss(output["latent_features"], labels, anchor_chunk_size=7)
    assert output["latent_features"].shape == (8, 4, 5)
    assert output["reconstructed_logits"].shape == (8, 4)
    assert output["prototype_similarities"].shape == (8, 4, 5)
    assert torch.isfinite(output["prototype_similarities"]).all()
    assert torch.isfinite(loss)
    latent_only = module(torch.randn(8, 12), compute_reconstruction=False, compute_similarities=True)
    assert "reconstructed_logits" not in latent_only
    assert latent_only["prototype_similarities"].shape == (8, 4, 5)


def test_acquisition_components_have_one_score_per_sample() -> None:
    torch.manual_seed(2)
    probabilities = torch.rand(9, 3)
    latents = torch.randn(9, 3, 5)
    prototypes = torch.randn(4, 5)
    scores = comal_acquisition_scores(probabilities, latents, prototypes, expected_cardinality=1.2)
    assert scores.combined.shape == (9,)
    assert torch.isfinite(scores.combined).all()
    labels = (torch.rand(9, 3) > 0.6).float()
    thresholds = positive_similarity_thresholds(latents, labels, prototypes)
    paper_scores = paper_comal_acquisition_scores(
        probabilities,
        latents,
        prototypes,
        thresholds,
        expected_cardinality=1.2,
    )
    assert paper_scores.combined.shape == (9,)
    assert torch.isfinite(paper_scores.combined).all()
