from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from mimic_comal.model import (
    CoMALModule,
    YangWuBertEncoderClassifier,
    comal_acquisition_scores,
    estimate_mm_comal_statistics,
    mm_comal_acquisition_scores,
    paper_comal_acquisition_scores,
    positive_similarity_thresholds,
    supervised_contrastive_loss,
)


class _TinyBert(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(32, hidden_dim)

    def forward(self, input_ids, token_type_ids, attention_mask):
        del token_type_ids
        values = self.embedding(input_ids)
        weights = attention_mask.unsqueeze(-1).float()
        pooled = (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        return SimpleNamespace(pooler_output=pooled)


def test_yang_wu_multimodal_multilabel_classifier_shapes() -> None:
    torch.manual_seed(3)
    model = YangWuBertEncoderClassifier(
        None,
        num_labels=1042,
        time_invariant_dim=5,
        time_invariant_hidden_dim=4,
        time_series_dim=7,
        time_series_hidden_dim=16,
        time_series_layers=1,
        time_series_heads=4,
        text_hidden_dim=8,
        dropout=0.0,
        text_encoder=_TinyBert(8),
    ).eval()
    batch = {
        "time_series": torch.randn(2, 3, 7),
        "time_invariant": torch.randn(2, 5),
        "input_ids": torch.randint(0, 32, (2, 6)),
        "token_type_ids": torch.zeros(2, 6, dtype=torch.long),
        "attention_mask": torch.ones(2, 6, dtype=torch.long),
    }
    result = model(batch, return_tokens=True)
    assert result["logits"].shape == (2, 1042)
    assert result["probabilities"].shape == (2, 1042)
    assert result["features"].shape == (2, 8)
    assert result["modality_tokens"].shape == (2, 3, 8)
    assert torch.allclose(
        result["features"], model.fuse_from_tokens(result["modality_tokens"]), atol=1e-6
    )
    assert torch.all((result["probabilities"] >= 0) & (result["probabilities"] <= 1))


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
    own_bg = module(torch.randn(8, 12), compute_reconstruction=False, compute_similarities="own_bg")
    assert own_bg["prototype_similarities"].shape == (8, 4, 2)


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


def test_mm_comal_reduces_to_paper_score_at_alpha_zero() -> None:
    torch.manual_seed(7)
    sample_count, view_count, label_count = 11, 4, 3
    probabilities = torch.rand(sample_count, label_count)
    labels = (torch.rand(9, label_count) > 0.55).float()
    labels[0] = 1.0
    labeled_similarity = torch.rand(9, view_count, label_count) * 2.0 - 1.0
    candidate_similarity = torch.rand(sample_count, view_count, label_count) * 2.0 - 1.0
    statistics = estimate_mm_comal_statistics(
        labeled_similarity,
        labels,
        threshold_estimator="midpoint",
    )
    expected_cardinality = labels.sum(dim=1).mean()
    mm_scores = mm_comal_acquisition_scores(
        probabilities,
        candidate_similarity,
        statistics,
        expected_cardinality=expected_cardinality,
        alpha=0.0,
    )
    paper_thresholds = positive_similarity_thresholds(
        None,
        labels,
        torch.empty(label_count + 1, 1),
        own_similarity=labeled_similarity[:, -1],
    )
    paper_scores = paper_comal_acquisition_scores(
        probabilities,
        None,
        torch.empty(label_count + 1, 1),
        paper_thresholds,
        expected_cardinality=expected_cardinality,
        own_similarity=candidate_similarity[:, -1],
    )
    assert torch.equal(mm_scores.combined, paper_scores.combined)


def test_multiview_comal_shapes() -> None:
    module = CoMALModule(12, 4, label_dim=6, prototype_dim=5, num_views=4)
    output = module(torch.randn(8, 4, 12), compute_similarities="own_bg")
    assert output["latent_features"].shape == (8, 4, 4, 5)
    assert output["prototype_similarities"].shape == (8, 4, 4, 2)
    assert output["reconstructed_features"].shape == (8, 12)
