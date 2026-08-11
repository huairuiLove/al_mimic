from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mimic_comal.mixup import MixupConfig, label_space_mixup


def _batch(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(16, 8, generator=generator)
    labels = torch.zeros(16, 40)
    # First half are sparse anchors, second half are label-rich partners.
    for row in range(8):
        labels[row, torch.arange(2)] = 1.0
    for row in range(8, 16):
        labels[row, torch.arange(20)] = 1.0
    return features, labels


def test_disabled_mixup_returns_nothing() -> None:
    features, labels = _batch()
    config = MixupConfig(enabled=False)
    assert label_space_mixup(features, labels, config, np.random.default_rng(0)) is None


def test_zero_weight_short_circuits() -> None:
    features, labels = _batch()
    config = MixupConfig(enabled=True, weight=0.0)
    assert label_space_mixup(features, labels, config, np.random.default_rng(0)) is None


def test_targeted_mixup_raises_positive_mass_of_sparse_anchors() -> None:
    features, labels = _batch()
    config = MixupConfig(enabled=True, pairing="targeted", anchor_quantile=0.5, alpha=0.4)
    mixed = label_space_mixup(features, labels, config, np.random.default_rng(0))
    assert mixed is not None
    # Anchors are the eight sparse rows carrying two positives each.
    assert mixed.diagnostics["anchor_positive_mean"] == pytest.approx(2.0)
    assert mixed.diagnostics["mixed_positive_mean"] > 2.0
    assert mixed.features.shape[1] == features.shape[1]
    assert mixed.labels.shape[1] == labels.shape[1]
    assert mixed.features.shape[0] == mixed.labels.shape[0]


def test_keep_anchor_bounds_lambda_to_the_anchor_side() -> None:
    features, labels = _batch()
    config = MixupConfig(enabled=True, keep_anchor=True, anchor_quantile=1.0)
    mixed = label_space_mixup(features, labels, config, np.random.default_rng(1))
    assert mixed is not None
    assert 0.5 <= mixed.diagnostics["mean_lambda"] <= 1.0


def test_mixed_labels_stay_within_the_convex_hull() -> None:
    features, labels = _batch()
    config = MixupConfig(enabled=True, anchor_quantile=1.0)
    mixed = label_space_mixup(features, labels, config, np.random.default_rng(2))
    assert mixed is not None
    assert float(mixed.labels.min()) >= 0.0
    assert float(mixed.labels.max()) <= 1.0


def test_same_seed_reproduces_the_same_mix() -> None:
    features, labels = _batch()
    config = MixupConfig(enabled=True)
    first = label_space_mixup(features, labels, config, np.random.default_rng(7))
    second = label_space_mixup(features, labels, config, np.random.default_rng(7))
    assert first is not None and second is not None
    assert torch.equal(first.features, second.features)
    assert torch.equal(first.labels, second.labels)


def test_anchor_quantile_restricts_the_mixed_subset() -> None:
    features, labels = _batch()
    half = label_space_mixup(
        features, labels, MixupConfig(enabled=True, anchor_quantile=0.5), np.random.default_rng(3)
    )
    everything = label_space_mixup(
        features, labels, MixupConfig(enabled=True, anchor_quantile=1.0), np.random.default_rng(3)
    )
    assert half is not None and everything is not None
    assert half.diagnostics["anchor_fraction"] < everything.diagnostics["anchor_fraction"]


def test_single_row_batch_is_skipped() -> None:
    config = MixupConfig(enabled=True)
    single = label_space_mixup(
        torch.randn(1, 8), torch.ones(1, 40), config, np.random.default_rng(0)
    )
    assert single is None


def test_invalid_settings_are_rejected() -> None:
    with pytest.raises(ValueError):
        MixupConfig.from_config({"mixup": {"enabled": True, "alpha": 0.0}})
    with pytest.raises(ValueError):
        MixupConfig.from_config({"mixup": {"enabled": True, "pairing": "nearest"}})
    with pytest.raises(ValueError):
        MixupConfig.from_config({"mixup": {"enabled": True, "anchor_quantile": 0.0}})
    with pytest.raises(ValueError):
        MixupConfig.from_config({"mixup": {"enabled": True, "weight": -1.0}})


def test_defaults_keep_mixup_off_for_the_formal_protocol() -> None:
    assert MixupConfig.from_config({}).enabled is False


class _TinyBert(torch.nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(32, hidden_dim)

    def forward(self, input_ids, token_type_ids, attention_mask):
        del token_type_ids
        values = self.embedding(input_ids)
        weights = attention_mask.unsqueeze(-1).float()
        pooled = (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        return SimpleNamespace(pooler_output=pooled)


def _tiny_classifier():
    from mimic_comal.model import YangWuBertEncoderClassifier

    torch.manual_seed(5)
    model = YangWuBertEncoderClassifier(
        None,
        num_labels=12,
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
        "time_series": torch.randn(6, 3, 7),
        "time_invariant": torch.randn(6, 5),
        "input_ids": torch.randint(0, 32, (6, 6)),
        "token_type_ids": torch.zeros(6, 6, dtype=torch.long),
        "attention_mask": torch.ones(6, 6, dtype=torch.long),
        "labels": (torch.rand(6, 12) > 0.5).float(),
    }
    return model, batch


def test_explicit_encoder_path_matches_the_plain_forward() -> None:
    """The mixup branch re-implements the forward to expose the fused tensor."""
    model, batch = _tiny_classifier()
    reference = model(batch)["logits"]
    text, series, static = model.encode_modalities(batch)
    fused = model.gate(text, static, series)
    assert torch.allclose(reference, model.classifier(fused), atol=1e-6)


def test_mixup_term_produces_finite_gradients_through_the_head() -> None:
    import torch.nn.functional as F

    model, batch = _tiny_classifier()
    text, series, static = model.encode_modalities(batch)
    fused = model.gate(text, static, series)
    mixed = label_space_mixup(
        fused, batch["labels"], MixupConfig(enabled=True, anchor_quantile=1.0),
        np.random.default_rng(0),
    )
    assert mixed is not None
    loss = F.binary_cross_entropy_with_logits(model.classifier(fused), batch["labels"])
    loss = loss + F.binary_cross_entropy_with_logits(
        model.classifier(mixed.features), mixed.labels
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(model.classifier.weight.grad).all()
    assert torch.isfinite(model.time_series_projection.weight.grad).all()
