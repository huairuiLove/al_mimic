from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from al_mimic.tasks.mimic_iii.mixup import (
    MixupConfig,
    label_space_mixup,
    modality_space_mixup,
)


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
    single = label_space_mixup(torch.randn(1, 8), torch.ones(1, 40), config, np.random.default_rng(0))
    assert single is None


def test_invalid_settings_are_rejected() -> None:
    with pytest.raises(ValueError):
        MixupConfig.from_config({"mixup": {"enabled": True, "alpha": 0.0}})
    with pytest.raises(ValueError):
        MixupConfig.from_config({"mixup": {"enabled": True, "space": "pixels"}})
    with pytest.raises(ValueError):
        MixupConfig.from_config({"mixup": {"enabled": True, "pairing": "nearest"}})
    with pytest.raises(ValueError):
        MixupConfig.from_config({"mixup": {"enabled": True, "anchor_quantile": 0.0}})
    with pytest.raises(ValueError):
        MixupConfig.from_config({"mixup": {"enabled": True, "weight": -1.0}})


def test_defaults_keep_mixup_off_for_the_formal_protocol() -> None:
    config = MixupConfig.from_config({})
    assert config.enabled is False
    assert config.space == "fused"


def test_modality_mixup_uses_one_shared_plan_for_all_views_and_labels() -> None:
    labels = torch.eye(8)
    first = torch.arange(16, dtype=torch.float32).reshape(8, 2)
    second = first * 10.0 + 3.0
    config = MixupConfig(
        enabled=True,
        space="modalities",
        pairing="random",
        anchor_quantile=1.0,
        keep_anchor=False,
    )

    mixed = modality_space_mixup((first, second, None), labels, config, np.random.default_rng(9))

    assert mixed is not None
    mixed_first, mixed_second, mixed_static = mixed.modalities
    assert mixed_first is not None and mixed_second is not None
    assert mixed_static is None
    assert torch.allclose(mixed_second, mixed_first * 10.0 + 3.0, atol=1e-6)
    assert float(mixed.labels.min()) >= 0.0
    assert float(mixed.labels.max()) <= 1.0
    assert mixed.diagnostics["virtual_samples"] == mixed.labels.shape[0]


def test_modality_mixup_rejects_misaligned_views() -> None:
    config = MixupConfig(enabled=True, space="modalities")
    labels = torch.ones(4, 2)
    with pytest.raises(ValueError, match="same batch dimension"):
        modality_space_mixup(
            (torch.randn(4, 3), torch.randn(3, 5)),
            labels,
            config,
            np.random.default_rng(0),
        )


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
    from al_mimic.tasks.mimic_iii.model import YangWuBertEncoderClassifier

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
    model, batch = _tiny_classifier()
    reference = model(batch)["logits"]
    text, series, static = model.encode_modalities(batch)
    explicit = model.forward_from_modalities(text, series, static)["logits"]
    assert torch.allclose(reference, explicit, atol=1e-6)


def test_mixup_term_produces_finite_gradients_through_the_head() -> None:
    import torch.nn.functional as F

    model, batch = _tiny_classifier()
    text, series, static = model.encode_modalities(batch)
    fused = model.gate(text, static, series)
    mixed = label_space_mixup(
        fused,
        batch["labels"],
        MixupConfig(enabled=True, anchor_quantile=1.0),
        np.random.default_rng(0),
    )
    assert mixed is not None
    loss = F.binary_cross_entropy_with_logits(model.classifier(fused), batch["labels"])
    loss = loss + F.binary_cross_entropy_with_logits(model.classifier(mixed.features), mixed.labels)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(model.classifier.weight.grad).all()
    assert torch.isfinite(model.time_series_projection.weight.grad).all()


def test_modality_mixup_backpropagates_through_all_encoders_and_gate() -> None:
    import torch.nn.functional as F

    model, batch = _tiny_classifier()
    text, series, static = model.encode_modalities(batch)
    mixed = modality_space_mixup(
        (text, series, static),
        batch["labels"],
        MixupConfig(
            enabled=True,
            space="modalities",
            pairing="random",
            anchor_quantile=1.0,
            keep_anchor=False,
        ),
        np.random.default_rng(4),
    )
    assert mixed is not None
    output = model.forward_from_modalities(*mixed.modalities)
    loss = F.binary_cross_entropy_with_logits(output["logits"], mixed.labels)
    loss.backward()

    assert torch.isfinite(loss)
    assert torch.isfinite(model.text_encoder.embedding.weight.grad).all()
    assert torch.isfinite(model.time_series_projection.weight.grad).all()
    assert model.time_invariant_encoder is not None
    assert torch.isfinite(model.time_invariant_encoder.weight.grad).all()
    assert torch.isfinite(model.gate.time_series_weight.weight.grad).all()
