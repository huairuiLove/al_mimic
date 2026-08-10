from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from brset_al.config import load_config
from brset_al.data import _parse_numeric
from brset_al.metrics import fit_f1_thresholds, multilabel_metrics
from brset_al.model import BrsetMultimodalClassifier, initialize_fusion_layers
from brset_al.training import aggregate_patient_outputs


class _TinyImageEncoder(nn.Module):
    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(3, output_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.projection(images.mean(dim=(-2, -1)))


def test_formal_brset_configs_are_full_multimodal_experiments() -> None:
    for name, strategy in (
        ("comal", "comal"),
        ("mm_comal", "mm_comal"),
        ("modis", "modis"),
        ("mosaic", "mosaic"),
    ):
        config = load_config(f"configs/brset_{name}.yaml")
        assert config["preprocessing"]["expected_images"] == 16266
        assert config["preprocessing"]["expected_patients"] == 8524
        assert config["model"]["output_size"] == 13
        assert config["training"]["epochs"] == 20
        assert config["active_learning"]["rounds"] == 6
        assert config["active_learning"]["query_unit"] == "patient"
        assert config["active_learning"]["strategy"] == strategy


def test_brset_shortcut_config_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(
        f"extends: {Path('configs/brset_base.yaml').resolve()}\ntraining:\n  dry_run: true\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="shortcut setting"):
        load_config(path)


def test_brset_multimodal_classifier_exposes_two_exact_fusion_tokens() -> None:
    torch.manual_seed(2)
    model = BrsetMultimodalClassifier(
        metadata_dim=9,
        num_labels=13,
        image_feature_dim=16,
        metadata_hidden_dim=8,
        fusion_dim=12,
        dropout=0.0,
        image_encoder=_TinyImageEncoder(16),
    )
    initialize_fusion_layers(model)
    model.eval()
    result = model(
        {"image": torch.randn(4, 3, 32, 32), "metadata": torch.randn(4, 9)},
        return_tokens=True,
    )
    assert result["logits"].shape == (4, 13)
    assert result["modality_tokens"].shape == (4, 2, 12)
    assert torch.allclose(result["features"], model.fuse_from_tokens(result["modality_tokens"]), atol=1e-6)


def test_brset_numeric_cleanup_preserves_decimal_and_missing_semantics() -> None:
    assert _parse_numeric("0,5") == 0.5
    assert _parse_numeric("1O") == 10.0
    assert _parse_numeric("Nao") is None
    assert _parse_numeric("") is None


def test_brset_validation_thresholds_and_metrics_cover_all_labels() -> None:
    rng = np.random.default_rng(3)
    labels = rng.integers(0, 2, size=(30, 13), dtype=np.int8)
    probabilities = 0.15 + 0.7 * labels + rng.normal(0.0, 0.04, size=labels.shape)
    probabilities = probabilities.clip(0.0, 1.0)
    thresholds = fit_f1_thresholds(labels, probabilities)
    metrics = multilabel_metrics(labels, probabilities, thresholds)
    assert thresholds.shape == (13,)
    assert len(metrics["per_label"]) == 13
    assert metrics["macro_auroc"] > 0.9
    assert metrics["macro_auprc"] > 0.9


def test_patient_outputs_average_paired_eyes_before_acquisition() -> None:
    classifier = BrsetMultimodalClassifier(
        metadata_dim=2,
        num_labels=13,
        image_feature_dim=4,
        metadata_hidden_dim=3,
        fusion_dim=5,
        dropout=0.0,
        image_encoder=_TinyImageEncoder(4),
    ).eval()
    store = SimpleNamespace(
        patient_ids_array=np.asarray(["p1", "p1", "p2"], dtype=object),
        patient_targets=lambda patients: np.asarray(
            [[1.0] + [0.0] * 12 if patient == "p1" else [0.0, 1.0] + [0.0] * 11 for patient in patients],
            dtype=np.float32,
        ),
    )
    tokens = torch.randn(3, 2, 5)
    grouped, patients = aggregate_patient_outputs(
        classifier,
        store,  # type: ignore[arg-type]
        {
            "indices": torch.arange(3),
            "modality_tokens": tokens,
            "labels": torch.zeros(3, 13),
            "features": torch.zeros(3, 5),
            "probabilities": torch.zeros(3, 13),
        },
    )
    assert patients == ["p1", "p2"]
    assert torch.allclose(grouped["modality_tokens"][0], tokens[:2].mean(dim=0))
    assert grouped["labels"].shape == (2, 13)
