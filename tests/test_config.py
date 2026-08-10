from __future__ import annotations

import pytest

from mimic_comal.config import load_config


def test_default_a800_config_is_random_multimodal() -> None:
    config = load_config("configs/mimic_a800_144c.yaml")
    assert config["features"]["encoder"] == "multimodal_scratch"
    assert config["model"]["architecture"] == "multimodal_transformer_scratch"
    assert config["model"]["initialization"] == "random"


def test_pretrained_input_is_rejected_for_scratch_model(tmp_path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(
        "features:\n  encoder: multimodal_scratch\n"
        "model:\n  architecture: multimodal_transformer_scratch\n  initialization: random\n"
        "training:\n  pretrained_checkpoint: forbidden.pt\n"
    )
    with pytest.raises(ValueError, match="forbidden"):
        load_config(path)


def test_modis_configs_are_valid_multimodal_strategies() -> None:
    full = load_config("configs/mimic_modis.yaml")
    smoke = load_config("configs/mimic_modis_smoke.yaml")

    assert full["active_learning"]["strategy"] == "modis"
    assert smoke["active_learning"]["strategy"] == "modis"
    assert full["model"]["architecture"] == "multimodal_transformer_scratch"
