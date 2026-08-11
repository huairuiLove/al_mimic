from __future__ import annotations

from pathlib import Path

import pytest

from mimic_comal.cli import build_parser
from mimic_comal.config import load_config


def test_formal_configs_use_multimodal_multilabel_baseline() -> None:
    for path, strategy in (
        ("configs/mimic_comal.yaml", "comal"),
        ("configs/mimic_mm_comal.yaml", "mm_comal"),
        ("configs/mimic_modis.yaml", "modis"),
        ("configs/mimic_mosaic.yaml", "mosaic"),
        ("configs/mimic_random.yaml", "random"),
    ):
        config = load_config(path)
        assert config["model"]["architecture"] == "yang_wu_bertencoder"
        assert config["model"]["output_activation"] == "sigmoid"
        assert config["preprocessing"]["expected_total_samples"] == 10258
        assert config["preprocessing"]["expected_label_count"] == 915
        assert config["preprocessing"]["observation_hours"] == 48
        assert config["preprocessing"]["time_series_dim"] == 7749
        assert config["preprocessing"]["time_invariant_dim"] == 97
        assert config["training"]["epochs"] == 20
        assert config["active_learning"]["rounds"] == 6
        assert config["active_learning"]["strategy"] == strategy


def test_cross_round_checkpoint_input_is_rejected(tmp_path: Path) -> None:
    base = Path("configs/mimic_a800_144c.yaml").resolve()
    path = tmp_path / "invalid.yaml"
    path.write_text(
        f"extends: {base}\ntraining:\n  resume_from: forbidden.pt\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="shortcut setting"):
        load_config(path)


def test_shortened_epoch_config_is_rejected(tmp_path: Path) -> None:
    base = Path("configs/mimic_a800_144c.yaml").resolve()
    path = tmp_path / "invalid.yaml"
    path.write_text(f"extends: {base}\ntraining:\n  epochs: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="training.epochs"):
        load_config(path)


def test_formal_cli_does_not_allow_device_override() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["active", "--device", "cpu"])
