from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from al_mimic.tasks.mds_ed.ecg import PreparedEcgRecord
from al_mimic.tasks.mds_ed.memmap import build_prepared_memmap
from al_mimic.tasks.mds_ed.tabular import TabularSpec
from al_mimic.tasks.mds_ed.training import (
    SupervisedTrainingConfig,
    TrainingDependencyError,
    _load_training_dependencies,
    train_supervised,
)


def _build_prepared_training_data(root: Path) -> None:
    records = []
    for index in range(20):
        waveform_path = root / f"waveform-{index}.npy"
        waveform = np.full((32, 12), index / 20, dtype=np.float32)
        np.save(waveform_path, waveform, allow_pickle=False)
        records.append(
            PreparedEcgRecord(
                subject_id=1_000 + index,
                study_id=2_000 + index,
                waveform_path=waveform_path,
                samples=32,
                channels=12,
            )
        )
    build_prepared_memmap(records, root)
    spec = TabularSpec(
        raw_feature_columns=("feature_a", "feature_b"),
        input_columns=("feature_a", "feature_b"),
        continuous_columns=("feature_a",),
        categorical_columns=("feature_b",),
        mask_columns=(),
        medians=(0.0, 0.0),
        category_values=((0.0, 1.0),),
        diagnosis_columns=tuple(f"diagnoses_{index}" for index in range(1_428)),
        deterioration_columns=tuple(f"deterioration_{index}" for index in range(15)),
    )
    spec.save(root / "tabular_spec.json")
    labels = np.zeros((20, 1_428), dtype=np.float32)
    labels[np.arange(20), np.arange(20)] = 1.0
    shard = root / "tabular-00000.npz"
    np.savez(
        shard,
        study_id=np.arange(2_000, 2_020, dtype=np.int64),
        subject_id=np.arange(1_000, 1_020, dtype=np.int64),
        fold=np.arange(20, dtype=np.int16),
        continuous=np.arange(20, dtype=np.float32).reshape(-1, 1),
        categorical=(np.arange(20) % 2).astype(np.int64).reshape(-1, 1),
        labels=labels,
    )
    (root / "tabular_manifest.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "task": "diagnoses",
                "rows": 20,
                "continuous_dim": 1,
                "categorical_dim": 1,
                "category_sizes": [2],
                "shards": [{"file": shard.name, "rows": 20}],
            }
        ),
        encoding="utf-8",
    )


def test_package_import_does_not_eagerly_load_training_dependencies() -> None:
    source_root = Path(__file__).resolve().parents[4] / "src"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(source_root)
    code = (
        "import sys; import al_mimic.tasks.mds_ed; "
        "assert 'torch' not in sys.modules; assert 'wfdb' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], env=environment, check=True)


def test_training_dependency_error_is_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_import(name: str):
        if name == "torch":
            raise ImportError("missing")
        raise AssertionError(name)

    monkeypatch.setattr("al_mimic.tasks.mds_ed.training.importlib.import_module", missing_import)
    with pytest.raises(TrainingDependencyError, match="PyTorch.*no external MDS-ED"):
        _load_training_dependencies()


def test_native_supervised_adapter_trains_on_synthetic_data(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    _build_prepared_training_data(tmp_path)

    result = train_supervised(
        tmp_path,
        tmp_path / "run",
        SupervisedTrainingConfig(
            epochs=1,
            batch_size=8,
            device="cpu",
            model_dim=8,
            temporal_layers=1,
            tabular_dim=4,
            dropout=0.0,
        ),
    )

    assert result["backend"] == "native_temporal_adapter"
    assert Path(result["checkpoint"]).is_file()
    assert len(result["history"]["train_loss"]) == 1
    assert np.isfinite(result["history"]["val_loss"][0])
