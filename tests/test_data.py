from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mimic_comal.multimodal_data import YangWuFeatureStore, audit_split_hdf5


def _write_split(path: Path, *, invalid_label: bool = False) -> dict:
    h5py = pytest.importorskip("h5py")
    counts = {"train": 3, "val": 2, "test": 1}
    with h5py.File(path, "w") as handle:
        root = handle.create_group("with_notes")
        for split, count in counts.items():
            group = root.create_group(split)
            group.create_dataset("X", data=np.zeros((count, 2, 3), dtype=np.float32))
            group.create_dataset("s", data=np.zeros((count, 2), dtype=np.float32))
            group.create_dataset("input_ids", data=np.ones((count, 5), dtype=np.int64))
            group.create_dataset("token_type_ids", data=np.zeros((count, 5), dtype=np.int64))
            group.create_dataset("attention_mask", data=np.ones((count, 5), dtype=np.int64))
            labels = np.zeros((count, 4), dtype=np.float32)
            labels[:, 0] = 1
            if invalid_label and split == "test":
                labels[0, 0] = 0
            group.create_dataset("label", data=labels)
    checkpoint = path.parent / "clinicalbert.bin"
    checkpoint.touch()
    return {
        "dataset": {
            "split_hdf5": str(path),
            "split_group": "with_notes",
            "clinicalbert_checkpoint": str(checkpoint),
        },
        "preprocessing": {
            "expected_total_samples": 6,
            "expected_label_count": 4,
            "observation_hours": 2,
            "time_series_dim": 3,
            "time_invariant_dim": 2,
            "max_note_tokens": 5,
        },
    }


def test_official_multimodal_hdf5_contract_and_global_indices(tmp_path: Path) -> None:
    config = _write_split(tmp_path / "splits.hdf5")
    audit = audit_split_hdf5(config)
    assert audit.split_counts == {"train": 3, "val": 2, "test": 1}
    assert audit.total_samples == 6
    assert audit.label_count == 4
    store = YangWuFeatureStore(config)
    assert store.indices("train").tolist() == [0, 1, 2]
    assert store.indices("val").tolist() == [3, 4]
    assert store.indices("test").tolist() == [5]
    assert store.locate(4) == ("val", 1)


def test_visit_without_diagnosis_label_is_rejected(tmp_path: Path) -> None:
    config = _write_split(tmp_path / "splits.hdf5", invalid_label=True)
    with pytest.raises(ValueError, match="at least one ICD-9"):
        audit_split_hdf5(config)
