from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from al_mimic.tasks.mimic_iii.preprocessing.build_splits import assemble_splits, attach_arrays

TASK_DIR_NAME = "features/outcome=Diagnoses,T=48.0,dt=1.0"
# Population order is deliberately not grouped by split, so every test exercises
# the partition filter rather than an already-sorted frame.
COHORT = [
    # icustay, subject, partition, label vector
    (3001, 11, "train", np.array([1, 0, 0, 0], dtype=np.uint8)),
    (3002, 21, "val", np.array([0, 1, 0, 0], dtype=np.uint8)),
    (3003, 12, "train", np.array([1, 1, 0, 0], dtype=np.uint8)),
    (3004, 31, "test", np.array([0, 0, 1, 0], dtype=np.uint8)),
    (3005, 13, "train", np.array([0, 0, 0, 1], dtype=np.uint8)),
    (3006, 22, "val", np.array([1, 0, 0, 1], dtype=np.uint8)),
]
MISSING_NOTE_STAY = 3006
STEPS = 3
SERIES_FEATURES = 5
STATIC_FEATURES = 4
TOKENS = 8


def _write_artifacts(base: Path, *, skip_note_stay: int | None = MISSING_NOTE_STAY) -> Path:
    task_dir = base / TASK_DIR_NAME
    (base / "population").mkdir(parents=True, exist_ok=True)
    (base / "prep").mkdir(parents=True, exist_ok=True)
    task_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame(
        {
            "ID": [row[0] for row in COHORT],
            "Diagnoses_LABEL": [row[3] for row in COHORT],
        }
    )
    frame.to_hdf(base / "population/population.hdf5", key="Diagnoses_48.0h", mode="w")
    pd.DataFrame({"ID": [row[0] for row in COHORT]}).to_csv(task_dir / "IDs.csv", index=False)
    pd.DataFrame(
        {
            "ICUSTAY_ID": [row[0] for row in COHORT],
            "SUBJECT_ID": [row[1] for row in COHORT],
            "FIRST_CAREUNIT": ["MICU", "SICU", "MICU", "CCU", "MICU", "CCU"],
            "partition": [row[2] for row in COHORT],
        }
    ).to_csv(base / "prep/icustays_MV.csv", index=False)

    with_note = [row for row in COHORT if row[0] != skip_note_stay]
    notes = pd.DataFrame({"ICUSTAY_ID": [row[0] for row in with_note]})
    for field, fill in (
        ("input_ids", 101),
        ("token_type_ids", 0),
        ("attention_mask", 1),
    ):
        notes[field] = [np.full(TOKENS, fill, dtype=np.int64) for _ in with_note]
    notes.to_hdf(task_dir / "notes.hdf5", key="notes", mode="w")

    count = len(COHORT)
    rng = np.random.default_rng(7)
    with h5py.File(task_dir / "Xs.hdf5", "w") as handle:
        handle.create_dataset("X", data=rng.normal(size=(count, STEPS, SERIES_FEATURES)).astype(np.float32))
        handle.create_dataset("s", data=rng.normal(size=(count, STATIC_FEATURES)).astype(np.float32))
    return task_dir


def _legacy_splits(path: Path, *, swap_train_rows: bool = False) -> None:
    """An artifact from the pre-subject_id pipeline, in population/partition order."""
    by_split: dict[str, list[tuple]] = {"train": [], "val": [], "test": []}
    for row in COHORT:
        by_split[row[2]].append(row)
    if swap_train_rows and len(by_split["train"]) >= 2:
        by_split["train"][0], by_split["train"][1] = by_split["train"][1], by_split["train"][0]
    with h5py.File(path, "w") as handle:
        root = handle.create_group("with_notes")
        for split, rows in by_split.items():
            group = root.create_group(split)
            group.create_dataset("X", data=np.zeros((len(rows), STEPS, SERIES_FEATURES), dtype=np.float32))
            group.create_dataset("s", data=np.zeros((len(rows), STATIC_FEATURES), dtype=np.float32))
            for field, fill in (
                ("input_ids", 101),
                ("token_type_ids", 0),
                ("attention_mask", 1),
            ):
                group.create_dataset(field, data=np.full((len(rows), TOKENS), fill, dtype=np.int64))
            group.create_dataset("label", data=np.stack([row[3].astype(np.int64) for row in rows]))


def _expected(split: str) -> list[tuple]:
    return [row for row in COHORT if row[2] == split]


def test_assemble_writes_the_loader_contract(tmp_path: Path) -> None:
    task_dir = _write_artifacts(tmp_path)

    output = assemble_splits(tmp_path, "Diagnoses", 48.0, 1.0)

    assert output == task_dir / "splits.hdf5"
    with h5py.File(output, "r") as handle:
        root = handle["with_notes"]
        assert json_careunits(root) == {"CCU": 0, "MICU": 1, "SICU": 2}
        assert int(root.attrs["observation_hours"]) == STEPS
        for split in ("train", "val", "test"):
            rows = _expected(split)
            group = root[split]
            assert group["X"].shape == (len(rows), STEPS, SERIES_FEATURES)
            assert group["s"].shape == (len(rows), STATIC_FEATURES)
            assert group["subject_id"].shape == (len(rows),)
            assert np.array_equal(np.asarray(group["subject_id"]), [row[1] for row in rows])
            assert np.array_equal(
                np.asarray(group["label"]), np.stack([row[3] for row in rows]).astype(np.int64)
            )
        train_position = _expected("train").index(next(r for r in _expected("train") if r[0] == 3001))
        assert bool(np.asarray(root["train"]["notes_available"])[train_position]) is True


def json_careunits(root: h5py.Group) -> dict:
    import json

    return json.loads(root.attrs["careunit_codes"])


def test_assemble_fills_missing_notes_with_an_empty_sequence(tmp_path: Path) -> None:
    _write_artifacts(tmp_path)

    output = assemble_splits(tmp_path, "Diagnoses", 48.0, 1.0)

    with h5py.File(output, "r") as handle:
        group = handle["with_notes"]["val"]
        position = _expected("val").index(next(r for r in _expected("val") if r[0] == MISSING_NOTE_STAY))
        assert bool(np.asarray(group["notes_available"])[position]) is False
        tokens = np.asarray(group["input_ids"][position])
        mask = np.asarray(group["attention_mask"][position])
        assert tokens[0] == 101 and tokens[1] == 102
        assert int(mask.sum()) == 2
        assert bool(np.asarray(handle["with_notes"]["train"]["notes_available"]).all()) is True


def test_assemble_refuses_to_overwrite(tmp_path: Path) -> None:
    _write_artifacts(tmp_path)
    assemble_splits(tmp_path, "Diagnoses", 48.0, 1.0)

    with pytest.raises(SystemExit, match="already exists"):
        assemble_splits(tmp_path, "Diagnoses", 48.0, 1.0)


def test_assemble_rejects_misaligned_population_order(tmp_path: Path) -> None:
    task_dir = _write_artifacts(tmp_path)
    shuffled = [row[0] for row in COHORT][::-1]
    pd.DataFrame({"ID": shuffled}).to_csv(task_dir / "IDs.csv", index=False)

    with pytest.raises(SystemExit, match="population order does not match"):
        assemble_splits(tmp_path, "Diagnoses", 48.0, 1.0)


def test_attach_patches_a_legacy_artifact_idempotently(tmp_path: Path) -> None:
    task_dir = _write_artifacts(tmp_path)
    legacy = task_dir / "splits.hdf5"
    _legacy_splits(legacy)

    attach_arrays(tmp_path, "Diagnoses", 48.0, 1.0, splits_path=legacy)
    attach_arrays(tmp_path, "Diagnoses", 48.0, 1.0, splits_path=legacy)

    with h5py.File(legacy, "r") as handle:
        root = handle["with_notes"]
        for split in ("train", "val", "test"):
            rows = _expected(split)
            assert np.array_equal(np.asarray(root[split]["subject_id"]), [row[1] for row in rows])
            assert "careunit_code" in root[split]
            assert "notes_available" in root[split]
        val_position = _expected("val").index(next(r for r in _expected("val") if r[0] == MISSING_NOTE_STAY))
        assert bool(np.asarray(root["val"]["notes_available"])[val_position]) is False


def test_attach_refuses_a_swapped_label_order(tmp_path: Path) -> None:
    task_dir = _write_artifacts(tmp_path)
    legacy = task_dir / "splits.hdf5"
    _legacy_splits(legacy, swap_train_rows=True)

    with pytest.raises(SystemExit, match="label order does not match"):
        attach_arrays(tmp_path, "Diagnoses", 48.0, 1.0, splits_path=legacy)


def test_attach_requires_an_existing_artifact(tmp_path: Path) -> None:
    _write_artifacts(tmp_path)

    with pytest.raises(SystemExit, match="does not exist"):
        attach_arrays(tmp_path, "Diagnoses", 48.0, 1.0, splits_path=tmp_path / "absent.hdf5")
