from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from al_mimic.tasks.mimic_iii.preprocessing.notes import TOKEN_FIELDS, read_notes


def _token_rows(count: int, length: int = 8) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(3)
    return {
        "input_ids": rng.integers(0, 28996, size=(count, length)),
        "token_type_ids": np.zeros((count, length), dtype=np.int64),
        "attention_mask": np.ones((count, length), dtype=np.int64),
    }


def _write_streamed(path: Path, stays: np.ndarray, rows: dict[str, np.ndarray]) -> None:
    with h5py.File(path, "w") as handle:
        handle.create_dataset("ICUSTAY_ID", data=stays.astype(np.int64))
        for field in TOKEN_FIELDS:
            handle.create_dataset(field, data=rows[field].astype(np.int32))


def _write_pandas(path: Path, stays: np.ndarray, rows: dict[str, np.ndarray]) -> None:
    frame = pd.DataFrame({"ICUSTAY_ID": stays})
    for field in TOKEN_FIELDS:
        frame[field] = list(rows[field].astype(np.int64))
    frame.to_hdf(str(path), key="notes", mode="w")


def test_streamed_and_pandas_layouts_read_identically(tmp_path: Path) -> None:
    """The rebuilt cohorts stream notes to disk; the 48h artifact predates that."""
    stays = np.array([200001, 200028, 200099])
    rows = _token_rows(len(stays))
    streamed_path = tmp_path / "streamed.hdf5"
    pandas_path = tmp_path / "pandas.hdf5"
    _write_streamed(streamed_path, stays, rows)
    _write_pandas(pandas_path, stays, rows)

    streamed = read_notes(streamed_path)
    legacy = read_notes(pandas_path)

    assert streamed["ICUSTAY_ID"].tolist() == legacy["ICUSTAY_ID"].tolist()
    for field in TOKEN_FIELDS:
        assert np.array_equal(np.stack(streamed[field].to_numpy()), np.stack(legacy[field].to_numpy()))


def test_streamed_tokens_widen_to_int64(tmp_path: Path) -> None:
    """Stored narrow to halve the file; the model path expects 64-bit ids."""
    stays = np.array([1, 2])
    path = tmp_path / "streamed.hdf5"
    _write_streamed(path, stays, _token_rows(len(stays)))

    frame = read_notes(path)
    assert np.asarray(frame["input_ids"].iloc[0]).dtype == np.int64


def test_reading_a_frame_without_token_datasets_falls_back(tmp_path: Path) -> None:
    stays = np.array([7, 8])
    path = tmp_path / "pandas.hdf5"
    _write_pandas(path, stays, _token_rows(len(stays)))

    with h5py.File(path, "r") as handle:
        assert "input_ids" not in handle or not isinstance(handle["input_ids"], h5py.Dataset)
    assert read_notes(path)["ICUSTAY_ID"].tolist() == stays.tolist()


def test_missing_note_file_raises(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        read_notes(tmp_path / "absent.hdf5")
