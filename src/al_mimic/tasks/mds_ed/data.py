"""Lazy datasets over native MDS-ED tabular shards and ECG memmaps."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .constants import TEST_FOLDS, TRAIN_FOLDS, VALIDATION_FOLDS

SPLIT_FOLDS = {
    "train": frozenset(TRAIN_FOLDS),
    "val": frozenset(VALIDATION_FOLDS),
    "validation": frozenset(VALIDATION_FOLDS),
    "test": frozenset(TEST_FOLDS),
}


class MdsEdPreparedDataset:
    """Map one tabular row to a waveform without retaining waveform bytes."""

    def __init__(self, prepared_dir: str | Path, split: str) -> None:
        if split not in SPLIT_FOLDS:
            raise ValueError(f"unknown MDS-ED split: {split}")
        self.root = Path(prepared_dir)
        self.manifest = json.loads((self.root / "tabular_manifest.json").read_text(encoding="utf-8"))
        self.shards = tuple(self.root / item["file"] for item in self.manifest["shards"])
        self._locations: list[tuple[int, int]] = []
        expected_studies: list[int] = []
        folds = SPLIT_FOLDS[split]
        for shard_index, path in enumerate(self.shards):
            with np.load(path, allow_pickle=False) as shard:
                selected = np.flatnonzero(np.isin(shard["fold"], tuple(folds)))
                self._locations.extend((shard_index, int(row)) for row in selected)
                expected_studies.extend(int(value) for value in shard["study_id"][selected])
        self._load_waveform_index(expected_studies)
        self._shard_index: int | None = None
        self._shard: Any | None = None
        self._memmaps: dict[int, np.memmap] = {}

    def _load_waveform_index(self, expected_studies: list[int]) -> None:
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("pandas is required to load prepared MDS-ED metadata") from exc
        frame = pd.read_pickle(self.root / "df_memmap.pkl")
        if frame["study_id"].duplicated().any():
            raise ValueError("prepared ECG metadata contains duplicate study_id values")
        study_to_record = dict(zip(frame["study_id"].astype(int), frame["data"].astype(int)))
        missing = sorted(set(expected_studies).difference(study_to_record))
        if missing:
            raise ValueError(
                f"prepared ECG memmap is missing {len(missing)} tabular studies; examples: {missing[:10]}"
            )
        self._record_indices = np.asarray(
            [study_to_record[study] for study in expected_studies], dtype=np.int64
        )
        with np.load(self.root / "memmap_meta.npz", allow_pickle=True) as metadata:
            self._starts = np.asarray(metadata["start"], dtype=np.int64)
            self._lengths = np.asarray(metadata["length"], dtype=np.int64)
            self._shapes = np.asarray(metadata["shape"], dtype=np.int64)
            self._file_indices = np.asarray(metadata["file_idx"], dtype=np.int64)
            self._dtype = np.dtype(str(np.asarray(metadata["dtype"]).item()))
            self._filenames = tuple(str(name) for name in np.asarray(metadata["filenames"]))

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_shard_index"] = None
        state["_shard"] = None
        state["_memmaps"] = {}
        return state

    def __len__(self) -> int:
        return len(self._locations)

    def _open_shard(self, index: int):
        if self._shard_index != index:
            if self._shard is not None:
                self._shard.close()
            self._shard = np.load(self.shards[index], allow_pickle=False)
            self._shard_index = index
        return self._shard

    def _waveform(self, dataset_index: int) -> np.ndarray:
        record_index = int(self._record_indices[dataset_index])
        file_index = int(self._file_indices[record_index])
        memmap = self._memmaps.get(file_index)
        if memmap is None:
            memmap = np.memmap(
                self.root / self._filenames[file_index],
                dtype=self._dtype,
                mode="r",
                shape=tuple(self._shapes[file_index]),
            )
            self._memmaps[file_index] = memmap
        start = int(self._starts[record_index])
        end = start + int(self._lengths[record_index])
        return np.asarray(memmap[start:end], dtype=np.float32).copy()

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        shard_index, row_index = self._locations[index]
        shard = self._open_shard(shard_index)
        return {
            "ecg": self._waveform(index),
            "continuous": np.asarray(shard["continuous"][row_index], dtype=np.float32),
            "categorical": np.asarray(shard["categorical"][row_index], dtype=np.int64),
            "labels": np.asarray(shard["labels"][row_index], dtype=np.float32),
            "study_id": np.asarray(shard["study_id"][row_index], dtype=np.int64),
            "subject_id": np.asarray(shard["subject_id"][row_index], dtype=np.int64),
        }


def make_dataloaders(
    prepared_dir: str | Path,
    *,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = False,
):
    """Create PyTorch loaders only when supervised training is requested."""
    try:
        from torch.utils.data import DataLoader
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("PyTorch is required to create MDS-ED training loaders") from exc
    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
    loaders = {}
    for split in ("train", "val", "test"):
        dataset = MdsEdPreparedDataset(prepared_dir, split)
        options: dict[str, Any] = {
            "batch_size": batch_size,
            "shuffle": split == "train",
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "drop_last": False,
        }
        if num_workers:
            options.update(persistent_workers=True, prefetch_factor=2)
        loaders[split] = DataLoader(dataset, **options)
    return loaders
