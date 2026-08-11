"""Strict loader for the official Yang-Wu MIMIC-III diagnosis tensors."""

from __future__ import annotations

import bisect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .config import require_paths


SPLIT_NAMES = ("train", "val", "test")
REQUIRED_ARRAYS = (
    "X",
    "s",
    "input_ids",
    "token_type_ids",
    "attention_mask",
    "label",
)
SUBJECT_ID_ARRAYS = ("subject_id", "SUBJECT_ID")


@dataclass(frozen=True, slots=True)
class YangWuDataAudit:
    split_counts: dict[str, int]
    total_samples: int
    label_count: int
    time_steps: int
    time_series_dim: int
    time_invariant_dim: int
    note_tokens: int
    positive_labels: int


def _import_h5py():
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - dependency error is environment-specific
        raise RuntimeError("h5py is required to read the official Yang-Wu split artifact") from exc
    return h5py


def _subject_id_array(group: Any) -> str:
    for name in SUBJECT_ID_ARRAYS:
        if name in group:
            return name
    raise ValueError(
        "official split artifact must contain a row-aligned subject_id/SUBJECT_ID array "
        "for grouped active-learning validation"
    )


def audit_split_hdf5(config: dict[str, Any]) -> YangWuDataAudit:
    """Validate every source-aligned tensor before a formal experiment starts."""
    paths = require_paths(config)
    group_name = str(config.get("dataset", {}).get("split_group", "with_notes"))
    expected = config.get("preprocessing", {})
    split_counts: dict[str, int] = {}
    label_count = time_steps = time_series_dim = time_invariant_dim = note_tokens = -1
    positive_labels = 0
    h5py = _import_h5py()
    with h5py.File(paths["split_hdf5"], "r") as handle:
        if group_name not in handle:
            raise ValueError(f"split artifact has no group {group_name!r}")
        root = handle[group_name]
        for split in SPLIT_NAMES:
            if split not in root:
                raise ValueError(f"split artifact has no {group_name}/{split} group")
            group = root[split]
            missing = [name for name in REQUIRED_ARRAYS if name not in group]
            if missing:
                raise ValueError(f"{group_name}/{split} is missing arrays: {missing}")
            subject_id_name = _subject_id_array(group)
            count = int(group["label"].shape[0])
            shapes = {name: tuple(group[name].shape) for name in REQUIRED_ARRAYS}
            shapes[subject_id_name] = tuple(group[subject_id_name].shape)
            if any(shape[0] != count for shape in shapes.values()):
                raise ValueError(f"unaligned rows in {group_name}/{split}: {shapes}")
            if len(shapes["X"]) != 3 or len(shapes["s"]) != 2:
                raise ValueError("official X and s tensors must have shapes [N,T,D] and [N,D]")
            if len(shapes[subject_id_name]) != 1:
                raise ValueError("subject_id/SUBJECT_ID must have shape [N]")
            if any(len(shapes[name]) != 2 for name in ("input_ids", "token_type_ids", "attention_mask")):
                raise ValueError("official BERT input tensors must have shape [N,512]")
            current = (
                shapes["label"][1],
                shapes["X"][1],
                shapes["X"][2],
                shapes["s"][1],
                shapes["input_ids"][1],
            )
            if label_count < 0:
                label_count, time_steps, time_series_dim, time_invariant_dim, note_tokens = current
            elif current != (label_count, time_steps, time_series_dim, time_invariant_dim, note_tokens):
                raise ValueError(f"feature dimensions differ across official splits: {current}")
            split_counts[split] = count
            subject_ids = np.asarray(group[subject_id_name], dtype=np.int64)
            if np.any(subject_ids <= 0):
                raise ValueError("subject_id/SUBJECT_ID values must be positive MIMIC subject identifiers")
            batch = 512
            for start in range(0, count, batch):
                labels = np.asarray(group["label"][start : start + batch])
                if not np.logical_or(labels == 0, labels == 1).all():
                    raise ValueError("diagnosis labels must be a binary multi-hot matrix")
                if np.any(labels.sum(axis=1) == 0):
                    raise ValueError("every diagnosis visit must contain at least one ICD-9 group")
                positive_labels += int(labels.sum())

    total = sum(split_counts.values())
    requirements = {
        "total samples": (total, int(expected.get("expected_total_samples", 10210))),
        "labels": (label_count, int(expected.get("expected_label_count", 1042))),
        "time steps": (time_steps, int(expected.get("observation_hours", 48))),
        "time-series dimension": (
            time_series_dim,
            int(expected.get("time_series_dim", 7411)),
        ),
        "time-invariant dimension": (
            time_invariant_dim,
            int(expected.get("time_invariant_dim", 97)),
        ),
        "note tokens": (note_tokens, int(expected.get("max_note_tokens", 512))),
    }
    mismatches = [f"{name}: expected {required}, got {actual}" for name, (actual, required) in requirements.items() if actual != required]
    if mismatches:
        raise ValueError("formal Yang-Wu tensor mismatch: " + "; ".join(mismatches))
    if not all(split_counts.values()):
        raise ValueError(f"official train/val/test splits must all be non-empty: {split_counts}")
    with h5py.File(paths["split_hdf5"], "r") as handle:
        root = handle[group_name]
        split_subjects = {
            split: set(
                np.asarray(root[split][_subject_id_array(root[split])], dtype=np.int64).tolist()
            )
            for split in SPLIT_NAMES
        }
    overlap = {
        f"{left}/{right}": sorted(split_subjects[left] & split_subjects[right])[:5]
        for position, left in enumerate(SPLIT_NAMES)
        for right in SPLIT_NAMES[position + 1 :]
        if split_subjects[left] & split_subjects[right]
    }
    if overlap:
        raise ValueError(f"subject leakage across official splits: {overlap}")
    return YangWuDataAudit(
        split_counts,
        total,
        label_count,
        time_steps,
        time_series_dim,
        time_invariant_dim,
        note_tokens,
        positive_labels,
    )


class YangWuFeatureStore:
    """Global row view over the three official HDF5 split groups."""

    def __init__(self, config: dict[str, Any], *, validate: bool = True) -> None:
        self.config = config
        paths = require_paths(config)
        self.path = paths["split_hdf5"]
        self.group_name = str(config.get("dataset", {}).get("split_group", "with_notes"))
        audit = audit_split_hdf5(config) if validate else self._shape_audit()
        self.audit = audit
        self.split_counts = audit.split_counts
        self.offsets = np.cumsum([0, *(self.split_counts[name] for name in SPLIT_NAMES)])
        self.labels = self._load_labels()
        self.subject_ids = self._load_subject_ids()
        self.splits = np.concatenate(
            [np.full(self.split_counts[name], name, dtype=object) for name in SPLIT_NAMES]
        )

    def _shape_audit(self) -> YangWuDataAudit:
        h5py = _import_h5py()
        with h5py.File(self.path, "r") as handle:
            root = handle[self.group_name]
            counts = {name: int(root[name]["label"].shape[0]) for name in SPLIT_NAMES}
            group = root["train"]
            return YangWuDataAudit(
                counts,
                sum(counts.values()),
                int(group["label"].shape[1]),
                int(group["X"].shape[1]),
                int(group["X"].shape[2]),
                int(group["s"].shape[1]),
                int(group["input_ids"].shape[1]),
                0,
            )

    def _load_labels(self) -> np.ndarray:
        h5py = _import_h5py()
        with h5py.File(self.path, "r") as handle:
            root = handle[self.group_name]
            return np.concatenate(
                [np.asarray(root[name]["label"], dtype=np.float32) for name in SPLIT_NAMES]
            )

    def _load_subject_ids(self) -> np.ndarray:
        h5py = _import_h5py()
        with h5py.File(self.path, "r") as handle:
            root = handle[self.group_name]
            return np.concatenate(
                [
                    np.asarray(
                        root[name][_subject_id_array(root[name])], dtype=np.int64
                    )
                    for name in SPLIT_NAMES
                ]
            )

    def indices(self, split: str) -> np.ndarray:
        normalized = "val" if split == "validation" else split
        if normalized not in SPLIT_NAMES:
            raise ValueError(f"unknown split: {split}")
        position = SPLIT_NAMES.index(normalized)
        return np.arange(self.offsets[position], self.offsets[position + 1], dtype=np.int64)

    def locate(self, global_index: int) -> tuple[str, int]:
        if not 0 <= global_index < int(self.offsets[-1]):
            raise IndexError(global_index)
        split_position = bisect.bisect_right(self.offsets, global_index) - 1
        return SPLIT_NAMES[split_position], global_index - int(self.offsets[split_position])

    def make_loader(
        self,
        indices: Iterable[int],
        *,
        batch_size: int,
        shuffle: bool,
        num_workers: int,
        pin_memory: bool,
    ) -> DataLoader[dict[str, torch.Tensor]]:
        dataset = YangWuDataset(
            self.path,
            self.group_name,
            np.asarray(list(indices), dtype=np.int64),
            self.offsets,
        )
        options: dict[str, Any] = {
            "batch_size": int(batch_size),
            "shuffle": bool(shuffle),
            "num_workers": int(num_workers),
            "pin_memory": bool(pin_memory),
            "drop_last": False,
        }
        if num_workers > 0:
            options["persistent_workers"] = True
            options["prefetch_factor"] = 2
        return DataLoader(dataset, **options)


class YangWuDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        path: Path,
        group_name: str,
        indices: np.ndarray,
        offsets: np.ndarray,
    ) -> None:
        self.path = path
        self.group_name = group_name
        self.indices = indices
        self.offsets = offsets
        self._handle = None

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_handle"] = None
        return state

    def _root(self):
        if self._handle is None:
            self._handle = _import_h5py().File(self.path, "r")
        return self._handle[self.group_name]

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        global_index = int(self.indices[item])
        split_position = bisect.bisect_right(self.offsets, global_index) - 1
        split = SPLIT_NAMES[split_position]
        local_index = global_index - int(self.offsets[split_position])
        group = self._root()[split]
        return {
            "time_series": torch.from_numpy(np.asarray(group["X"][local_index], dtype=np.float32)),
            "time_invariant": torch.from_numpy(np.asarray(group["s"][local_index], dtype=np.float32)),
            "input_ids": torch.from_numpy(np.asarray(group["input_ids"][local_index], dtype=np.int64)),
            "token_type_ids": torch.from_numpy(
                np.asarray(group["token_type_ids"][local_index], dtype=np.int64)
            ),
            "attention_mask": torch.from_numpy(
                np.asarray(group["attention_mask"][local_index], dtype=np.int64)
            ),
            "labels": torch.from_numpy(np.asarray(group["label"][local_index], dtype=np.float32)),
            "subject_id": torch.tensor(
                int(np.asarray(group[_subject_id_array(group)][local_index])), dtype=torch.long
            ),
            "index": torch.tensor(global_index, dtype=torch.long),
        }


def prepare_official_artifacts(
    config: dict[str, Any], output_dir: str | Path | None = None
) -> dict[str, Any]:
    """Audit official tensors and write a provenance manifest, never synthetic data."""
    audit = audit_split_hdf5(config)
    dataset = config.get("dataset", {})
    output = Path(output_dir or dataset.get("prepared_dir", "prepared/yang_wu_diagnoses_48h"))
    output.mkdir(parents=True, exist_ok=True)
    paths = require_paths(config)
    payload = {
        "format_version": 1,
        "protocol": "Yang and Wu (EMNLP 2021) MIMIC-III Diagnoses 48h",
        "source_repository": "https://github.com/emnlp-mimic/mimic",
        "split_hdf5": str(paths["split_hdf5"].resolve()),
        "clinicalbert_checkpoint": str(paths["clinicalbert_checkpoint"].resolve()),
        "split_group": str(dataset.get("split_group", "with_notes")),
        "split_counts": audit.split_counts,
        "total_samples": audit.total_samples,
        "label_count": audit.label_count,
        "time_steps": audit.time_steps,
        "time_series_dim": audit.time_series_dim,
        "time_invariant_dim": audit.time_invariant_dim,
        "note_tokens": audit.note_tokens,
        "positive_labels": audit.positive_labels,
    }
    (output / "manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload | {"output_dir": str(output)}
