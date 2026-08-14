"""Strict unified tensor loader for native multi-label MIMIC-III tasks."""

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
from .scenarios import ScenarioSpec, empty_note, scenario_from_config
from .tasks import task_manifest, task_spec


SPLIT_NAMES = ("train", "val", "test")
CAREUNIT_ARRAY = "careunit_code"
NOTES_AVAILABLE_ARRAY = "notes_available"
REQUIRED_ARRAYS = (
    "X",
    "s",
    "input_ids",
    "token_type_ids",
    "attention_mask",
    "label",
)
SUBJECT_ID_ARRAYS = ("subject_id", "SUBJECT_ID")
OPTIONAL_ARRAYS = ("time_series_mask", "stay_id")


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
    label_names: tuple[str, ...] = ()


def _import_h5py():
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - dependency error is environment-specific
        raise RuntimeError("h5py is required to read the MIMIC split artifact") from exc
    return h5py


def _subject_id_array(group: Any) -> str:
    for name in SUBJECT_ID_ARRAYS:
        if name in group:
            return name
    raise ValueError(
        "MIMIC split artifact must contain a row-aligned subject_id/SUBJECT_ID array "
        "for grouped active-learning validation"
    )


def audit_split_hdf5(config: dict[str, Any]) -> YangWuDataAudit:
    """Validate every source-aligned tensor before a formal experiment starts."""
    paths = require_paths(config)
    spec = task_spec(config)
    group_name = str(config.get("dataset", {}).get("split_group", "with_notes"))
    cohort_mode = str(config.get("dataset", {}).get("cohort_mode", "official")).lower()
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
            shapes.update(
                {
                    name: tuple(group[name].shape)
                    for name in OPTIONAL_ARRAYS
                    if name in group
                }
            )
            shapes[subject_id_name] = tuple(group[subject_id_name].shape)
            if any(shape[0] != count for shape in shapes.values()):
                raise ValueError(f"unaligned rows in {group_name}/{split}: {shapes}")
            if len(shapes["X"]) != 3 or len(shapes["s"]) != 2:
                raise ValueError("X and s tensors must have shapes [N,T,D] and [N,D]")
            if len(shapes[subject_id_name]) != 1:
                raise ValueError("subject_id/SUBJECT_ID must have shape [N]")
            if any(len(shapes[name]) != 2 for name in ("input_ids", "token_type_ids", "attention_mask")):
                raise ValueError("BERT input tensors must have shape [N,L]")
            if "time_series_mask" in shapes and shapes["time_series_mask"] != shapes["X"][:2]:
                raise ValueError("time_series_mask must have shape [N,T] matching X")
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
                raise ValueError(f"feature dimensions differ across MIMIC task splits: {current}")
            split_counts[split] = count
            subject_ids = np.asarray(group[subject_id_name], dtype=np.int64)
            if np.any(subject_ids <= 0):
                raise ValueError("subject_id/SUBJECT_ID values must be positive MIMIC subject identifiers")
            batch = 512
            for start in range(0, count, batch):
                labels = np.asarray(group["label"][start : start + batch])
                if not np.logical_or(labels == 0, labels == 1).all():
                    raise ValueError("task labels must be a binary multi-hot matrix")
                if bool(expected.get("require_positive_label_per_sample", True)) and np.any(
                    labels.sum(axis=1) == 0
                ):
                    raise ValueError("this task requires at least one positive label per ICU stay")
                positive_labels += int(labels.sum())

    total = sum(split_counts.values())
    requirements = {
        "labels": (label_count, int(expected.get("expected_label_count", spec.label_count))),
        "time steps": (
            time_steps,
            int(expected.get("max_time_steps", expected.get("observation_hours", 48))),
        ),
        "time-series dimension": (
            time_series_dim,
            int(expected.get("time_series_dim", 7749)),
        ),
        "time-invariant dimension": (
            time_invariant_dim,
            int(expected.get("time_invariant_dim", 97)),
        ),
        "note tokens": (note_tokens, int(expected.get("max_note_tokens", 512))),
    }
    expected_total = expected.get("expected_total_samples", 10258)
    if expected_total is not None:
        requirements["total samples"] = (total, int(expected_total))
    mismatches = [f"{name}: expected {required}, got {actual}" for name, (actual, required) in requirements.items() if actual != required]
    if mismatches:
        raise ValueError("formal MIMIC task tensor mismatch: " + "; ".join(mismatches))
    if not all(split_counts.values()):
        raise ValueError(f"MIMIC train/val/test splits must all be non-empty: {split_counts}")
    if cohort_mode == "full_cohort" and total <= 10258:
        raise ValueError(
            "full_cohort requires a source artifact larger than the 10,258-row local rebuild; "
            f"got {total} rows"
        )
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
        raise ValueError(f"subject leakage across MIMIC task splits: {overlap}")
    label_names: tuple[str, ...] = ()
    with h5py.File(paths["split_hdf5"], "r") as handle:
        root = handle[group_name]
        artifact_task = root.attrs.get("task_id")
        if isinstance(artifact_task, bytes):
            artifact_task = artifact_task.decode("utf-8")
        if artifact_task is not None and str(artifact_task) != spec.task_id:
            raise ValueError(
                f"artifact task_id={artifact_task!r} does not match config task.id={spec.task_id!r}"
            )
        if "label_names" in root:
            label_names = tuple(
                value.decode("utf-8") if isinstance(value, bytes) else str(value)
                for value in np.asarray(root["label_names"])
            )
            if len(label_names) != label_count:
                raise ValueError(
                    f"label_names contains {len(label_names)} entries for {label_count} labels"
                )
    return YangWuDataAudit(
        split_counts,
        total,
        label_count,
        time_steps,
        time_series_dim,
        time_invariant_dim,
        note_tokens,
        positive_labels,
        label_names,
    )


class YangWuFeatureStore:
    """Global row view over three task HDF5 split groups."""

    def __init__(self, config: dict[str, Any], *, validate: bool = True) -> None:
        self.config = config
        paths = require_paths(config)
        self.path = paths["split_hdf5"]
        self.group_name = str(config.get("dataset", {}).get("split_group", "with_notes"))
        audit = audit_split_hdf5(config) if validate else self._shape_audit()
        self.audit = audit
        self.split_counts = audit.split_counts
        self.offsets = np.cumsum([0, *(self.split_counts[name] for name in SPLIT_NAMES)])
        self.subject_ids = self._load_subject_ids()
        self.label_names = audit.label_names or tuple(
            f"{task_spec(config).task_id}:{index}" for index in range(audit.label_count)
        )
        self.splits = np.concatenate(
            [np.full(self.split_counts[name], name, dtype=object) for name in SPLIT_NAMES]
        )
        raw_labels = self._load_labels()
        self.scenario = scenario_from_config(
            config,
            labels=raw_labels,
            split_names=self.splits,
            careunit_codes=self._optional_array(CAREUNIT_ARRAY, np.int64),
            notes_available=self._optional_array(NOTES_AVAILABLE_ARRAY, bool),
        )
        self.labels = self.scenario.select_labels(raw_labels)
        self.label_count = int(self.labels.shape[1])

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
                (),
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

    def _optional_array(self, name: str, dtype: Any) -> np.ndarray | None:
        """Row array the rebuilt cohorts carry and the original 48h artifact does not."""
        h5py = _import_h5py()
        with h5py.File(self.path, "r") as handle:
            root = handle[self.group_name]
            if any(name not in root[split] for split in SPLIT_NAMES):
                return None
            return np.concatenate(
                [np.asarray(root[split][name], dtype=dtype) for split in SPLIT_NAMES]
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
            self.scenario,
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
        scenario: ScenarioSpec | None = None,
    ) -> None:
        self.path = path
        self.group_name = group_name
        self.indices = indices
        self.offsets = offsets
        self.scenario = scenario
        self._handle = None
        self._empty_note: dict[str, np.ndarray] | None = None

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
        labels = np.asarray(group["label"][local_index], dtype=np.float32)
        note_missing = self.scenario is not None and self.scenario.note_is_missing(global_index)
        if note_missing:
            note = self._withheld_note(int(group["input_ids"].shape[1]))
        else:
            note = {
                name: np.asarray(group[name][local_index], dtype=np.int64)
                for name in ("input_ids", "token_type_ids", "attention_mask")
            }
        if self.scenario is not None:
            labels = self.scenario.select_labels(labels)
        result = {
            "time_series": torch.from_numpy(np.asarray(group["X"][local_index], dtype=np.float32)),
            "time_invariant": torch.from_numpy(np.asarray(group["s"][local_index], dtype=np.float32)),
            "input_ids": torch.from_numpy(note["input_ids"]),
            "token_type_ids": torch.from_numpy(note["token_type_ids"]),
            "attention_mask": torch.from_numpy(note["attention_mask"]),
            "labels": torch.from_numpy(labels),
            "notes_available": torch.tensor(not note_missing, dtype=torch.bool),
            "subject_id": torch.tensor(
                int(np.asarray(group[_subject_id_array(group)][local_index])), dtype=torch.long
            ),
            "index": torch.tensor(global_index, dtype=torch.long),
        }
        if "time_series_mask" in group:
            result["time_series_mask"] = torch.from_numpy(
                np.asarray(group["time_series_mask"][local_index], dtype=np.bool_)
            )
        if "stay_id" in group:
            result["stay_id"] = torch.tensor(
                int(np.asarray(group["stay_id"][local_index])), dtype=torch.long
            )
        return result

    def _withheld_note(self, length: int) -> dict[str, np.ndarray]:
        if self._empty_note is None:
            self._empty_note = empty_note(length)
        # Copied because torch.from_numpy aliases, and the batch is mutated downstream.
        return {name: value.copy() for name, value in self._empty_note.items()}


def prepare_official_artifacts(
    config: dict[str, Any], output_dir: str | Path | None = None
) -> dict[str, Any]:
    """Audit task tensors and write a provenance manifest, never synthetic data."""
    audit = audit_split_hdf5(config)
    dataset = config.get("dataset", {})
    output = Path(output_dir or dataset.get("prepared_dir", "prepared/yang_wu_diagnoses_48h"))
    output.mkdir(parents=True, exist_ok=True)
    paths = require_paths(config)
    cohort_mode = str(dataset.get("cohort_mode", "official")).lower()
    spec = task_spec(config)
    payload = {
        "format_version": 2,
        "task": task_manifest(config),
        "protocol": spec.display_name,
        "cohort_mode": cohort_mode,
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
        "label_names": list(audit.label_names),
    }
    (output / "manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload | {"output_dir": str(output)}


# Backward-compatible names retained for scripts built around the original task.
MimicDataAudit = YangWuDataAudit
MimicFeatureStore = YangWuFeatureStore
