"""Streaming audits for the official MDS-ED CSV and prepared ECG memmaps."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .constants import (
    DETERIORATION_LABEL_COUNT,
    DIAGNOSIS_LABEL_COUNT,
    ECG_CHANNEL_COUNT,
    ECG_SAMPLE_COUNT,
    FEATURE_PREFIXES,
    RAW_TABULAR_FEATURE_COUNT,
    REQUIRED_RELEASE_COLUMNS,
    STRATIFIED_FOLDS,
)


@dataclass(frozen=True, slots=True)
class ReleaseSchema:
    columns: tuple[str, ...]
    diagnosis_columns: tuple[str, ...]
    deterioration_columns: tuple[str, ...]
    raw_feature_columns: tuple[str, ...]
    mask_columns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ReleaseAudit:
    rows: int
    patients: int
    visits: int
    diagnosis_labels: int
    deterioration_labels: int
    zero_diagnosis_rows: int
    raw_tabular_features: int
    mask_features: int
    folds: tuple[int, ...]

    def to_dict(self) -> dict[str, int | list[int]]:
        payload = asdict(self)
        payload["folds"] = list(self.folds)
        return payload

    def __getitem__(self, key: str) -> int | list[int]:
        """Retain dict-style reads used by the original preparation script."""
        return self.to_dict()[key]


@dataclass(frozen=True, slots=True)
class PreparedMemmapAudit:
    rows: int
    waveform_files: int
    channels: int
    samples_per_record: int
    dtype: str

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)

    def __getitem__(self, key: str) -> int | str:
        return self.to_dict()[key]


def read_release_schema(
    csv_path: str | Path,
    *,
    expected_diagnosis_labels: int = DIAGNOSIS_LABEL_COUNT,
    expected_deterioration_labels: int = DETERIORATION_LABEL_COUNT,
    expected_raw_features: int = RAW_TABULAR_FEATURE_COUNT,
) -> ReleaseSchema:
    """Read and validate only the release header."""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"MDS-ED release CSV not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        columns = tuple(next(csv.reader(handle), ()))
    if not columns:
        raise ValueError(f"MDS-ED release CSV has no header: {path}")
    missing = sorted(set(REQUIRED_RELEASE_COLUMNS).difference(columns))
    if missing:
        raise ValueError(f"MDS-ED CSV is missing required columns: {missing}")

    diagnosis = tuple(column for column in columns if column.startswith("diagnoses_"))
    deterioration = tuple(column for column in columns if column.startswith("deterioration_"))
    features = tuple(
        column
        for column in columns
        if column.startswith(FEATURE_PREFIXES) and not column.endswith(("_nan", "_m"))
    )
    masks = tuple(
        column
        for column in columns
        if column.startswith(FEATURE_PREFIXES) and column.endswith(("_nan", "_m"))
    )
    expected = (
        ("diagnosis labels", len(diagnosis), expected_diagnosis_labels),
        ("deterioration labels", len(deterioration), expected_deterioration_labels),
        ("raw tabular features", len(features), expected_raw_features),
    )
    mismatches = [
        f"expected {wanted} {name}, found {actual}" for name, actual, wanted in expected if actual != wanted
    ]
    if mismatches:
        raise ValueError("; ".join(mismatches))
    return ReleaseSchema(columns, diagnosis, deterioration, features, masks)


def _parse_int(value: str, column: str, row_number: int) -> int:
    try:
        parsed_float = float(value)
        parsed = int(parsed_float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid integer in {column} at CSV row {row_number}: {value!r}") from exc
    if not np.isfinite(parsed_float) or parsed_float != parsed:
        raise ValueError(f"invalid integer in {column} at CSV row {row_number}: {value!r}")
    return parsed


def _binary_sum(values: Iterable[str], row_number: int) -> int:
    total = 0
    for value in values:
        if value not in {"0", "0.0", "1", "1.0"}:
            raise ValueError(f"diagnosis labels must be binary; CSV row {row_number} has {value!r}")
        total += value in {"1", "1.0"}
    return total


def audit_release_csv(
    csv_path: str | Path,
    *,
    expected_diagnosis_labels: int = DIAGNOSIS_LABEL_COUNT,
    expected_deterioration_labels: int = DETERIORATION_LABEL_COUNT,
    expected_raw_features: int = RAW_TABULAR_FEATURE_COUNT,
    expected_folds: Iterable[int] = STRATIFIED_FOLDS,
) -> ReleaseAudit:
    """Audit a release in one streaming pass without loading the 600 MB table."""
    path = Path(csv_path)
    schema = read_release_schema(
        path,
        expected_diagnosis_labels=expected_diagnosis_labels,
        expected_deterioration_labels=expected_deterioration_labels,
        expected_raw_features=expected_raw_features,
    )
    index = {column: position for position, column in enumerate(schema.columns)}
    diagnosis_indices = tuple(index[column] for column in schema.diagnosis_columns)
    deterioration_indices = tuple(index[column] for column in schema.deterioration_columns)
    subjects: set[int] = set()
    studies: set[int] = set()
    visits: set[int] = set()
    subjects_by_fold: dict[int, set[int]] = {}
    folds: set[int] = set()
    rows = zero_diagnosis_rows = 0
    visit_index = index.get("general_ed_stay_id")

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        next(reader)
        for row_number, row in enumerate(reader, start=2):
            if len(row) != len(schema.columns):
                raise ValueError(
                    f"CSV row {row_number} has {len(row)} fields; expected {len(schema.columns)}"
                )
            subject = _parse_int(row[index["general_subject_id"]], "general_subject_id", row_number)
            study = _parse_int(row[index["general_study_id"]], "general_study_id", row_number)
            fold = _parse_int(row[index["general_strat_fold"]], "general_strat_fold", row_number)
            ecg_number = _parse_int(
                row[index["general_ecg_no_within_stay"]],
                "general_ecg_no_within_stay",
                row_number,
            )
            if subject <= 0 or study <= 0:
                raise ValueError("MDS-ED subject/study identifiers must be positive")
            if ecg_number < 0:
                raise ValueError("general_ecg_no_within_stay must be non-negative")
            zero_diagnosis_rows += (
                _binary_sum((row[position] for position in diagnosis_indices), row_number) == 0
            )
            invalid_deterioration = [
                row[position]
                for position in deterioration_indices
                if row[position] not in {"-999", "-999.0", "0", "0.0", "1", "1.0"}
            ]
            if invalid_deterioration:
                raise ValueError(
                    "deterioration labels must be -999 (missing), 0, or 1; "
                    f"CSV row {row_number} has {invalid_deterioration[0]!r}"
                )
            if study in studies:
                raise ValueError(f"duplicate general_study_id in MDS-ED release: {study}")
            studies.add(study)
            subjects.add(subject)
            folds.add(fold)
            subjects_by_fold.setdefault(fold, set()).add(subject)
            if visit_index is not None and row[visit_index]:
                visits.add(_parse_int(row[visit_index], "general_ed_stay_id", row_number))
            rows += 1

    required_folds = set(int(fold) for fold in expected_folds)
    if folds != required_folds:
        raise ValueError(f"expected stratified folds {sorted(required_folds)}, found {sorted(folds)}")
    subject_fold: dict[int, int] = {}
    for fold, fold_subjects in subjects_by_fold.items():
        for subject in fold_subjects:
            previous = subject_fold.setdefault(subject, fold)
            if previous != fold:
                raise ValueError(
                    f"subject leakage detected: subject {subject} appears in folds {previous} and {fold}"
                )
    return ReleaseAudit(
        rows=rows,
        patients=len(subjects),
        visits=len(visits) if visit_index is not None else -1,
        diagnosis_labels=len(schema.diagnosis_columns),
        deterioration_labels=len(schema.deterioration_columns),
        zero_diagnosis_rows=zero_diagnosis_rows,
        raw_tabular_features=len(schema.raw_feature_columns),
        mask_features=len(schema.mask_columns),
        folds=tuple(sorted(folds)),
    )


def audit_prepared_memmap(
    output_dir: str | Path,
    expected_records: int,
    *,
    expected_samples: int = ECG_SAMPLE_COUNT,
    expected_channels: int = ECG_CHANNEL_COUNT,
) -> PreparedMemmapAudit:
    """Validate row mappings and metadata without reading waveform bytes."""
    output = Path(output_dir)
    required = (
        output / "mds_ed.csv",
        output / "df_memmap.pkl",
        output / "memmap_meta.npz",
        output / "mean.npy",
        output / "std.npy",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("prepared MDS-ED directory is incomplete; missing: " + ", ".join(missing))
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pandas is required to audit prepared MDS-ED metadata") from exc

    frame = pd.read_pickle(output / "df_memmap.pkl")
    with np.load(output / "memmap_meta.npz", allow_pickle=True) as metadata:
        required_keys = {"start", "length", "shape", "file_idx", "dtype", "filenames"}
        missing_keys = sorted(required_keys.difference(metadata.files))
        if missing_keys:
            raise ValueError(f"memmap metadata is missing arrays: {missing_keys}")
        starts = np.asarray(metadata["start"], dtype=np.int64)
        lengths = np.asarray(metadata["length"], dtype=np.int64)
        shapes = np.asarray(metadata["shape"], dtype=np.int64)
        file_indices = np.asarray(metadata["file_idx"], dtype=np.int64)
        filenames = tuple(str(value) for value in np.asarray(metadata["filenames"]).tolist())
        dtype = str(np.asarray(metadata["dtype"]).item())

    if len(frame) != expected_records:
        raise ValueError(f"prepared memmap has {len(frame)} rows, expected {expected_records}")
    if any(len(values) != expected_records for values in (starts, lengths, file_indices)):
        raise ValueError("memmap metadata is not row-aligned with df_memmap.pkl")
    if "data" not in frame:
        raise ValueError("df_memmap.pkl is missing the data index column")
    data_indices = frame["data"].to_numpy(dtype=np.int64)
    if len(np.unique(data_indices)) != expected_records or not np.array_equal(
        np.sort(data_indices), np.arange(expected_records)
    ):
        raise ValueError("df_memmap.pkl data indices must be a unique range [0, N)")
    if len(filenames) == 0:
        raise ValueError("memmap metadata has no waveform files")
    if shapes.ndim != 2 or shapes.shape != (len(filenames), 2):
        raise ValueError("memmap shapes must have one [timesteps, channels] row per file")
    if np.any(shapes[:, 1] != expected_channels):
        raise ValueError(f"expected {expected_channels} ECG channels, found {shapes.tolist()}")
    if np.any(lengths != expected_samples):
        raise ValueError(f"expected every ECG to contain {expected_samples} samples")
    if np.any(starts < 0) or np.any(file_indices < 0) or np.any(file_indices >= len(filenames)):
        raise ValueError("memmap metadata contains an out-of-range offset or file index")
    for record, (start, length, file_index) in enumerate(zip(starts, lengths, file_indices)):
        if start + length > shapes[file_index, 0]:
            raise ValueError(f"memmap record {record} extends beyond its waveform file")
    missing_waveforms = [str(output / name) for name in filenames if not (output / name).is_file()]
    if missing_waveforms:
        raise FileNotFoundError(
            "prepared memmap references missing waveform files; examples: " + ", ".join(missing_waveforms[:5])
        )
    try:
        itemsize = np.dtype(dtype).itemsize
    except TypeError as exc:
        raise ValueError(f"invalid memmap dtype metadata: {dtype!r}") from exc
    for file_index, (filename, shape) in enumerate(zip(filenames, shapes)):
        expected_bytes = int(np.prod(shape, dtype=np.int64)) * itemsize
        actual_bytes = (output / filename).stat().st_size
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"memmap file {file_index} has {actual_bytes} bytes; metadata declares {expected_bytes}"
            )
        intervals = sorted(
            (int(start), int(start + length))
            for start, length, mapped_file in zip(starts, lengths, file_indices)
            if mapped_file == file_index
        )
        if any(
            right_start < left_end
            for (_left_start, left_end), (right_start, _right_end) in zip(intervals, intervals[1:])
        ):
            raise ValueError(f"memmap file {file_index} contains overlapping record intervals")
    return PreparedMemmapAudit(
        rows=len(frame),
        waveform_files=len(filenames),
        channels=expected_channels,
        samples_per_record=expected_samples,
        dtype=dtype,
    )


# Compatibility alias for callers that used the upstream function name.
audit_mdsed_csv = audit_release_csv
