"""Resumable native preparation of MIMIC-IV-ECG records used by MDS-ED."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable, Iterable, Mapping

import numpy as np

from .constants import CHANNEL_TO_INDEX, ECG_CHANNEL_COUNT, ECG_SAMPLE_COUNT, ECG_SAMPLE_RATE


@dataclass(frozen=True, slots=True)
class EcgRecord:
    subject_id: int
    study_id: int
    record_path: Path


@dataclass(frozen=True, slots=True)
class PreparedEcgRecord:
    subject_id: int
    study_id: int
    waveform_path: Path
    samples: int
    channels: int
    source_fs: float | None = None
    nan_count: int = 0


RecordReader = Callable[[Path], tuple[np.ndarray, Mapping[str, object]]]


def normalize_ecg_root(data_path: str | Path) -> tuple[Path, Path]:
    root = Path(data_path).expanduser()
    if root.is_file():
        raise ValueError(f"MIMIC-IV-ECG must be a fully extracted directory, not a file: {root}")
    if not root.is_dir():
        raise FileNotFoundError(f"MIMIC-IV-ECG extracted directory not found: {root}")
    files_root = root / "files"
    return root, files_root if files_root.is_dir() else root


def _record_base(root: Path, files_root: Path, relative_value: str) -> Path | None:
    relative = Path(relative_value)
    candidates = [relative] if relative.is_absolute() else [root / relative, files_root / relative]
    for candidate in candidates:
        base = candidate.with_suffix("")
        if base.with_suffix(".hea").is_file():
            return base
    return None


def discover_ecg_records(
    data_path: str | Path,
    study_ids: Iterable[int] | None = None,
) -> list[EcgRecord]:
    """Find requested headers using record_list.csv, with a scan fallback."""
    root, files_root = normalize_ecg_root(data_path)
    requested = None if study_ids is None else {int(value) for value in study_ids}
    records: dict[int, EcgRecord] = {}
    index_path = root / "record_list.csv"
    if index_path.is_file():
        with index_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"study_id", "subject_id", "path"}
            missing = sorted(required.difference(reader.fieldnames or ()))
            if missing:
                raise ValueError(f"ECG record_list.csv is missing columns: {missing}")
            for row_number, row in enumerate(reader, start=2):
                try:
                    study_id = int(row["study_id"])
                    subject_id = int(row["subject_id"])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"invalid ECG index identifier at row {row_number}") from exc
                if requested is not None and study_id not in requested:
                    continue
                base = _record_base(root, files_root, row["path"])
                if base is not None:
                    records.setdefault(study_id, EcgRecord(subject_id, study_id, base))
        return [records[study] for study in sorted(records)]

    for header in files_root.rglob("*.hea"):
        try:
            study_id = int(header.stem)
        except ValueError:
            continue
        if requested is not None and study_id not in requested:
            continue
        subject_id = None
        for parent in header.parents:
            name = parent.name.lower()
            if name.startswith("p") and name[1:].isdigit() and len(name) > 2:
                subject_id = int(name[1:])
                break
        if subject_id is not None:
            records.setdefault(study_id, EcgRecord(subject_id, study_id, header.with_suffix("")))
    return [records[study] for study in sorted(records)]


def repair_ecg_signal(signal: np.ndarray, *, clip_amplitude: float = 3.0) -> tuple[np.ndarray, int]:
    """Interpolate interior NaNs, extend boundaries, then clip each lead."""
    repaired = np.asarray(signal, dtype=np.float32).copy()
    if repaired.ndim != 2 or repaired.shape[1] < 1:
        raise ValueError(f"ECG signal must have shape [samples, channels], got {repaired.shape}")
    nan_count = int(np.isnan(repaired).sum())
    positions = np.arange(repaired.shape[0])
    for channel in range(repaired.shape[1]):
        values = repaired[:, channel]
        valid = np.isfinite(values)
        if not valid.any():
            values.fill(0.0)
        elif not valid.all():
            values[~valid] = np.interp(positions[~valid], positions[valid], values[valid])
    if clip_amplitude > 0:
        np.clip(repaired, -clip_amplitude, clip_amplitude, out=repaired)
    return repaired, nan_count


def resample_ecg(
    signal: np.ndarray,
    channel_names: Iterable[str],
    source_fs: float,
    *,
    target_fs: float = ECG_SAMPLE_RATE,
    target_samples: int | None = ECG_SAMPLE_COUNT,
    channel_to_index: Mapping[str, int] | None = CHANNEL_TO_INDEX,
    channels: int = ECG_CHANNEL_COUNT,
) -> np.ndarray:
    """Anti-aliased vectorized resampling followed by deterministic lead mapping."""
    if source_fs <= 0 or target_fs <= 0:
        raise ValueError(f"sampling rates must be positive, got {source_fs} and {target_fs}")
    try:
        from scipy.signal import resample_poly
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("scipy is required to resample MDS-ED ECG waveforms") from exc
    values = np.asarray(signal, dtype=np.float32)
    names = tuple(str(name).lower() for name in channel_names)
    if values.ndim != 2 or values.shape[1] != len(names):
        raise ValueError("ECG signal channel dimension does not match the channel labels")
    ratio = Fraction(str(float(target_fs))) / Fraction(str(float(source_fs)))
    ratio = ratio.limit_denominator(10_000)
    resampled = resample_poly(values, ratio.numerator, ratio.denominator, axis=0).astype(
        np.float32, copy=False
    )
    expected_length = int(values.shape[0] * float(target_fs) / float(source_fs))
    if target_samples is not None:
        expected_length = int(target_samples)
    if len(resampled) < expected_length:
        resampled = np.pad(resampled, ((0, expected_length - len(resampled)), (0, 0)))
    resampled = resampled[:expected_length]
    if channel_to_index is None:
        return resampled
    output = np.zeros((expected_length, channels), dtype=np.float32)
    for source_index, channel_name in enumerate(names):
        output_index = channel_to_index.get(channel_name)
        if output_index is not None and 0 <= output_index < channels:
            output[:, output_index] = resampled[:, source_index]
    return output


def _default_record_reader(record_path: Path) -> tuple[np.ndarray, Mapping[str, object]]:
    try:
        import wfdb
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "wfdb is required to prepare MDS-ED ECG waveforms; install wfdb or pass record_reader"
        ) from exc
    signal, fields = wfdb.rdsamp(str(record_path))
    return np.asarray(signal), fields


def _prepared_shape(path: Path, samples: int, channels: int | None) -> tuple[int, int] | None:
    if not path.is_file():
        return None
    try:
        value = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError):
        return None
    if (
        value.ndim != 2
        or value.shape[0] != samples
        or (channels is not None and value.shape[1] != channels)
        or value.dtype != np.float32
    ):
        return None
    return int(value.shape[0]), int(value.shape[1])


def _atomic_save(path: Path, value: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npy")
    np.save(temporary, value, allow_pickle=False)
    temporary.replace(path)


def prepare_ecg_records(
    records: Iterable[EcgRecord],
    output_dir: str | Path,
    *,
    record_reader: RecordReader | None = None,
    target_fs: float = ECG_SAMPLE_RATE,
    target_samples: int = ECG_SAMPLE_COUNT,
    channels: int = ECG_CHANNEL_COUNT,
    channel_to_index: Mapping[str, int] | None = CHANNEL_TO_INDEX,
    clip_amplitude: float = 3.0,
    resume: bool = True,
) -> list[PreparedEcgRecord]:
    """Prepare records one at a time and atomically checkpoint each waveform."""
    output = Path(output_dir)
    waveform_dir = output / "waveforms"
    waveform_dir.mkdir(parents=True, exist_ok=True)
    reader = record_reader or _default_record_reader
    prepared: list[PreparedEcgRecord] = []
    manifest_path = output / "ecg_prepare_manifest.jsonl"
    for record in records:
        waveform_path = waveform_dir / f"p{record.subject_id}_s{record.study_id}.npy"
        expected_channels = channels if channel_to_index is not None else None
        existing_shape = _prepared_shape(waveform_path, target_samples, expected_channels)
        if resume and existing_shape is not None:
            prepared.append(
                PreparedEcgRecord(
                    record.subject_id,
                    record.study_id,
                    waveform_path,
                    existing_shape[0],
                    existing_shape[1],
                )
            )
            continue
        signal, fields = reader(record.record_path)
        source_fs = float(fields.get("fs", 0.0))
        channel_names = fields.get("sig_name")
        if channel_names is None:
            raise ValueError(f"WFDB record has no sig_name metadata: {record.record_path}")
        repaired, nan_count = repair_ecg_signal(signal, clip_amplitude=clip_amplitude)
        transformed = resample_ecg(
            repaired,
            channel_names,
            source_fs,
            target_fs=target_fs,
            target_samples=target_samples,
            channel_to_index=channel_to_index,
            channels=channels,
        )
        _atomic_save(waveform_path, transformed)
        item = PreparedEcgRecord(
            record.subject_id,
            record.study_id,
            waveform_path,
            int(transformed.shape[0]),
            int(transformed.shape[1]),
            source_fs,
            nan_count,
        )
        prepared.append(item)
        with manifest_path.open("a", encoding="utf-8") as handle:
            payload = asdict(item)
            payload["waveform_path"] = str(waveform_path)
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    return prepared


def prepare_mimicecg(
    data_path: str | Path = "",
    clip_amp: float = 3.0,
    target_fs: float = ECG_SAMPLE_RATE,
    channels: int = ECG_CHANNEL_COUNT,
    strat_folds: int = 20,
    channel_stoi: Mapping[str, int] | None = CHANNEL_TO_INDEX,
    target_folder: str | Path | None = None,
    recreate_data: bool = True,
    study_ids: Iterable[int] | None = None,
    resume: bool = False,
):
    """Native compatibility entry point for upstream ECG preparation callers."""
    del strat_folds  # The release CSV, not waveform preparation, owns the official folds.
    if target_folder is None:
        raise ValueError("target_folder is required for MDS-ED ECG preparation")
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pandas is required to prepare MDS-ED ECG metadata") from exc

    output = Path(target_folder)
    if not recreate_data:
        frame = pd.read_pickle(output / "df.pkl")
        return (
            frame,
            np.load(output / "lbl_itos.npy", allow_pickle=False),
            np.load(output / "mean.npy", allow_pickle=False),
            np.load(output / "std.npy", allow_pickle=False),
        )

    records = discover_ecg_records(data_path, study_ids)
    prepared = prepare_ecg_records(
        records,
        output,
        target_fs=target_fs,
        target_samples=int(round(float(target_fs) * 10)),
        channels=channels,
        channel_to_index=channel_stoi,
        clip_amplitude=clip_amp,
        resume=resume,
    )
    rows: list[dict[str, object]] = []
    for item in prepared:
        waveform = np.load(item.waveform_path, mmap_mode="r", allow_pickle=False)
        rows.append(
            {
                "data": str(item.waveform_path.relative_to(output)),
                "study_id": item.study_id,
                "subject_id": item.subject_id,
                "nans": item.nan_count,
                "data_mean": np.mean(waveform, axis=0),
                "data_std": np.std(waveform, axis=0),
                "data_length": len(waveform),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("no requested ECG records were found in the extracted release")
    mean = np.stack(frame["data_mean"]).mean(axis=0).astype(np.float32)
    std = np.stack(frame["data_std"]).mean(axis=0).astype(np.float32)
    labels = np.asarray([], dtype="U1")
    frame.to_pickle(output / "df.pkl")
    np.save(output / "lbl_itos.npy", labels, allow_pickle=False)
    np.save(output / "mean.npy", mean, allow_pickle=False)
    np.save(output / "std.npy", std, allow_pickle=False)
    return frame, labels, mean, std


def resample_data(
    sigbufs: np.ndarray,
    channel_labels: Iterable[str],
    fs: float,
    target_fs: float,
    channels: int = ECG_CHANNEL_COUNT,
    channel_stoi: Mapping[str, int] | None = None,
) -> np.ndarray:
    """Compatibility wrapper around :func:`resample_ecg`."""
    return resample_ecg(
        sigbufs,
        channel_labels,
        fs,
        target_fs=target_fs,
        target_samples=None,
        channel_to_index=channel_stoi,
        channels=channels,
    )


# Compatibility alias for the upstream private helper.
_directory_records = discover_ecg_records
