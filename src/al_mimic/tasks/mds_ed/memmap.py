"""Native conversion of prepared ECG arrays to MDS-ED memmap artifacts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import numpy as np

from .ecg import PreparedEcgRecord


def build_prepared_memmap(
    records: Iterable[PreparedEcgRecord],
    output_dir: str | Path,
    *,
    delete_waveforms: bool = False,
) -> Path:
    """Build a row-aligned raw memmap using constant memory."""
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pandas is required to write MDS-ED memmap metadata") from exc
    items = list(records)
    if not items:
        raise ValueError("cannot build an MDS-ED memmap without ECG records")
    samples = items[0].samples
    channels = items[0].channels
    if any((item.samples, item.channels) != (samples, channels) for item in items):
        raise ValueError("all prepared ECG records must have the same shape")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    final_path = output / "memmap.npy"
    temporary = output / f".memmap.{os.getpid()}.tmp"
    total_samples = len(items) * samples
    target = np.memmap(temporary, dtype=np.float32, mode="w+", shape=(total_samples, channels))
    starts = np.arange(len(items), dtype=np.int64) * samples
    value_sum = np.zeros(channels, dtype=np.float64)
    value_square_sum = np.zeros(channels, dtype=np.float64)
    for index, item in enumerate(items):
        waveform = np.load(item.waveform_path, mmap_mode="r", allow_pickle=False)
        if waveform.shape != (samples, channels):
            raise ValueError(
                f"prepared ECG {item.waveform_path} has shape {waveform.shape}, "
                f"expected {(samples, channels)}"
            )
        if waveform.dtype != np.float32 or not np.isfinite(waveform).all():
            raise ValueError(f"prepared ECG must contain finite float32 values: {item.waveform_path}")
        start = int(starts[index])
        target[start : start + samples] = waveform
        value_sum += np.sum(waveform, axis=0, dtype=np.float64)
        value_square_sum += np.sum(np.square(waveform, dtype=np.float64), axis=0)
    target.flush()
    del target
    temporary.replace(final_path)

    count = total_samples
    mean = value_sum / count
    variance = np.maximum(value_square_sum / count - np.square(mean), 0.0)
    np.save(output / "mean.npy", mean.astype(np.float32), allow_pickle=False)
    np.save(output / "std.npy", np.sqrt(variance).astype(np.float32), allow_pickle=False)
    np.savez(
        output / "memmap_meta.npz",
        start=starts,
        length=np.full(len(items), samples, dtype=np.int64),
        shape=np.asarray([[total_samples, channels]], dtype=np.int64),
        file_idx=np.zeros(len(items), dtype=np.int64),
        dtype=np.asarray("float32"),
        filenames=np.asarray([final_path.name]),
    )
    frame = pd.DataFrame(
        {
            "data": np.arange(len(items), dtype=np.int64),
            "data_original": [item.waveform_path.name for item in items],
            "study_id": np.asarray([item.study_id for item in items], dtype=np.int64),
            "subject_id": np.asarray([item.subject_id for item in items], dtype=np.int64),
            "data_length": np.full(len(items), samples, dtype=np.int64),
        }
    )
    frame.to_pickle(output / "df_memmap.pkl")
    np.save(output / "lbl_itos.npy", np.asarray([], dtype="U1"), allow_pickle=False)
    if delete_waveforms:
        for item in items:
            item.waveform_path.unlink(missing_ok=True)
    return final_path
