from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from al_mimic.tasks.mds_ed.audit import audit_prepared_memmap
from al_mimic.tasks.mds_ed.ecg import (
    EcgRecord,
    discover_ecg_records,
    prepare_ecg_records,
    repair_ecg_signal,
    resample_data,
    resample_ecg,
)
from al_mimic.tasks.mds_ed.memmap import build_prepared_memmap


def test_record_discovery_prefers_release_index(tmp_path: Path) -> None:
    record_base = tmp_path / "files" / "p1000" / "p10000001" / "s200" / "200"
    record_base.parent.mkdir(parents=True)
    record_base.with_suffix(".hea").write_text("synthetic", encoding="utf-8")
    with (tmp_path / "record_list.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("subject_id", "study_id", "path"))
        writer.writeheader()
        writer.writerow(
            {
                "subject_id": 10000001,
                "study_id": 200,
                "path": str(record_base.relative_to(tmp_path)),
            }
        )

    records = discover_ecg_records(tmp_path, {200, 999})

    assert records == [EcgRecord(10000001, 200, record_base)]


def test_repair_resample_and_resumable_preparation(tmp_path: Path) -> None:
    signal = np.column_stack(
        (
            np.linspace(-5.0, 5.0, 20, dtype=np.float32),
            np.linspace(1.0, 2.0, 20, dtype=np.float32),
        )
    )
    signal[0, 0] = np.nan
    signal[5, 1] = np.nan
    repaired, nan_count = repair_ecg_signal(signal)
    assert nan_count == 2
    assert np.isfinite(repaired).all()
    assert np.max(np.abs(repaired)) <= 3.0

    resampled = resample_ecg(
        repaired,
        ("II", "I"),
        200,
        target_fs=100,
        target_samples=10,
    )
    assert resampled.shape == (10, 12)
    assert np.any(resampled[:, 0])
    assert np.any(resampled[:, 1])
    assert np.count_nonzero(resampled[:, 2:]) == 0
    assert resample_data(repaired, ("II", "I"), 200, 100).shape == (10, 2)

    records = [
        EcgRecord(101, 201, tmp_path / "record-201"),
        EcgRecord(102, 202, tmp_path / "record-202"),
    ]
    calls = []

    def reader(path: Path):
        calls.append(path)
        return signal, {"fs": 200, "sig_name": ("II", "I")}

    prepared = prepare_ecg_records(
        records,
        tmp_path,
        record_reader=reader,
        target_samples=10,
    )
    prepare_ecg_records(
        records,
        tmp_path,
        record_reader=lambda _path: (_ for _ in ()).throw(AssertionError("resume decoded ECG")),
        target_samples=10,
        resume=True,
    )
    assert len(calls) == 2

    build_prepared_memmap(prepared, tmp_path)
    (tmp_path / "mds_ed.csv").write_text("synthetic\n", encoding="utf-8")
    audit = audit_prepared_memmap(
        tmp_path,
        expected_records=2,
        expected_samples=10,
        expected_channels=12,
    )
    assert audit.to_dict() == {
        "rows": 2,
        "waveform_files": 1,
        "channels": 12,
        "samples_per_record": 10,
        "dtype": "float32",
    }
