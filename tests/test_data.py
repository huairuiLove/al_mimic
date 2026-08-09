from __future__ import annotations

import csv

import numpy as np

from mimic_comal.data import MIMICRecord, audit_records
from mimic_comal.multimodal import build_structured_modalities, count_observed_measurement_bins


def test_subject_group_leakage_is_detected() -> None:
    labels = ("250", "401")
    clean = [
        MIMICRecord(0, "1", "10", "train", ("250",), "a"),
        MIMICRecord(1, "2", "11", "validation", ("401",), "b"),
        MIMICRecord(2, "3", "12", "test", ("250", "401"), "c"),
    ]
    assert not audit_records(clean, labels)["group_leakage"]
    leaked = clean + [MIMICRecord(3, "4", "10", "test", ("250",), "d")]
    assert audit_records(leaked, labels)["group_leakage"]


def test_structured_modalities_are_binned_and_train_normalized(tmp_path) -> None:
    def write_table(name: str, header: list[str], rows: list[list[object]]) -> None:
        with (tmp_path / name).open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)

    write_table(
        "PATIENTS.csv",
        ["SUBJECT_ID", "GENDER", "DOB"],
        [["10", "M", "1980-01-01 00:00:00"], ["11", "F", "1970-01-01 00:00:00"]],
    )
    write_table(
        "ADMISSIONS.csv",
        ["SUBJECT_ID", "HADM_ID", "ADMITTIME", "ADMISSION_TYPE"],
        [
            ["10", "1", "2010-01-01 00:00:00", "EMERGENCY"],
            ["11", "2", "2010-01-02 00:00:00", "ELECTIVE"],
        ],
    )
    write_table(
        "ICUSTAYS.csv",
        ["HADM_ID", "ICUSTAY_ID", "INTIME"],
        [["1", "100", "2010-01-01 01:00:00"], ["2", "200", "2010-01-02 01:00:00"]],
    )
    write_table(
        "CHARTEVENTS.csv",
        ["HADM_ID", "ICUSTAY_ID", "ITEMID", "CHARTTIME", "VALUENUM"],
        [
            ["1", "100", "211", "2010-01-01 01:30:00", "80"],
            ["1", "100", "646", "2010-01-01 03:30:00", "97"],
            ["2", "200", "220045", "2010-01-02 01:30:00", "90"],
            ["2", "200", "223761", "2010-01-02 02:00:00", "98.6"],
        ],
    )
    records = [
        MIMICRecord(0, "1", "10", "train", ("A",), "note one"),
        MIMICRecord(1, "2", "11", "validation", ("A",), "note two"),
    ]
    paths = {
        "patients": tmp_path / "PATIENTS.csv",
        "admissions": tmp_path / "ADMISSIONS.csv",
        "icustays": tmp_path / "ICUSTAYS.csv",
        "chartevents": tmp_path / "CHARTEVENTS.csv",
    }
    measurements, static, metadata = build_structured_modalities(
        records, paths, {"measurement_window_hours": 4, "measurement_bin_hours": 2}
    )
    observed_bins = count_observed_measurement_bins(
        records, paths, {"measurement_window_hours": 4, "measurement_bin_hours": 2}
    )
    assert measurements.shape == (2, 2 * 7 * 2)
    assert static.shape == (2, 8)
    assert metadata["measurement_shape"] == [2, 14]
    assert np.isfinite(measurements).all()
    assert measurements[0].sum() >= 2.0  # two observation-mask entries
    assert observed_bins.tolist() == [2, 1]
