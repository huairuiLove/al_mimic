"""MIMIC-III structured modalities used by the scratch multimodal encoder."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
from tqdm import tqdm

from .data import MIMICRecord, _read_rows


@dataclass(frozen=True, slots=True)
class AdmissionContext:
    subject_id: str
    admit_time: datetime
    icu_start: datetime | None
    icu_stay_id: str | None
    age: float
    gender: str
    admission_type: str


def _parse_time(value: str) -> datetime | None:
    value = value.strip()
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _columns(header: list[str], names: tuple[str, ...]) -> dict[str, int]:
    missing = [name for name in names if name not in header]
    if missing:
        raise ValueError(f"missing columns {missing!r}")
    return {name: header.index(name) for name in names}


def _patient_demographics(path: Path) -> dict[str, tuple[str, datetime | None]]:
    rows = _read_rows(path)
    positions = _columns(next(rows), ("SUBJECT_ID", "GENDER", "DOB"))
    return {
        row[positions["SUBJECT_ID"]]: (
            row[positions["GENDER"]].strip().upper(),
            _parse_time(row[positions["DOB"]]),
        )
        for row in rows
        if len(row) > max(positions.values())
    }


def _first_icu_stays(path: Path, selected_hadm: set[str]) -> dict[str, tuple[datetime, str]]:
    rows = _read_rows(path)
    positions = _columns(next(rows), ("HADM_ID", "ICUSTAY_ID", "INTIME"))
    result: dict[str, tuple[datetime, str]] = {}
    for row in rows:
        if len(row) <= max(positions.values()):
            continue
        hadm_id = row[positions["HADM_ID"]]
        if hadm_id not in selected_hadm:
            continue
        intime = _parse_time(row[positions["INTIME"]])
        if intime is None:
            continue
        current = result.get(hadm_id)
        if current is None or intime < current[0]:
            result[hadm_id] = (intime, row[positions["ICUSTAY_ID"]])
    return result


def load_admission_contexts(
    records: list[MIMICRecord], paths: dict[str, Path]
) -> dict[str, AdmissionContext]:
    selected = {record.hadm_id for record in records}
    demographics = _patient_demographics(paths["patients"])
    first_icu = _first_icu_stays(paths["icustays"], selected)
    rows = _read_rows(paths["admissions"])
    positions = _columns(next(rows), ("SUBJECT_ID", "HADM_ID", "ADMITTIME", "ADMISSION_TYPE"))
    result: dict[str, AdmissionContext] = {}
    for row in rows:
        if len(row) <= max(positions.values()):
            continue
        hadm_id = row[positions["HADM_ID"]]
        if hadm_id not in selected:
            continue
        admit_time = _parse_time(row[positions["ADMITTIME"]])
        if admit_time is None:
            continue
        subject_id = row[positions["SUBJECT_ID"]]
        gender, dob = demographics.get(subject_id, ("", None))
        age = (admit_time - dob).days / 365.2425 if dob is not None else 0.0
        # MIMIC-III shifts dates for patients over 89; the benchmark convention
        # represents them as approximately 91.4 rather than the shifted age.
        if age > 120:
            age = 91.4
        age = float(np.clip(age, 0.0, 100.0))
        stay = first_icu.get(hadm_id)
        result[hadm_id] = AdmissionContext(
            subject_id,
            admit_time,
            stay[0] if stay else None,
            stay[1] if stay else None,
            age,
            gender,
            row[positions["ADMISSION_TYPE"]].strip().upper(),
        )
    return result


def _identity(value: float) -> float:
    return value


def _fahrenheit_to_celsius(value: float) -> float:
    return (value - 32.0) * (5.0 / 9.0)


# ITEMIDs cover CareVue and MetaVision variants of seven high-coverage vital
# channels. They are a strict subset of the physiological channels used by the
# MIMIC-III benchmark pipeline referenced by the selected paper.
_MEASUREMENTS: tuple[tuple[str, tuple[tuple[int, Callable[[float], float]], ...], float, float], ...] = (
    ("heart_rate", ((211, _identity), (220045, _identity)), 20.0, 250.0),
    (
        "systolic_blood_pressure",
        tuple((item, _identity) for item in (51, 442, 455, 6701, 220179, 220050)),
        20.0,
        300.0,
    ),
    (
        "diastolic_blood_pressure",
        tuple((item, _identity) for item in (8368, 8440, 8441, 8555, 220180, 220051)),
        10.0,
        200.0,
    ),
    (
        "mean_blood_pressure",
        tuple((item, _identity) for item in (52, 443, 456, 6702, 220052, 220181)),
        10.0,
        250.0,
    ),
    ("respiratory_rate", tuple((item, _identity) for item in (615, 618, 220210, 224690)), 2.0, 80.0),
    ("oxygen_saturation", tuple((item, _identity) for item in (646, 220277)), 1.0, 100.0),
    (
        "temperature_celsius",
        (
            (676, _identity),
            (223762, _identity),
            (678, _fahrenheit_to_celsius),
            (223761, _fahrenheit_to_celsius),
        ),
        20.0,
        45.0,
    ),
)


def measurement_names() -> list[str]:
    return [entry[0] for entry in _MEASUREMENTS]


def _measurement_lookup() -> dict[str, tuple[int, Callable[[float], float], float, float]]:
    result: dict[str, tuple[int, Callable[[float], float], float, float]] = {}
    for variable, (_name, item_ids, minimum, maximum) in enumerate(_MEASUREMENTS):
        for item_id, transform in item_ids:
            result[str(item_id)] = (variable, transform, minimum, maximum)
    return result


def build_structured_modalities(
    records: list[MIMICRecord], paths: dict[str, Path], cfg: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    contexts = load_admission_contexts(records, paths)
    row_by_hadm = {record.hadm_id: record.row_index for record in records}
    window_hours = float(cfg.get("measurement_window_hours", 48))
    bin_hours = float(cfg.get("measurement_bin_hours", 2))
    if window_hours <= 0 or bin_hours <= 0 or window_hours % bin_hours != 0:
        raise ValueError("measurement window and bin hours must be positive and evenly divisible")
    time_bins = int(window_hours / bin_hours)
    variable_count = len(_MEASUREMENTS)
    sums = np.zeros((len(records), time_bins, variable_count), dtype=np.float64)
    counts = np.zeros((len(records), time_bins, variable_count), dtype=np.uint16)
    lookup = _measurement_lookup()

    rows = _read_rows(paths["chartevents"])
    positions = _columns(next(rows), ("HADM_ID", "ICUSTAY_ID", "ITEMID", "CHARTTIME", "VALUENUM"))
    minimum_width = max(positions.values())
    for row in tqdm(rows, desc="scan MIMIC-III ICU measurements", unit="rows"):
        if len(row) <= minimum_width:
            continue
        item = lookup.get(row[positions["ITEMID"]])
        hadm_id = row[positions["HADM_ID"]]
        context = contexts.get(hadm_id)
        if item is None or context is None or context.icu_start is None:
            continue
        stay_id = row[positions["ICUSTAY_ID"]]
        if context.icu_stay_id and stay_id and stay_id != context.icu_stay_id:
            continue
        chart_time = _parse_time(row[positions["CHARTTIME"]])
        if chart_time is None:
            continue
        elapsed = (chart_time - context.icu_start).total_seconds() / 3600.0
        if elapsed < 0 or elapsed >= window_hours:
            continue
        try:
            raw_value = float(row[positions["VALUENUM"]])
        except ValueError:
            continue
        variable, transform, minimum, maximum = item
        value = transform(raw_value)
        if not np.isfinite(value) or value < minimum or value > maximum:
            continue
        record_row = row_by_hadm[hadm_id]
        time_bin = min(int(elapsed / bin_hours), time_bins - 1)
        sums[record_row, time_bin, variable] += value
        if counts[record_row, time_bin, variable] < np.iinfo(np.uint16).max:
            counts[record_row, time_bin, variable] += 1

    observed = counts > 0
    values = np.divide(sums, counts, out=np.zeros_like(sums), where=observed)
    train_mask = np.fromiter((record.split == "train" for record in records), dtype=bool, count=len(records))
    means = np.zeros(variable_count, dtype=np.float64)
    stds = np.ones(variable_count, dtype=np.float64)
    for variable in range(variable_count):
        selected = values[train_mask, :, variable][observed[train_mask, :, variable]]
        if selected.size:
            means[variable] = selected.mean()
            stds[variable] = max(float(selected.std()), 1e-6)
    normalized = (values - means[None, None, :]) / stds[None, None, :]
    normalized[~observed] = 0.0
    time_series = np.concatenate((normalized.astype(np.float32), observed.astype(np.float32)), axis=2)

    admission_types = ("EMERGENCY", "ELECTIVE", "URGENT", "NEWBORN")
    static = np.zeros((len(records), 8), dtype=np.float32)
    train_ages = [contexts[r.hadm_id].age for r in records if r.split == "train" and r.hadm_id in contexts]
    age_mean = float(np.mean(train_ages)) if train_ages else 0.0
    age_std = max(float(np.std(train_ages)), 1.0) if train_ages else 1.0
    for record in records:
        context = contexts.get(record.hadm_id)
        if context is None:
            continue
        row = record.row_index
        static[row, 0] = (context.age - age_mean) / age_std
        static[row, 1] = float(context.gender == "M")
        static[row, 2] = float(context.gender == "F")
        for offset, admission_type in enumerate(admission_types, start=3):
            static[row, offset] = float(context.admission_type == admission_type)
        static[row, 7] = float(context.icu_start is not None)

    return (
        time_series.reshape(len(records), -1),
        static,
        {
            "measurement_names": measurement_names(),
            "measurement_shape": [time_bins, variable_count * 2],
            "measurement_window_hours": window_hours,
            "measurement_bin_hours": bin_hours,
            "measurement_observed_fraction": float(observed.mean()),
            "measurement_train_mean": means.tolist(),
            "measurement_train_std": stds.tolist(),
            "static_names": [
                "age_z",
                "gender_male",
                "gender_female",
                "admission_emergency",
                "admission_elective",
                "admission_urgent",
                "admission_newborn",
                "has_icu_stay",
            ],
            "age_train_mean": age_mean,
            "age_train_std": age_std,
        },
    )
