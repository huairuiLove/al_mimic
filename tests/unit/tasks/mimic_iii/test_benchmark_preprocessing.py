from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from al_mimic.tasks.mimic_iii.preprocessing.benchmark_cohort import (
    assign_partitions,
    load_benchmark_stays,
)
from al_mimic.tasks.mimic_iii.preprocessing.benchmark_cohort import (
    test_subjects as official_test_subjects,
)
from al_mimic.tasks.mimic_iii.preprocessing.benchmark_cohort import (
    validation_subjects as official_validation_subjects,
)
from al_mimic.tasks.mimic_iii.preprocessing.benchmark_labels import (
    attach_ccs_groups,
    benchmark_25_labels,
    benchmark_label_names,
    ccs_239_labels,
    load_ccs_definitions,
)


def _write_mimic_tables(
    base: Path,
    *,
    transfers: bool = False,
    multi_stay_admission: bool = False,
    child_subject: bool = False,
    shifted_dob_subject: bool = False,
) -> None:
    """Two clean adult stays, plus opt-in stays that each violate one filter."""
    rows = [
        {"subject": 900001, "hadm": 9001, "icustay": 90001, "los_h": 24.0},
        {"subject": 900002, "hadm": 9002, "icustay": 90002, "los_h": 12.0},
    ]
    if transfers:
        rows.append(
            {"subject": 900003, "hadm": 9003, "icustay": 90003, "los_h": 24.0, "transfer": True}
        )
    if multi_stay_admission:
        rows.append({"subject": 900004, "hadm": 9004, "icustay": 90004, "los_h": 24.0})
        rows.append({"subject": 900004, "hadm": 9004, "icustay": 90005, "los_h": 24.0})
    if child_subject:
        rows.append({"subject": 900005, "hadm": 9005, "icustay": 90006, "los_h": 24.0, "child": True})
    if shifted_dob_subject:
        rows.append(
            {"subject": 900006, "hadm": 9006, "icustay": 90007, "los_h": 24.0, "shifted": True}
        )

    subjects = sorted({row["subject"] for row in rows})
    intimes = {row["icustay"]: pd.Timestamp("2101-01-01 00:00") for row in rows}

    def dob_of(subject: int) -> str:
        for row in rows:
            if row["subject"] == subject and row.get("child"):
                return "2100-06-01"  # six months old at admission
        for row in rows:
            if row["subject"] == subject and row.get("shifted"):
                return "2101-06-01"  # after INTIME: the >89 shifted-DOB artifact
        return "1980-01-01"

    patients = pd.DataFrame(
        {
            "SUBJECT_ID": subjects,
            "GENDER": ["M"] * len(subjects),
            "DOB": [dob_of(subject) for subject in subjects],
            "DOD": [None] * len(subjects),
        }
    )
    admissions = pd.DataFrame(
        {
            "SUBJECT_ID": [row["subject"] for row in rows],
            "HADM_ID": [row["hadm"] for row in rows],
            "ADMITTIME": ["2101-01-01 00:00"] * len(rows),
            "DISCHTIME": ["2101-01-05 00:00"] * len(rows),
            "ETHNICITY": ["WHITE"] * len(rows),
        }
    )
    stays = pd.DataFrame(
        {
            "SUBJECT_ID": [row["subject"] for row in rows],
            "HADM_ID": [row["hadm"] for row in rows],
            "ICUSTAY_ID": [row["icustay"] for row in rows],
            "DBSOURCE": ["metavision"] * len(rows),
            "FIRST_CAREUNIT": ["CCU" if row.get("transfer") else "MICU" for row in rows],
            "LAST_CAREUNIT": ["MICU"] * len(rows),
            "FIRST_WARDID": ["B" if row.get("transfer") else "A" for row in rows],
            "LAST_WARDID": ["A"] * len(rows),
            "INTIME": [intimes[row["icustay"]] for row in rows],
            "OUTTIME": [
                intimes[row["icustay"]] + pd.Timedelta(hours=row["los_h"]) for row in rows
            ],
        }
    )
    base.mkdir(parents=True, exist_ok=True)
    patients.to_csv(base / "PATIENTS.csv", index=False)
    admissions.to_csv(base / "ADMISSIONS.csv", index=False)
    stays.to_csv(base / "ICUSTAYS.csv", index=False)


def test_cohort_applies_the_benchmark_filters(tmp_path: Path) -> None:
    _write_mimic_tables(tmp_path)
    stays = load_benchmark_stays(tmp_path)
    assert set(stays["ICUSTAY_ID"]) == {90001, 90002}


def test_cohort_drops_transfer_stays(tmp_path: Path) -> None:
    _write_mimic_tables(tmp_path, transfers=True)
    stays = load_benchmark_stays(tmp_path)
    assert 90003 not in set(stays["ICUSTAY_ID"])


def test_cohort_drops_admissions_with_multiple_stays(tmp_path: Path) -> None:
    _write_mimic_tables(tmp_path, multi_stay_admission=True)
    stays = load_benchmark_stays(tmp_path)
    assert not set(stays["ICUSTAY_ID"]) & {90004, 90005}


def test_cohort_drops_children(tmp_path: Path) -> None:
    _write_mimic_tables(tmp_path, child_subject=True)
    stays = load_benchmark_stays(tmp_path)
    assert 90006 not in set(stays["ICUSTAY_ID"])


def test_shifted_dates_of_birth_read_as_age_90(tmp_path: Path) -> None:
    _write_mimic_tables(tmp_path, shifted_dob_subject=True)
    stays = load_benchmark_stays(tmp_path)
    assert 90007 in set(stays["ICUSTAY_ID"])
    assert float(stays.loc[stays["ICUSTAY_ID"] == 90007, "AGE"].iloc[0]) == 90.0


def test_partitions_follow_the_official_subject_lists() -> None:
    official_test = next(iter(official_test_subjects()))
    official_val = next(iter(official_validation_subjects()))
    subjects = pd.Series([official_test, official_val, 900001])
    partitions = assign_partitions(subjects)
    assert partitions.tolist() == ["test", "val", "train"]


def test_definitions_materialize_the_25_benchmark_groups() -> None:
    definitions = load_ccs_definitions()
    names = benchmark_label_names(definitions)
    assert len(names) == 25
    assert names == sorted(names)
    assert "Septicemia (except in labor)" in names


def _diagnoses_frame(rows: list[tuple[int, str]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["ICUSTAY_ID", "ICD9_CODE"])
    return attach_ccs_groups(frame, load_ccs_definitions())


def test_benchmark_25_labels_mark_only_benchmark_groups() -> None:
    definitions = load_ccs_definitions()
    stays = pd.DataFrame({"ICUSTAY_ID": [1, 2]})
    code = next(iter(definitions["Septicemia (except in labor)"]["codes"]))
    non_benchmark = next(
        code
        for group, definition in definitions.items()
        if not definition["use_in_benchmark"]
        for code in definition["codes"]
    )
    diagnoses = _diagnoses_frame([(1, code), (1, non_benchmark), (2, "999999")])
    labels, names = benchmark_25_labels(stays, diagnoses, definitions)
    assert labels.shape == (2, 25)
    assert labels.sum() == 1
    assert labels[0, names.index("Septicemia (except in labor)")] == 1
    assert names.index("Septicemia (except in labor)") == sorted(names).index(
        "Septicemia (except in labor)"
    )


def test_ccs_239_selection_orders_by_hcup_id_and_enforces_the_count() -> None:
    definitions = {
        "Beta": {"id": 2, "codes": ["001"], "use_in_benchmark": False},
        "Alpha": {"id": 1, "codes": ["002"], "use_in_benchmark": False},
        "Rare": {"id": 3, "codes": ["003"], "use_in_benchmark": False},
    }
    stays = pd.DataFrame({"ICUSTAY_ID": [11, 22, 33]})
    diagnoses = pd.DataFrame({"ICUSTAY_ID": [11, 22, 22, 33], "ICD9_CODE": ["001", "001", "002", "002"]})
    diagnoses = attach_ccs_groups(diagnoses, definitions)
    labels, names = ccs_239_labels(
        stays, diagnoses, definitions, minimum_episodes=2, expected_labels=2
    )
    assert names == ["Alpha", "Beta"]
    assert labels.tolist() == [[0, 1], [1, 1], [1, 0]]

    with pytest.raises(ValueError, match="expected 239"):
        ccs_239_labels(stays, diagnoses, definitions, minimum_episodes=1, expected_labels=239)
