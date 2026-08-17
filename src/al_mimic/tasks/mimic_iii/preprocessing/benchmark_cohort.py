"""First-party MIMIC-III Benchmark cohort construction.

Reimplements the cohort chain of the YerevaNN MIMIC-III Benchmark directly over
the raw MIMIC-III v1.4 CSV tables, so the phenotyping pipeline never executes
the upstream checkout. The chain is, in order:

1. drop ICU stays with an intra-stay ward or care-unit transfer
   (``FIRST_WARDID == LAST_WARDID`` and ``FIRST_CAREUNIT == LAST_CAREUNIT``);
2. inner-join the surviving stays with ADMISSIONS and PATIENTS;
3. keep admissions with exactly one ICU stay;
4. drop stays of patients younger than 18 years at ICU admission
   (the >89 shifted date-of-birth convention maps to age 90, never negative).

The official benchmark split is a fixed subject assignment: ``testset.csv``
names the held-out test subjects and ``valset.csv`` the validation subjects
carved out of the training side. Both lists ship with the benchmark and are
materialised under ``resources/``; every other subject is train.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

RESOURCES = Path(__file__).with_name("resources")
MIN_BENCHMARK_AGE = 18


def _official_subjects(name: str) -> set[int]:
    subjects: set[int] = set()
    with (RESOURCES / name).open(encoding="utf-8") as handle:
        for line in handle:
            subject, flag = line.strip().split(",")
            if int(flag) == 1:
                subjects.add(int(subject))
    return subjects


def test_subjects() -> set[int]:
    return _official_subjects("testset.csv")


def validation_subjects() -> set[int]:
    return _official_subjects("valset.csv")


def assign_partitions(subjects: pd.Series) -> pd.Series:
    """Map subject ids to the official train/val/test assignment."""
    test = test_subjects()
    validation = validation_subjects()
    overlap = test & validation
    if overlap:
        raise ValueError(f"official valset and testset share subjects: {sorted(overlap)[:5]}")
    return pd.Series(
        [
            "test" if subject in test else "val" if subject in validation else "train"
            for subject in subjects
        ],
        index=subjects.index,
        dtype=object,
    )


def load_benchmark_stays(mimic_dir: str | Path) -> pd.DataFrame:
    """Build the benchmark cohort from the raw MIMIC-III tables."""
    mimic_dir = Path(mimic_dir)
    patients = pd.read_csv(
        mimic_dir / "PATIENTS.csv", usecols=["SUBJECT_ID", "GENDER", "DOB", "DOD"], parse_dates=["DOB"]
    )
    admissions = pd.read_csv(
        mimic_dir / "ADMISSIONS.csv",
        usecols=["SUBJECT_ID", "HADM_ID", "ADMITTIME", "DISCHTIME", "ETHNICITY"],
    )
    stays = pd.read_csv(mimic_dir / "ICUSTAYS.csv", parse_dates=["INTIME", "OUTTIME"])

    stays = stays[
        (stays["FIRST_WARDID"] == stays["LAST_WARDID"])
        & (stays["FIRST_CAREUNIT"] == stays["LAST_CAREUNIT"])
    ]
    stays = stays.merge(admissions, on=["SUBJECT_ID", "HADM_ID"], how="inner")
    stays = stays.merge(patients, on="SUBJECT_ID", how="inner")

    stays_per_admission = stays.groupby("HADM_ID")["ICUSTAY_ID"].transform("count")
    stays = stays[stays_per_admission == 1]

    age_years = (stays["INTIME"] - stays["DOB"]).dt.total_seconds() / 3600.0 / 24.0 / 365.0
    # MIMIC-III shifts dates of patients older than 89 into the far past, which
    # surfaces as a negative age; the benchmark reads those as 90-year-olds.
    age_years = age_years.where(age_years >= 0, 90.0)
    stays = stays.assign(AGE=age_years)
    stays = stays[stays["AGE"] >= MIN_BENCHMARK_AGE]

    los_hours = (stays["OUTTIME"] - stays["INTIME"]).dt.total_seconds() / 3600.0
    stays = stays.assign(LOS_HOURS=los_hours)
    return stays.reset_index(drop=True)[
        [
            "SUBJECT_ID",
            "HADM_ID",
            "ICUSTAY_ID",
            "LAST_CAREUNIT",
            "DBSOURCE",
            "INTIME",
            "OUTTIME",
            "LOS_HOURS",
            "AGE",
            "GENDER",
            "ETHNICITY",
        ]
    ].rename(columns={"LAST_CAREUNIT": "FIRST_CAREUNIT"})


def cohort_stays_table(mimic_dir: str | Path) -> pd.DataFrame:
    """Benchmark cohort with the official partition column attached."""
    stays = load_benchmark_stays(mimic_dir)
    stays = stays.assign(partition=assign_partitions(stays["SUBJECT_ID"]).to_numpy())
    return stays
