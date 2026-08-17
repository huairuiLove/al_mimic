#!/usr/bin/env python3
"""Build the benchmark cohort table and per-stay CCS diagnoses from raw MIMIC-III.

Stage one of the first-party phenotyping pipeline. Reads the five raw tables
(PATIENTS, ADMISSIONS, ICUSTAYS, DIAGNOSES_ICD, D_ICD_DIAGNOSES) and writes:

``prep/benchmark_icustays.csv``
    the benchmark cohort (no transfers, one stay per admission, adults) with
    the official subject partition column;
``prep/all_diagnoses.csv``
    cohort diagnoses expanded to ICU stays, with HCUP CCS group and
    use_in_benchmark flags attached.

Event filtering and the final label matrices happen in
``build_phenotyping_features``, which knows which stays carry measurements.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .benchmark_cohort import cohort_stays_table
from .benchmark_labels import attach_ccs_groups, load_ccs_definitions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimic-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()

    prep = args.data_dir / "prep"
    prep.mkdir(parents=True, exist_ok=True)

    stays = cohort_stays_table(args.mimic_dir)
    print(
        "benchmark cohort:",
        f"{len(stays):,} stays / {stays['SUBJECT_ID'].nunique():,} subjects",
        "(transfers and multiple stays per admission removed, age >= 18)",
        flush=True,
    )
    counts = stays["partition"].value_counts()
    print("partition sizes:", dict(counts), flush=True)
    stays.to_csv(prep / "benchmark_icustays.csv", index=False)
    print("wrote", prep / "benchmark_icustays.csv")

    codes = pd.read_csv(
        args.mimic_dir / "D_ICD_DIAGNOSES.csv",
        usecols=["ICD9_CODE", "SHORT_TITLE", "LONG_TITLE"],
        dtype={"ICD9_CODE": str},
    )
    diagnoses = pd.read_csv(
        args.mimic_dir / "DIAGNOSES_ICD.csv",
        usecols=["SUBJECT_ID", "HADM_ID", "SEQ_NUM", "ICD9_CODE"],
        dtype={"ICD9_CODE": str},
    )
    diagnoses = diagnoses.merge(codes, on="ICD9_CODE", how="inner")
    diagnoses = diagnoses.merge(
        stays[["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID"]].drop_duplicates(),
        on=["SUBJECT_ID", "HADM_ID"],
        how="inner",
    )
    diagnoses = attach_ccs_groups(diagnoses, load_ccs_definitions())
    diagnoses = diagnoses.sort_values(["ICUSTAY_ID", "SEQ_NUM"]).reset_index(drop=True)
    mapped = diagnoses["HCUP_CCS_2015"].notna().sum()
    print(
        f"cohort diagnoses: {len(diagnoses):,} rows, {mapped:,} carry a CCS group, "
        f"{int(diagnoses['USE_IN_BENCHMARK'].fillna(False).sum()):,} are benchmark-flagged",
        flush=True,
    )
    diagnoses.to_csv(prep / "all_diagnoses.csv", index=False)
    print("wrote", prep / "all_diagnoses.csv")


if __name__ == "__main__":
    main()
