from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from al_mimic.tasks.mds_ed.audit import audit_release_csv
from al_mimic.tasks.mds_ed.tabular import fit_tabular_transform, transform_tabular_chunks

FEATURE_COLUMNS = (
    "demographics_gender",
    "demographics_ethnicity_asian",
    "demographics_ethnicity_black",
    "demographics_ethnicity_hispanic",
    "demographics_ethnicity_other",
    "demographics_ethnicity_white",
    "labvalues_creatinine",
)
DIAGNOSIS_COLUMNS = ("diagnoses_a", "diagnoses_b")
DETERIORATION_COLUMNS = ("deterioration_a",)


def _write_release(path: Path, *, leak_subject: bool = False) -> None:
    columns = (
        "general_subject_id",
        "general_study_id",
        "general_strat_fold",
        "general_ecg_no_within_stay",
        "general_data",
        "general_ed_stay_id",
        *FEATURE_COLUMNS,
        *DIAGNOSIS_COLUMNS,
        *DETERIORATION_COLUMNS,
    )
    rows = []
    for fold in range(20):
        subject = 1_000 if leak_subject and fold == 19 else 1_000 + fold
        ethnicity = [float(index == fold % 5) for index in range(5)]
        rows.append(
            [
                subject,
                10_000 + fold,
                fold,
                0,
                f"record-{fold}",
                20_000 + fold,
                float(fold % 2),
                *ethnicity,
                "" if fold == 0 else float(fold),
                int(fold % 3 == 0),
                int(fold % 4 == 0),
                int(fold % 2),
            ]
        )
    for fold in (18, 19):
        ethnicity = [float(index == fold % 5) for index in range(5)]
        rows.append(
            [
                1_000 + fold,
                11_000 + fold,
                fold,
                1,
                f"repeat-{fold}",
                20_000 + fold,
                float(fold % 2),
                *ethnicity,
                float(fold),
                1,
                0,
                0,
            ]
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)


def test_release_audit_streams_and_detects_subject_leakage(tmp_path: Path) -> None:
    release = tmp_path / "mds_ed.csv"
    _write_release(release)

    audit = audit_release_csv(
        release,
        expected_diagnosis_labels=2,
        expected_deterioration_labels=1,
        expected_raw_features=7,
    )

    assert audit.rows == 22
    assert audit.patients == 20
    assert audit.visits == 20
    assert audit["folds"] == list(range(20))
    assert audit.zero_diagnosis_rows > 0

    leaked = tmp_path / "leaked.csv"
    _write_release(leaked, leak_subject=True)
    with pytest.raises(ValueError, match="subject leakage"):
        audit_release_csv(
            leaked,
            expected_diagnosis_labels=2,
            expected_deterioration_labels=1,
            expected_raw_features=7,
        )


def test_tabular_transform_is_chunked_train_fitted_and_filters_repeat_ecgs(
    tmp_path: Path,
) -> None:
    release = tmp_path / "mds_ed.csv"
    _write_release(release)

    spec = fit_tabular_transform(
        release,
        chunksize=3,
        expected_diagnosis_labels=2,
        expected_deterioration_labels=1,
        expected_raw_features=7,
        workspace=tmp_path,
    )
    batches = list(transform_tabular_chunks(release, spec, chunksize=3))

    assert spec.categorical_columns == (
        "demographics_gender",
        "demographics_ethnicity",
    )
    assert spec.continuous_columns == (
        "labvalues_creatinine",
        "labvalues_creatinine_nan",
    )
    assert dict(zip(spec.input_columns, spec.medians))["labvalues_creatinine"] == 9.0
    assert sum(batch.rows for batch in batches) == 20
    assert max(batch.rows for batch in batches) <= 3
    assert all(batch.labels.shape[1] == 2 for batch in batches)
    assert all(np.isfinite(batch.continuous).all() for batch in batches)
    assert all(batch.categorical.dtype == np.int64 for batch in batches)
