#!/usr/bin/env python3
"""Build the 239-label CCS matrix from cohort tables.

Consumes the stage-one artifacts of the first-party phenotyping pipeline (or
tables with the same columns): a stays CSV with ICUSTAY_ID/SUBJECT_ID and a
per-stay diagnoses CSV with ICUSTAY_ID/ICD9_CODE. The selection rule -- CCS
groups occurring in at least ``--minimum-episodes`` stays, ordered by HCUP CCS
id then name -- must yield exactly ``--expected-labels`` columns or the build
fails, because the label width is part of the task contract.

The source paper states this rule as "at least 30 episodes" and reports 172
phenotypes, but neither the paper nor its released code narrows the rule to
172: the paper never enumerates the phenotypes and the code selects only the
25 groups flagged ``use_in_benchmark``. The stated rule applied to this
cohort selects 239 groups, so 239 is the width this repository commits to.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pandas as pd

from .benchmark_labels import (
    DEFINITIONS_YAML,
    attach_ccs_groups,
    ccs_239_labels,
    load_ccs_definitions,
)

UPSTREAM_REVISION = "fa378b828fb1f832635c4259c3dff97ab81bd19d"


def _read_stays(path: Path) -> tuple[dict[int, int], set[int]]:
    subjects: dict[int, int] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            stay_id = int(float(row["ICUSTAY_ID"]))
            subject_id = int(float(row["SUBJECT_ID"]))
            if stay_id in subjects and subjects[stay_id] != subject_id:
                raise ValueError(f"ICU stay {stay_id} maps to multiple subjects")
            subjects[stay_id] = subject_id
    return subjects, set(subjects)


def build_ccs_labels(
    stays_csv: Path,
    diagnoses_csv: Path,
    definitions_yaml: Path,
    *,
    minimum_episodes: int,
    expected_labels: int,
) -> tuple[dict[int, int], tuple[str, ...], dict[int, set[str]], dict[str, int]]:
    """Legacy frame-based entry point kept for the CSV-driven shell wrapper."""
    subjects, _ = _read_stays(stays_csv)
    stays = pd.DataFrame(
        {"ICUSTAY_ID": sorted(subjects), "SUBJECT_ID": [subjects[stay] for stay in sorted(subjects)]}
    )
    diagnoses = pd.read_csv(diagnoses_csv, dtype={"ICD9_CODE": str})[
        ["ICUSTAY_ID", "ICD9_CODE"]
    ].dropna(subset=["ICUSTAY_ID"])
    diagnoses["ICUSTAY_ID"] = diagnoses["ICUSTAY_ID"].astype(int)
    definitions = load_ccs_definitions(definitions_yaml)
    diagnoses = attach_ccs_groups(diagnoses, definitions)
    labels, names = ccs_239_labels(
        stays,
        diagnoses,
        definitions,
        minimum_episodes=minimum_episodes,
        expected_labels=expected_labels,
    )
    labels_by_stay = {
        int(stay): {name for name, flag in zip(names, row, strict=True) if flag}
        for stay, row in zip(stays["ICUSTAY_ID"], labels, strict=True)
    }
    counts = {
        name: int(labels[:, position].sum()) for position, name in enumerate(names)
    }
    return subjects, tuple(names), labels_by_stay, counts


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stays-csv", type=Path, required=True)
    parser.add_argument("--diagnoses-csv", type=Path, required=True)
    parser.add_argument(
        "--definitions-yaml",
        type=Path,
        default=DEFINITIONS_YAML,
        help="first-party materialized CCS definition artifact",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-episodes", type=int, default=30)
    parser.add_argument("--expected-labels", type=int, default=239)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.minimum_episodes < 1 or args.expected_labels < 1:
        raise ValueError("minimum-episodes and expected-labels must be positive")
    subjects, labels, labels_by_stay, counts = build_ccs_labels(
        args.stays_csv,
        args.diagnoses_csv,
        args.definitions_yaml,
        minimum_episodes=args.minimum_episodes,
        expected_labels=args.expected_labels,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["SUBJECT_ID", "ICUSTAY_ID", *labels])
        for stay_id in sorted(subjects):
            present = labels_by_stay.get(stay_id, set())
            writer.writerow([subjects[stay_id], stay_id, *(int(label in present) for label in labels)])
    manifest = {
        "task_id": "phenotyping_ccs_239",
        "native_multilabel": True,
        "query_unit": "icu_stay",
        "source_repository": "https://github.com/amoldwin/notes_benchmark",
        "source_revision": UPSTREAM_REVISION,
        "minimum_episodes": args.minimum_episodes,
        "label_count": len(labels),
        "labels": list(labels),
        "positive_counts": {label: counts[label] for label in labels},
        "stay_count": len(subjects),
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "manifest": str(manifest_path), **manifest}, indent=2))


if __name__ == "__main__":
    main()
