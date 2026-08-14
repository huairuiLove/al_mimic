#!/usr/bin/env python3
"""Build the paper's 172-label CCS matrix from notes_benchmark cohort tables."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml


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


def _definitions(path: Path) -> tuple[dict[str, str], dict[str, int]]:
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    code_to_group: dict[str, str] = {}
    group_ids: dict[str, int] = {}
    for group, definition in values.items():
        group = str(group)
        group_ids[group] = int(definition["id"])
        for code in definition["codes"]:
            code = str(code)
            previous = code_to_group.get(code)
            if previous is not None and previous != group:
                raise ValueError(
                    f"ICD-9 code {code!r} maps to both {previous!r} and {group!r}"
                )
            code_to_group[code] = group
    duplicate_ids = [group_id for group_id, count in Counter(group_ids.values()).items() if count > 1]
    if duplicate_ids:
        raise ValueError(f"duplicate HCUP CCS ids in definitions: {duplicate_ids}")
    return code_to_group, group_ids


def build_ccs_labels(
    stays_csv: Path,
    diagnoses_csv: Path,
    definitions_yaml: Path,
    *,
    minimum_episodes: int,
    expected_labels: int,
) -> tuple[dict[int, int], tuple[str, ...], dict[int, set[str]], dict[str, int]]:
    subjects, cohort_stays = _read_stays(stays_csv)
    code_to_group, group_ids = _definitions(definitions_yaml)
    labels_by_stay: dict[int, set[str]] = defaultdict(set)
    with diagnoses_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            raw_stay = row.get("ICUSTAY_ID", "")
            if not raw_stay:
                continue
            stay_id = int(float(raw_stay))
            if stay_id not in cohort_stays:
                continue
            group = code_to_group.get(str(row.get("ICD9_CODE", "")).strip())
            if group is not None:
                labels_by_stay[stay_id].add(group)

    counts = Counter(group for groups in labels_by_stay.values() for group in groups)
    selected = tuple(
        sorted(
            (group for group, count in counts.items() if count >= minimum_episodes),
            key=lambda group: (group_ids[group], group),
        )
    )
    if len(selected) != expected_labels:
        raise ValueError(
            f"paper rule selected {len(selected)} CCS labels, expected {expected_labels}; "
            "verify that all_stays/all_diagnoses came from the authors' full filtered cohort"
        )
    return subjects, selected, labels_by_stay, dict(counts)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stays-csv", type=Path, required=True)
    parser.add_argument("--diagnoses-csv", type=Path, required=True)
    parser.add_argument(
        "--definitions-yaml",
        type=Path,
        default=Path("third_party/notes_benchmark/mimic3benchmark/resources/hcup_ccs_2015_definitions.yaml"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-episodes", type=int, default=30)
    parser.add_argument("--expected-labels", type=int, default=172)
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
            writer.writerow(
                [subjects[stay_id], stay_id, *(int(label in present) for label in labels)]
            )
    manifest = {
        "task_id": "phenotyping_ccs_172",
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
