"""Streaming MIMIC-III admission-level data preparation.

Only structured diagnosis codes and discharge summaries are used.  Splits are
grouped by subject, so repeated admissions from one patient never cross a
train/validation/test boundary.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .config import require_paths


@dataclass(frozen=True, slots=True)
class MIMICRecord:
    row_index: int
    hadm_id: str
    subject_id: str
    split: str
    labels: tuple[str, ...]
    text: str


_WHITESPACE_RE = re.compile(r"\s+")


def _read_rows(path: Path) -> Iterable[list[str]]:
    # Large decode buffers keep NOTEEVENTS.csv.gz from syscall-thrashing the 18-core quota.
    if path.suffix == ".gz":
        raw = gzip.open(path, "rb")
        handle = io.TextIOWrapper(
            io.BufferedReader(raw, buffer_size=8 * 1024 * 1024),
            encoding="utf-8",
            errors="replace",
            newline="",
        )
    else:
        handle = open(path, "rt", encoding="utf-8", errors="replace", newline="", buffering=8 * 1024 * 1024)
    with handle:
        # csv.reader over a buffered TextIO-like stream; avoid per-row Python wrapping.
        yield from csv.reader(handle)


def _code_prefix(value: str, length: int) -> str:
    code = value.strip().upper().replace(".", "")
    if not code:
        return ""
    return code[:length]


def _stable_fraction(value: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def _clean_text(value: str, max_chars: int) -> str:
    # Keep section content while removing line-level PHI formatting noise.
    text = _WHITESPACE_RE.sub(" ", value.replace("\x00", " ")).strip()
    return text[:max_chars]


def _select_split(subject_id: str, cfg: dict[str, Any]) -> str:
    seed = int(cfg.get("seed", 17))
    train = float(cfg.get("train_fraction", 0.8))
    validation = float(cfg.get("validation_fraction", 0.1))
    fraction = _stable_fraction(subject_id, seed)
    if fraction < train:
        return "train"
    if fraction < train + validation:
        return "validation"
    return "test"


def _admission_map(path: Path, max_records: int | None) -> dict[str, str]:
    rows = _read_rows(path)
    header = next(rows)
    positions = {name: header.index(name) for name in ("SUBJECT_ID", "HADM_ID")}
    result: dict[str, str] = {}
    for row in rows:
        if len(row) <= max(positions.values()):
            continue
        hadm, subject = row[positions["HADM_ID"]], row[positions["SUBJECT_ID"]]
        result[hadm] = subject
        if max_records and len(result) >= max_records:
            break
    return result


def _diagnosis_map(path: Path, admissions: dict[str, str], length: int) -> dict[str, set[str]]:
    rows = _read_rows(path)
    header = next(rows)
    hadm_i = header.index("HADM_ID")
    code_i = header.index("ICD9_CODE")
    min_width = max(hadm_i, code_i)
    result: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if len(row) <= min_width:
            continue
        hadm = row[hadm_i]
        if hadm in admissions:
            code = _code_prefix(row[code_i], length)
            if code:
                result[hadm].add(code)
    return result


def _note_map(
    path: Path,
    admissions: dict[str, str],
    categories: set[str],
    max_chars: int,
    max_notes: int,
) -> dict[str, str]:
    rows = _read_rows(path)
    header = next(rows)
    hadm_i = header.index("HADM_ID")
    category_i = header.index("CATEGORY")
    description_i = header.index("DESCRIPTION")
    text_i = header.index("TEXT")
    min_width = max(hadm_i, category_i, description_i, text_i)
    allowed_descriptions = {"", "report", "addendum"}
    chunks: dict[str, list[str]] = defaultdict(list)
    full: set[str] = set()
    for row in rows:
        if len(row) <= min_width:
            continue
        hadm = row[hadm_i]
        if hadm not in admissions or hadm in full:
            continue
        # MIMIC has several discharge-summary spellings; exact categories remain configurable.
        if categories and row[category_i].strip() not in categories:
            continue
        if row[description_i].strip().lower() not in allowed_descriptions:
            continue
        text = _clean_text(row[text_i], max_chars)
        if not text:
            continue
        bucket = chunks[hadm]
        bucket.append(text)
        if len(bucket) >= max_notes:
            full.add(hadm)
    return {hadm: "\n".join(values)[:max_chars] for hadm, values in chunks.items() if values}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def prepare_mimic(config: dict[str, Any], output_dir: str | Path | None = None) -> dict[str, Any]:
    """Create deterministic JSONL records and an audit manifest.

    The expensive NOTEVENTS scan happens only during this command.  Subsequent
    feature and active-learning commands read the compact JSONL artifact.
    """
    paths = require_paths(config)
    dataset_cfg = config.setdefault("dataset", {})
    prep_cfg = config.get("preprocessing", {})
    output = Path(output_dir or dataset_cfg.get("prepared_dir", "prepared/mimic_iii"))
    output.mkdir(parents=True, exist_ok=True)
    max_records = prep_cfg.get("max_records")
    max_records = int(max_records) if max_records else None
    admissions = _admission_map(paths["admissions"], max_records)
    categories = {str(value) for value in prep_cfg.get("note_categories", ["Discharge summary"])}
    # DIAGNOSES and NOTEEVENTS are independent once admissions are known; overlap their I/O.
    with ThreadPoolExecutor(max_workers=2) as pool:
        diagnosis_future = pool.submit(
            _diagnosis_map,
            paths["diagnoses"],
            admissions,
            int(prep_cfg.get("code_prefix_length", 3)),
        )
        notes_future = pool.submit(
            _note_map,
            paths["notes"],
            admissions,
            categories,
            int(prep_cfg.get("max_text_chars", 12000)),
            int(prep_cfg.get("max_notes_per_admission", 4)),
        )
        diagnosis = diagnosis_future.result()
        notes = notes_future.result()
    candidates: list[tuple[str, str, str, set[str]]] = []
    for hadm, subject in admissions.items():
        if hadm in notes and diagnosis.get(hadm):
            candidates.append((hadm, subject, _select_split(subject, prep_cfg), diagnosis[hadm]))
    if not candidates:
        raise RuntimeError("no admissions have both a selected note and an ICD-9 diagnosis")
    # Label vocabulary is estimated from train only, preventing test-label leakage.
    frequencies = Counter(code for _, _, split, codes in candidates if split == "train" for code in codes)
    minimum = int(prep_cfg.get("min_label_frequency", 50))
    top_k = int(prep_cfg.get("label_top_k", 50))
    labels = [code for code, count in frequencies.most_common() if count >= minimum]
    if top_k > 0:
        labels = labels[:top_k]
    if not labels:
        raise RuntimeError("label vocabulary is empty; lower preprocessing.min_label_frequency")
    label_set = set(labels)
    records: list[MIMICRecord] = []
    for hadm, subject, split, codes in sorted(candidates, key=lambda item: int(item[0])):
        selected = tuple(sorted(codes & label_set))
        if not selected:
            continue
        records.append(MIMICRecord(len(records), hadm, subject, split, selected, notes[hadm]))
    if len({record.split for record in records}) < 3:
        # Tiny smoke-test datasets can hash into one partition.  Keep the main
        # protocol grouped while guaranteeing all artifacts are usable.
        subjects = sorted({record.subject_id for record in records})
        if len(subjects) < 3:
            raise RuntimeError("at least three subjects are required for grouped splits")
        train_boundary = max(1, min(len(subjects) - 2, int(len(subjects) * 0.8)))
        validation_boundary = max(train_boundary + 1, min(len(subjects) - 1, int(len(subjects) * 0.9)))
        split_by_subject = {
            subject: (
                "train" if index < train_boundary else "validation" if index < validation_boundary else "test"
            )
            for index, subject in enumerate(subjects)
        }
        ordered = sorted(records, key=lambda record: int(record.hadm_id))
        records = [
            MIMICRecord(
                index,
                record.hadm_id,
                record.subject_id,
                split_by_subject[record.subject_id],
                record.labels,
                record.text,
            )
            for index, record in enumerate(ordered)
        ]
    with (output / "records.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
    audit = audit_records(records, tuple(labels))
    _write_json(output / "labels.json", {"labels": labels, "frequencies": dict(frequencies)})
    _write_json(output / "audit.json", audit)
    _write_json(
        output / "manifest.json",
        {
            "format_version": 1,
            "source": {name: str(path.resolve()) for name, path in paths.items()},
            "records": len(records),
            "labels": labels,
            "preprocessing": prep_cfg,
            "split_group": "SUBJECT_ID",
        },
    )
    return {"output_dir": str(output), "records": len(records), "labels": len(labels), "audit": audit}


def load_records(prepared_dir: str | Path) -> list[MIMICRecord]:
    path = Path(prepared_dir) / "records.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"prepared records not found: {path}; run `prepare` first")
    try:
        import orjson

        loads = orjson.loads
    except ImportError:  # pragma: no cover - stdlib fallback
        loads = json.loads
    records: list[MIMICRecord] = []
    # Large buffered reads keep the 18-core host in sequential decode instead of syscall chatter.
    with path.open("rb", buffering=8 * 1024 * 1024) as handle:
        for raw in handle:
            payload = loads(raw)
            records.append(
                MIMICRecord(
                    int(payload["row_index"]),
                    str(payload["hadm_id"]),
                    str(payload["subject_id"]),
                    str(payload["split"]),
                    tuple(payload["labels"]),
                    str(payload["text"]),
                )
            )
    return records


def audit_records(records: list[MIMICRecord], labels: tuple[str, ...]) -> dict[str, Any]:
    label_set = set(labels)
    result: dict[str, Any] = {"records": len(records), "labels": len(labels), "splits": {}}
    groups: dict[str, set[str]] = {}
    # Single pass over records instead of three list comprehensions.
    split_records: dict[str, list[MIMICRecord]] = {"train": [], "validation": [], "test": []}
    coverage = Counter()
    for record in records:
        split_records.setdefault(record.split, []).append(record)
        coverage.update(record.labels)
    for split in ("train", "validation", "test"):
        subset = split_records.get(split, [])
        counts = Counter(label for record in subset for label in record.labels)
        subjects = {record.subject_id for record in subset}
        result["splits"][split] = {
            "records": len(subset),
            "subjects": len(subjects),
            "positive_counts": {label: int(counts.get(label, 0)) for label in labels},
            "cardinality_mean": sum(len(record.labels) for record in subset) / max(len(subset), 1),
        }
        groups[split] = subjects
    result["group_leakage"] = bool(
        groups["train"] & groups["validation"]
        or groups["train"] & groups["test"]
        or groups["validation"] & groups["test"]
    )
    result["label_coverage"] = {label: int(coverage.get(label, 0)) for label in label_set}
    return result
