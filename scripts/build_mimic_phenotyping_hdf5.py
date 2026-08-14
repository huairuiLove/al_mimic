#!/usr/bin/env python3
"""Adapt official MIMIC-III phenotyping artifacts to the active-learning HDF5 contract."""

from __future__ import annotations

import argparse
import csv
import gzip
import importlib.util
import json
import sqlite3
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import h5py
import numpy as np

from mimic_comal.tasks import TASKS, task_manifest


SPLITS = ("train", "val", "test")
UPSTREAM_BENCHMARK = Path("third_party/mimic3-benchmarks")


@dataclass(frozen=True, slots=True)
class ListfileRow:
    name: str
    period_length: float
    labels: tuple[int, ...]
    subject_id: int


@dataclass(frozen=True, slots=True)
class Stay:
    subject_id: int
    hadm_id: int
    stay_id: int
    intime: str
    outtime: str


def _open_csv(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", newline="", encoding="utf-8")
    return path.open("r", newline="", encoding="utf-8")


def read_listfile(path: Path) -> tuple[tuple[str, ...], list[ListfileRow]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if header[:2] != ["stay", "period_length"]:
            raise ValueError(f"unexpected phenotyping listfile header in {path}: {header[:2]}")
        labels = tuple(header[2:])
        rows = []
        for values in reader:
            if len(values) != len(header):
                raise ValueError(f"malformed row in {path}: expected {len(header)} columns")
            name = values[0]
            subject_token = name.split("_", 1)[0]
            if not subject_token.isdigit():
                raise ValueError(f"cannot recover SUBJECT_ID from listfile sample {name!r}")
            rows.append(
                ListfileRow(
                    name=name,
                    period_length=float(values[1]),
                    labels=tuple(int(value) for value in values[2:]),
                    subject_id=int(subject_token),
                )
            )
    return labels, rows


def _listfile(task_root: Path, split: str) -> Path:
    path = task_root / f"{split}_listfile.csv"
    if path.is_file():
        return path
    fallback = task_root / ("test" if split == "test" else "train") / "listfile.csv"
    if split != "val" and fallback.is_file():
        return fallback
    raise FileNotFoundError(
        f"missing {path}; run the upstream mimic3models.split_train_val command first"
    )


def load_listfiles(
    task_root: Path, task_id: str
) -> tuple[tuple[str, ...], dict[str, list[ListfileRow]]]:
    split_rows: dict[str, list[ListfileRow]] = {}
    headers: list[tuple[str, ...]] = []
    for split in SPLITS:
        header, rows = read_listfile(_listfile(task_root, split))
        headers.append(header)
        split_rows[split] = rows
    if any(header != headers[0] for header in headers[1:]):
        raise ValueError("train/val/test phenotyping label headers differ")
    if task_id == "phenotyping_25" and len(headers[0]) != 25:
        raise ValueError(f"official 25-label listfile contains {len(headers[0])} labels")
    split_subjects = {
        split: {row.subject_id for row in rows} for split, rows in split_rows.items()
    }
    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1 :]:
            overlap = split_subjects[left] & split_subjects[right]
            if overlap:
                raise ValueError(f"subject leakage across {left}/{right}: {sorted(overlap)[:5]}")
    return headers[0], split_rows


def load_ccs_labels(
    path: Path,
) -> tuple[tuple[str, ...], dict[int, tuple[int, tuple[int, ...]]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        if header[:2] != ["SUBJECT_ID", "ICUSTAY_ID"]:
            raise ValueError("CCS label table must start with SUBJECT_ID,ICUSTAY_ID")
        labels = tuple(header[2:])
        if len(set(labels)) != len(labels):
            raise ValueError("CCS label names must be unique")
        values: dict[int, tuple[int, tuple[int, ...]]] = {}
        for row in reader:
            if len(row) != len(header):
                raise ValueError("malformed CCS label row")
            stay_id = int(row[1])
            if stay_id in values:
                raise ValueError(f"duplicate CCS label row for ICU stay {stay_id}")
            subject_id = int(row[0])
            if subject_id < 1 or stay_id < 1:
                raise ValueError("CCS SUBJECT_ID and ICUSTAY_ID values must be positive")
            row_labels = tuple(int(value) for value in row[2:])
            if any(value not in (0, 1) for value in row_labels):
                raise ValueError(f"non-binary CCS label for ICU stay {stay_id}")
            values[stay_id] = (subject_id, row_labels)
    if len(labels) != 172:
        raise ValueError(f"CCS label table contains {len(labels)} labels, expected 172")
    return labels, values


def _raw_table(root: Path, name: str) -> Path:
    for candidate in (root / f"{name}.csv", root / f"{name}.csv.gz"):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"missing {name}.csv or {name}.csv.gz under {root}")


def load_stays(mimic_root: Path) -> dict[int, Stay]:
    result: dict[int, Stay] = {}
    with _open_csv(_raw_table(mimic_root, "ICUSTAYS")) as handle:
        for row in csv.DictReader(handle):
            stay = Stay(
                subject_id=int(row["SUBJECT_ID"]),
                hadm_id=int(row["HADM_ID"]),
                stay_id=int(row["ICUSTAY_ID"]),
                intime=row["INTIME"],
                outtime=row["OUTTIME"],
            )
            result[stay.stay_id] = stay
    return result


def _episode_stay_id(subject_root: Path, partition: str, row: ListfileRow) -> int:
    episode_name = row.name.split("_", 1)[1].replace("_timeseries.csv", ".csv")
    candidates = (
        subject_root / partition / str(row.subject_id) / episode_name,
        subject_root / str(row.subject_id) / episode_name,
    )
    metadata = next((path for path in candidates if path.is_file()), None)
    if metadata is None:
        raise FileNotFoundError(f"missing episode metadata for {row.name}: {candidates}")
    with metadata.open(newline="", encoding="utf-8") as handle:
        values = next(csv.DictReader(handle))
    return int(float(values["Icustay"]))


def attach_stays(
    split_rows: dict[str, list[ListfileRow]],
    subject_root: Path,
    stays: dict[int, Stay],
) -> dict[str, list[tuple[ListfileRow, Stay]]]:
    result: dict[str, list[tuple[ListfileRow, Stay]]] = {}
    for split, rows in split_rows.items():
        partition = "test" if split == "test" else "train"
        attached = []
        for row in rows:
            stay_id = _episode_stay_id(subject_root, partition, row)
            stay = stays.get(stay_id)
            if stay is None or stay.subject_id != row.subject_id:
                raise ValueError(f"invalid stay/subject mapping for {row.name}: {stay_id}")
            attached.append((row, stay))
        result[split] = attached
    return result


def build_note_index(notes_csv: Path, hadm_ids: set[int], database: Path) -> None:
    connection = sqlite3.connect(database)
    connection.execute(
        "CREATE TABLE notes (hadm_id INTEGER, chart_time TEXT, row_id INTEGER, text TEXT)"
    )
    batch: list[tuple[int, str, int, str]] = []
    with _open_csv(notes_csv) as handle:
        for position, row in enumerate(csv.DictReader(handle)):
            raw_hadm = row.get("HADM_ID", "")
            if not raw_hadm or int(float(raw_hadm)) not in hadm_ids:
                continue
            if str(row.get("ISERROR", "")).strip() == "1" or not row.get("TEXT", "").strip():
                continue
            batch.append(
                (
                    int(float(raw_hadm)),
                    row.get("CHARTTIME") or row.get("CHARTDATE") or "",
                    int(row.get("ROW_ID") or position),
                    row["TEXT"],
                )
            )
            if len(batch) >= 1000:
                connection.executemany("INSERT INTO notes VALUES (?, ?, ?, ?)", batch)
                batch.clear()
    if batch:
        connection.executemany("INSERT INTO notes VALUES (?, ?, ?, ?)", batch)
    connection.execute("CREATE INDEX notes_hadm_time ON notes (hadm_id, chart_time, row_id)")
    connection.commit()
    connection.close()


def note_text(connection: sqlite3.Connection, stay: Stay, max_characters: int = 250_000) -> str:
    chunks: list[str] = []
    size = 0
    cursor = connection.execute(
        "SELECT text FROM notes WHERE hadm_id=? AND chart_time>=? AND chart_time<=? "
        "ORDER BY chart_time, row_id",
        (stay.hadm_id, stay.intime, stay.outtime),
    )
    for (text,) in cursor:
        chunks.append(str(text))
        size += len(str(text))
        if size >= max_characters:
            break
    return "\n".join(chunks)


def has_note(connection: sqlite3.Connection, stay: Stay) -> bool:
    row = connection.execute(
        "SELECT 1 FROM notes WHERE hadm_id=? AND chart_time>=? AND chart_time<=? LIMIT 1",
        (stay.hadm_id, stay.intime, stay.outtime),
    ).fetchone()
    return row is not None


def _upstream_preprocessing(upstream_root: Path) -> Any:
    path = upstream_root / "mimic3models/preprocessing.py"
    module_spec = importlib.util.spec_from_file_location("official_mimic3_preprocessing", path)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f"cannot load official preprocessing module: {path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


class StructuredEncoder:
    def __init__(self, upstream_root: Path, max_steps: int) -> None:
        module = _upstream_preprocessing(upstream_root)
        self.max_steps = max_steps
        self.discretizer = module.Discretizer(
            timestep=1.0,
            store_masks=True,
            impute_strategy="previous",
            start_time="zero",
            config_path=str(upstream_root / "mimic3models/resources/discretizer_config.json"),
        )
        self.normalizer_class = module.Normalizer
        self.normalizer_path = upstream_root / (
            "mimic3models/phenotyping/ph_ts1.0.input_str-previous.start_time-zero.normalizer"
        )
        self.normalizer = None

    def encode(self, path: Path, period_length: float) -> tuple[np.ndarray, np.ndarray]:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            rows = np.asarray(list(reader), dtype=object)
        if rows.size == 0:
            raise ValueError(f"empty time series: {path}")
        required_header = list(self.discretizer._header)
        missing = [name for name in required_header if name not in header]
        if missing:
            raise ValueError(f"time series {path} is missing benchmark variables: {missing}")
        positions = [header.index(name) for name in required_header]
        rows = rows[:, positions]
        header = required_header
        values, output_header = self.discretizer.transform(
            rows, header=header, end=min(float(period_length), float(self.max_steps))
        )
        if self.normalizer is None:
            continuous = [
                index for index, name in enumerate(output_header.split(",")) if "->" not in name
            ]
            self.normalizer = self.normalizer_class(fields=continuous)
            self.normalizer.load_params(str(self.normalizer_path))
        values = self.normalizer.transform(values).astype(np.float32)
        if values.shape[1] != 76:
            raise ValueError(f"official discretizer produced {values.shape[1]} features, expected 76")
        length = min(values.shape[0], self.max_steps)
        output = np.zeros((self.max_steps, 76), dtype=np.float32)
        output[:length] = values[:length]
        mask = np.zeros(self.max_steps, dtype=np.bool_)
        mask[:length] = True
        return output, mask


def _create_split_group(
    root: h5py.Group, split: str, count: int, max_steps: int, note_tokens: int, labels: int
) -> h5py.Group:
    group = root.create_group(split)
    compression = {"compression": "gzip", "compression_opts": 1}
    group.create_dataset("X", shape=(count, max_steps, 76), dtype="f4", **compression)
    group.create_dataset("time_series_mask", shape=(count, max_steps), dtype="?", **compression)
    group.create_dataset("s", shape=(count, 0), dtype="f4")
    group.create_dataset("input_ids", shape=(count, note_tokens), dtype="i4", **compression)
    group.create_dataset("token_type_ids", shape=(count, note_tokens), dtype="i1", **compression)
    group.create_dataset("attention_mask", shape=(count, note_tokens), dtype="i1", **compression)
    group.create_dataset("label", shape=(count, labels), dtype="i1", **compression)
    group.create_dataset("subject_id", shape=(count,), dtype="i8", **compression)
    group.create_dataset("stay_id", shape=(count,), dtype="i8", **compression)
    return group


def _time_series_path(task_root: Path, split: str, name: str) -> Path:
    return task_root / ("test" if split == "test" else "train") / name


def build_hdf5(args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoTokenizer

    spec = TASKS[args.task]
    listfile_labels, listfile_rows = load_listfiles(args.task_root, args.task)
    ccs_labels: dict[int, tuple[int, tuple[int, ...]]] | None = None
    label_names = listfile_labels
    if args.task == "phenotyping_ccs_172":
        if args.ccs_labels_csv is None:
            raise ValueError("phenotyping_ccs_172 requires --ccs-labels-csv")
        label_names, ccs_labels = load_ccs_labels(args.ccs_labels_csv)
    if len(label_names) != spec.label_count:
        raise ValueError(f"task {args.task} requires {spec.label_count} labels, got {len(label_names)}")

    stays = load_stays(args.mimic_root)
    examples = attach_stays(listfile_rows, args.subject_root, stays)
    all_stays = [stay for values in examples.values() for _row, stay in values]
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer))
    encoder = StructuredEncoder(args.upstream_root, args.max_time_steps)

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {args.output}; pass --overwrite explicitly")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    notes_csv = _raw_table(args.mimic_root, "NOTEEVENTS")
    with tempfile.TemporaryDirectory(prefix="mimic_notes_") as temporary:
        note_db = Path(temporary) / "notes.sqlite3"
        build_note_index(notes_csv, {stay.hadm_id for stay in all_stays}, note_db)
        connection = sqlite3.connect(note_db)
        dropped_without_notes: dict[str, int] = {}
        for split in SPLITS:
            before = len(examples[split])
            examples[split] = [
                example for example in examples[split] if has_note(connection, example[1])
            ]
            dropped_without_notes[split] = before - len(examples[split])
            if not examples[split]:
                raise ValueError(f"no multimodal rows remain in {split} after note filtering")
        with tempfile.NamedTemporaryFile(
            dir=args.output.parent,
            prefix=f".{args.output.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_output:
            temporary_path = Path(temporary_output.name)
        try:
            with h5py.File(temporary_path, "w") as handle:
                root = handle.create_group("with_notes")
                root.attrs["task_id"] = args.task
                root.attrs["native_multilabel"] = True
                root.attrs["query_unit"] = "icu_stay"
                string_type = h5py.string_dtype(encoding="utf-8")
                root.create_dataset(
                    "label_names",
                    data=np.asarray(label_names, dtype=object),
                    dtype=string_type,
                )
                for split in SPLITS:
                    rows = examples[split]
                    group = _create_split_group(
                        root,
                        split,
                        len(rows),
                        args.max_time_steps,
                        args.max_note_tokens,
                        len(label_names),
                    )
                    for index, (row, stay) in enumerate(rows):
                        values, series_mask = encoder.encode(
                            _time_series_path(args.task_root, split, row.name),
                            row.period_length,
                        )
                        text = note_text(connection, stay)
                        if not text.strip():
                            raise ValueError(f"no ICU-window note text for stay {stay.stay_id}")
                        tokens = tokenizer(
                            text,
                            max_length=args.max_note_tokens,
                            truncation=True,
                            padding="max_length",
                            return_attention_mask=True,
                        )
                        if ccs_labels is None:
                            labels = row.labels
                        else:
                            ccs_row = ccs_labels.get(stay.stay_id)
                            if ccs_row is None:
                                raise ValueError(
                                    f"CCS label table has no row for stay {stay.stay_id}"
                                )
                            ccs_subject_id, labels = ccs_row
                            if ccs_subject_id != stay.subject_id:
                                raise ValueError(
                                    "CCS label subject does not match ICU stay: "
                                    f"{ccs_subject_id} != {stay.subject_id}"
                                )
                        if len(labels) != len(label_names):
                            raise ValueError(f"wrong label width for stay {stay.stay_id}")
                        group["X"][index] = values
                        group["time_series_mask"][index] = series_mask
                        group["input_ids"][index] = np.asarray(
                            tokens["input_ids"], dtype=np.int32
                        )
                        group["attention_mask"][index] = np.asarray(
                            tokens["attention_mask"], dtype=np.int8
                        )
                        group["token_type_ids"][index] = np.asarray(
                            tokens.get(
                                "token_type_ids", [0] * args.max_note_tokens
                            ),
                            dtype=np.int8,
                        )
                        group["label"][index] = np.asarray(labels, dtype=np.int8)
                        group["subject_id"][index] = stay.subject_id
                        group["stay_id"][index] = stay.stay_id
            temporary_path.replace(args.output)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
        connection.close()
    return {
        "output": str(args.output),
        "task": task_manifest(
            {
                "task": {"id": args.task},
            }
        ),
        "split_counts": {split: len(rows) for split, rows in examples.items()},
        "dropped_without_icu_window_notes": dropped_without_notes,
        "label_count": len(label_names),
        "max_time_steps": args.max_time_steps,
        "time_series_dim": 76,
        "max_note_tokens": args.max_note_tokens,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("phenotyping_25", "phenotyping_ccs_172"), required=True)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--subject-root", type=Path, required=True)
    parser.add_argument("--mimic-root", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ccs-labels-csv", type=Path)
    parser.add_argument("--upstream-root", type=Path, default=UPSTREAM_BENCHMARK)
    parser.add_argument("--max-time-steps", type=int, default=256)
    parser.add_argument("--max-note-tokens", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.max_time_steps < 1 or args.max_note_tokens < 1:
        raise ValueError("max-time-steps and max-note-tokens must be positive")
    print(json.dumps(build_hdf5(args), indent=2))


if __name__ == "__main__":
    main()
