#!/usr/bin/env python
"""Assemble with_notes splits.hdf5 from the FIDDLE artifacts.

Two modes share one row-order contract: population.hdf5 rows in FIDDLE's
feature order (asserted against IDs.csv), notes attached by ICUSTAY_ID, and
train/val/test taken from the partition column of prep/icustays_MV.csv.

assemble (default)
    Stream X/s from the dense Xs.hdf5 into with_notes/{train,val,test} and
    write the token tensors, labels, subject_id, careunit_code and
    notes_available alongside. The loader's audit requires subject_id, so an
    artifact assembled without it cannot pass validate-data.

attach (--attach-only)
    Upgrade an existing splits.hdf5 that predates the subject_id contract.
    The expected label order is rebuilt from the same population/partition
    mapping and must match the stored labels row by row before anything is
    written; missing arrays and root attributes are then added in place, so a
    15 GiB artifact is patched in seconds instead of being rewritten.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from ..scenarios import empty_note
from .notes import TOKEN_FIELDS, read_notes

SPLIT_NAMES = ("train", "val", "test")


def _label_metadata(data_dir: Path, task: str, duration: float) -> dict:
    """Label names and task id from the population sidecar, when one exists."""
    meta_path = data_dir / "population" / f"{task}_{duration}h_meta.json"
    if not meta_path.is_file():
        return {}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    names = meta.get("label_names") or meta.get("groups")
    payload: dict = {}
    if meta.get("task_id"):
        payload["task_id"] = str(meta["task_id"])
    if names:
        payload["label_names"] = [str(name) for name in names]
    return payload


def _write_label_metadata(root: h5py.Group, metadata: dict) -> None:
    if "task_id" in metadata and "task_id" not in root.attrs:
        root.attrs["task_id"] = metadata["task_id"]
    if "label_names" in metadata and "label_names" not in root:
        root.create_dataset(
            "label_names",
            data=np.array(metadata["label_names"], dtype=h5py.string_dtype(encoding="utf-8")),
        )


def load_cohort(
    data_dir: Path,
    task: str,
    duration: float,
    task_dir: Path,
    *,
    stays_csv: Path | None = None,
) -> pd.DataFrame:
    """Population rows in FIDDLE's feature order, with notes, subject and unit attached."""
    population = pd.read_hdf(data_dir / "population/population.hdf5", f"{task}_{duration}h").rename(
        columns={"ID": "ICUSTAY_ID", f"{task}_LABEL": "LABEL"}
    )
    population = population.reset_index(drop=True)

    feature_ids = pd.read_csv(task_dir / "IDs.csv")["ID"].to_numpy()
    if not np.array_equal(feature_ids, population["ICUSTAY_ID"].to_numpy()):
        raise SystemExit(
            "population order does not match FIDDLE IDs.csv; features and labels would "
            "be misaligned. Rebuild the population before rebuilding features."
        )
    population["feature_row"] = np.arange(len(population), dtype=np.int64)

    # Left join: a stay whose note has not been charted inside the window stays in
    # the cohort with an empty note and notes_available=0, because that absence is
    # modality heterogeneity, not a row to discard.
    notes = read_notes(task_dir / "notes.hdf5")
    cohort = population.merge(notes, on="ICUSTAY_ID", how="left")
    cohort["notes_available"] = cohort["input_ids"].notna()
    missing = int((~cohort["notes_available"]).sum())
    print(
        f"stays without a note in the window: {missing:,} / {len(cohort):,} "
        f"({100 * missing / max(len(cohort), 1):.1f}%)",
        flush=True,
    )
    if missing:
        token_length = int(len(cohort.loc[cohort["notes_available"], "input_ids"].iloc[0]))
        blank = empty_note(token_length)
        for field in TOKEN_FIELDS:
            gaps = cohort.index[~cohort["notes_available"]]
            cohort.loc[gaps, field] = pd.Series([blank[field].copy() for _ in gaps], index=gaps, dtype=object)

    stays = pd.read_csv(stays_csv or (data_dir / "prep/icustays_MV.csv"))[
        ["ICUSTAY_ID", "SUBJECT_ID", "FIRST_CAREUNIT", "partition"]
    ]
    cohort = cohort.merge(stays, on="ICUSTAY_ID", how="left")
    missing = cohort["partition"].isna().sum()
    if missing:
        raise SystemExit(f"{missing} cohort stays have no split assignment in icustays_MV.csv")
    if (cohort["SUBJECT_ID"] <= 0).any():
        raise SystemExit("icustays_MV.csv carries non-positive SUBJECT_ID values")
    return cohort


def _split_rows(cohort: pd.DataFrame, split: str) -> pd.DataFrame:
    return cohort[cohort["partition"] == split].reset_index(drop=True)


def _subject_leaks(output: Path, group: str) -> dict[str, int]:
    with h5py.File(output, "r") as handle:
        root = handle[group]
        subjects = {split: set(np.asarray(root[split]["subject_id"]).tolist()) for split in SPLIT_NAMES}
    return {
        f"{left}/{right}": len(subjects[left] & subjects[right])
        for index, left in enumerate(SPLIT_NAMES)
        for right in SPLIT_NAMES[index + 1 :]
        if subjects[left] & subjects[right]
    }


def assemble_splits(
    data_dir: Path,
    task: str,
    duration: float,
    timestep: float,
    *,
    group: str = "with_notes",
    chunk_rows: int = 512,
    compression: str | None = None,
    output: Path | None = None,
    stays_csv: Path | None = None,
) -> Path:
    """Write a fresh splits.hdf5 from Xs.hdf5, notes, labels and split assignments."""
    task_dir = data_dir / f"features/outcome={task},T={duration},dt={timestep}"
    target = output or (task_dir / "splits.hdf5")
    if target.exists():
        raise SystemExit(f"{target} already exists; remove it or pass --output to rebuild elsewhere")

    cohort = load_cohort(data_dir, task, duration, task_dir, stays_csv=stays_csv)
    print(f"cohort: {len(cohort):,}", flush=True)

    dense_path = task_dir / "Xs.hdf5"
    if not dense_path.is_file():
        raise SystemExit(
            f"{dense_path} is missing; assemble reads the dense float32 X/s tensors. "
            "Materialize Xs.hdf5 from the FIDDLE sparse artifacts first."
        )
    options = {"compression": compression} if compression else {}
    units = sorted(cohort["FIRST_CAREUNIT"].dropna().unique())
    careunit_codes = {unit: index for index, unit in enumerate(units)}

    with h5py.File(dense_path, "r") as dense, h5py.File(target, "w") as handle:
        time_series = dense["X"]
        static = dense["s"]
        step_mask = dense["step_mask"] if "step_mask" in dense else None
        rows_total, steps, features = time_series.shape
        if rows_total != len(cohort) or static.shape[0] != len(cohort):
            raise SystemExit(
                f"Xs.hdf5 holds {rows_total}/{static.shape[0]} rows for {len(cohort)} cohort stays"
            )
        root = handle.create_group(group)
        root.attrs["careunit_codes"] = json.dumps(careunit_codes)
        root.attrs["observation_hours"] = steps
        _write_label_metadata(root, _label_metadata(data_dir, task, duration))
        for split in SPLIT_NAMES:
            selected = _split_rows(cohort, split)
            count = len(selected)
            print(f"  {split}: {count:,} stays", flush=True)
            split_group = root.create_group(split)
            x = split_group.create_dataset("X", shape=(count, steps, features), dtype=np.float32, **options)
            s = split_group.create_dataset("s", shape=(count, static.shape[1]), dtype=np.float32, **options)
            mask = (
                split_group.create_dataset("time_series_mask", shape=(count, steps), dtype=bool, **options)
                if step_mask is not None
                else None
            )
            feature_rows = selected["feature_row"].to_numpy()
            for start in range(0, count, chunk_rows):
                window = feature_rows[start : start + chunk_rows]
                x[start : start + len(window)] = time_series[window]
                s[start : start + len(window)] = static[window]
                if mask is not None:
                    mask[start : start + len(window)] = step_mask[window]
                print(f"    rows {min(start + chunk_rows, count):,}/{count:,}", flush=True)
            for field in TOKEN_FIELDS:
                split_group.create_dataset(
                    field, data=np.stack(selected[field].to_numpy()).astype(np.int64), **options
                )
            split_group.create_dataset(
                "label", data=np.stack(selected["LABEL"].to_numpy()).astype(np.int64), **options
            )
            split_group.create_dataset("subject_id", data=selected["SUBJECT_ID"].to_numpy().astype(np.int64))
            split_group.create_dataset(
                "careunit_code",
                data=selected["FIRST_CAREUNIT"].map(careunit_codes).to_numpy().astype(np.int64),
            )
            split_group.create_dataset(
                "notes_available", data=selected["notes_available"].to_numpy().astype(bool)
            )

    leaks = _subject_leaks(target, group)
    if leaks:
        target.unlink(missing_ok=True)
        raise SystemExit(f"subject leakage across splits: {leaks}")
    print(f"wrote {target} ({target.stat().st_size / 2**30:.1f} GiB), no subject leakage", flush=True)
    return target


def attach_arrays(
    data_dir: Path,
    task: str,
    duration: float,
    timestep: float,
    *,
    group: str = "with_notes",
    splits_path: Path | None = None,
    stays_csv: Path | None = None,
) -> Path:
    """Add subject_id and companion arrays to an existing artifact, order-verified."""
    task_dir = data_dir / f"features/outcome={task},T={duration},dt={timestep}"
    target = splits_path or (task_dir / "splits.hdf5")
    if not target.is_file():
        raise SystemExit(f"{target} does not exist; nothing to attach to")

    cohort = load_cohort(data_dir, task, duration, task_dir, stays_csv=stays_csv)
    units = sorted(cohort["FIRST_CAREUNIT"].dropna().unique())
    careunit_codes = {unit: index for index, unit in enumerate(units)}

    with h5py.File(target, "a") as handle:
        if group not in handle:
            raise SystemExit(f"{target} has no group {group!r}")
        root = handle[group]
        for split in SPLIT_NAMES:
            if split not in root:
                raise SystemExit(f"{target} has no group {group}/{split}")
            selected = _split_rows(cohort, split)
            expected_labels = np.stack(selected["LABEL"].to_numpy())
            actual_labels = np.asarray(root[split]["label"])
            if expected_labels.shape != actual_labels.shape or not np.array_equal(
                expected_labels.astype(actual_labels.dtype, copy=False), actual_labels
            ):
                raise SystemExit(
                    f"refusing to attach: {split} label order does not match the population/partition mapping"
                )
            arrays = {
                "subject_id": selected["SUBJECT_ID"].to_numpy().astype(np.int64),
                "careunit_code": selected["FIRST_CAREUNIT"].map(careunit_codes).to_numpy().astype(np.int64),
                "notes_available": selected["notes_available"].to_numpy().astype(bool),
            }
            for name, values in arrays.items():
                if name in root[split]:
                    if not np.array_equal(np.asarray(root[split][name]), values):
                        raise SystemExit(f"existing {split}/{name} values do not match")
                    print(f"  {split}/{name}: already present and matching", flush=True)
                else:
                    root[split].create_dataset(name, data=values)
                    print(f"  {split}/{name}: attached", flush=True)
        root.attrs.setdefault("careunit_codes", json.dumps(careunit_codes))
        root.attrs.setdefault("observation_hours", int(root["train"]["X"].shape[1]))
        _write_label_metadata(root, _label_metadata(data_dir, task, duration))

    leaks = _subject_leaks(target, group)
    if leaks:
        raise SystemExit(f"subject leakage across splits after attach: {leaks}")
    print(f"patched {target}, no subject leakage", flush=True)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="FIDDLE working directory")
    parser.add_argument("--task", default="Diagnoses")
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--timestep", type=float, default=1.0)
    parser.add_argument("--group", default="with_notes")
    parser.add_argument("--chunk-rows", type=int, default=512)
    parser.add_argument(
        "--compression",
        default=None,
        help="h5py filter for the bulk arrays, e.g. gzip; unset keeps random reads fastest",
    )
    parser.add_argument("--output", type=Path, default=None, help="assembly output path")
    parser.add_argument(
        "--stays-csv",
        type=Path,
        default=None,
        help="cohort table with SUBJECT_ID/FIRST_CAREUNIT/partition columns "
        "(default: prep/icustays_MV.csv; phenotyping uses prep/benchmark_icustays.csv)",
    )
    parser.add_argument(
        "--attach-only",
        action="store_true",
        help="patch an existing splits.hdf5 with subject_id instead of rebuilding",
    )
    args = parser.parse_args()

    if args.attach_only:
        attach_arrays(
            args.data_dir,
            args.task,
            args.duration,
            args.timestep,
            group=args.group,
            splits_path=args.output,
            stays_csv=args.stays_csv,
        )
    else:
        assemble_splits(
            args.data_dir,
            args.task,
            args.duration,
            args.timestep,
            group=args.group,
            chunk_rows=args.chunk_rows,
            compression=args.compression,
            output=args.output,
            stays_csv=args.stays_csv,
        )


if __name__ == "__main__":
    main()
