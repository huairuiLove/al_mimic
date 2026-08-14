#!/usr/bin/env python
"""Assemble splits.hdf5 directly from the FIDDLE sparse tensors.

Differs from MimicDataModule.split() in three ways that matter here:

- No Xs.hdf5 intermediate. That file densifies the whole cohort before the split
  is taken, which cost 15 GB of disk and a full extra pass; rows are streamed
  from the sparse matrices in chunks instead.
- subject_id is written into the artifact. The active-learning loader requires
  it to prove no patient spans two splits, and the 48h build had to have it
  injected afterwards.
- careunit_code is written alongside, giving scenarios a real subgroup axis to
  concentrate modality missingness on.

Row order is asserted against FIDDLE's own IDs.csv, so a population rebuilt in a
different order cannot silently misalign features and labels.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import h5py
import numpy as np
import pandas as pd
import sparse

from mimic_comal.scenarios import empty_note

SPLIT_NAMES = ("train", "val", "test")
TOKEN_FIELDS = ("input_ids", "token_type_ids", "attention_mask")


def read_notes(path: Path) -> pd.DataFrame:
    """Read either note layout: the streamed datasets, or the original pickled frame.

    scripts/extract_notes.py writes plain HDF5 datasets so it can stream to disk;
    the 48h cohort predates that and holds a pandas frame of object columns.
    """
    with h5py.File(path, "r") as handle:
        streamed = "ICUSTAY_ID" in handle and all(field in handle for field in TOKEN_FIELDS)
        if streamed:
            frame = pd.DataFrame({"ICUSTAY_ID": np.asarray(handle["ICUSTAY_ID"], dtype=np.int64)})
            for field in TOKEN_FIELDS:
                frame[field] = list(np.asarray(handle[field], dtype=np.int64))
            return frame
    return pd.read_hdf(path, "notes")


def load_cohort(data_dir: Path, task: str, duration: float, task_dir: Path) -> pd.DataFrame:
    """Population rows in FIDDLE's feature order, with notes, subject and unit attached."""
    population = pd.read_hdf(
        data_dir / "population/population.hdf5", f"{task}_{duration}h"
    ).rename(columns={"ID": "ICUSTAY_ID", f"{task}_LABEL": "LABEL"})
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
    # the modality heterogeneity the experiment is about, not a row to discard.
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
            cohort.loc[gaps, field] = pd.Series(
                [blank[field].copy() for _ in gaps], index=gaps, dtype=object
            )

    stays = pd.read_csv(data_dir / "prep/icustays_MV.csv")[
        ["ICUSTAY_ID", "SUBJECT_ID", "FIRST_CAREUNIT", "partition"]
    ]
    cohort = cohort.merge(stays, on="ICUSTAY_ID", how="left")
    missing = cohort["partition"].isna().sum()
    if missing:
        raise SystemExit(f"{missing} cohort stays have no split assignment in icustays_MV.csv")
    return cohort


def write_split(
    group: h5py.Group,
    rows: pd.DataFrame,
    time_series,
    static: np.ndarray,
    shape: tuple[int, int],
    careunit_codes: dict[str, int],
    chunk_rows: int,
    compression: str | None,
) -> None:
    count = len(rows)
    steps, features = shape
    options = {"compression": compression} if compression else {}

    x = group.create_dataset("X", shape=(count, steps, features), dtype=np.float32, **options)
    s = group.create_dataset("s", shape=(count, static.shape[1]), dtype=np.float32, **options)
    feature_rows = rows["feature_row"].to_numpy()

    for start in range(0, count, chunk_rows):
        window = feature_rows[start : start + chunk_rows]
        dense = time_series[window].toarray().astype(np.float32, copy=False)
        x[start : start + len(window)] = dense.reshape(len(window), steps, features)
        s[start : start + len(window)] = static[window]
        print(f"    rows {min(start + chunk_rows, count):,}/{count:,}", flush=True)

    for field in TOKEN_FIELDS:
        group.create_dataset(
            field, data=np.stack(rows[field].to_numpy()).astype(np.int64), **options
        )
    group.create_dataset("label", data=np.stack(rows["LABEL"].to_numpy()).astype(np.int64), **options)
    group.create_dataset("subject_id", data=rows["SUBJECT_ID"].to_numpy().astype(np.int64))
    group.create_dataset(
        "careunit_code",
        data=rows["FIRST_CAREUNIT"].map(careunit_codes).to_numpy().astype(np.int64),
    )
    group.create_dataset("notes_available", data=rows["notes_available"].to_numpy().astype(bool))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
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
    args = parser.parse_args()

    task_dir = (
        args.data_dir / f"features/outcome={args.task},T={args.duration},dt={args.timestep}"
    )
    output = task_dir / "splits.hdf5"
    if output.exists():
        raise SystemExit(f"{output} already exists; remove it to rebuild")

    cohort = load_cohort(args.data_dir, args.task, args.duration, task_dir)
    print(f"cohort with notes: {len(cohort):,}", flush=True)

    print("loading sparse features...", flush=True)
    x_sparse = sparse.load_npz(task_dir / "X.npz")
    s_sparse = sparse.load_npz(task_dir / "S.npz")
    rows, steps, features = x_sparse.shape
    print(f"X {x_sparse.shape} nnz={x_sparse.nnz:,}  s {s_sparse.shape}", flush=True)
    # CSR over the flattened time axis gives cheap row gathers; the dense form of
    # the whole tensor would be tens of gigabytes.
    time_series = x_sparse.reshape((rows, steps * features)).tocsr()
    del x_sparse
    static = s_sparse.todense().astype(np.float32, copy=False)
    del s_sparse

    units = sorted(cohort["FIRST_CAREUNIT"].dropna().unique())
    careunit_codes = {unit: index for index, unit in enumerate(units)}

    with h5py.File(output, "w") as handle:
        root = handle.create_group(args.group)
        root.attrs["careunit_codes"] = json.dumps(careunit_codes)
        root.attrs["observation_hours"] = steps
        for split in SPLIT_NAMES:
            selected = cohort[cohort["partition"] == split].reset_index(drop=True)
            print(f"  {split}: {len(selected):,} stays", flush=True)
            write_split(
                root.create_group(split),
                selected,
                time_series,
                static,
                (steps, features),
                careunit_codes,
                args.chunk_rows,
                args.compression,
            )

    with h5py.File(output, "r") as handle:
        root = handle[args.group]
        for split in SPLIT_NAMES:
            shapes = {name: root[split][name].shape for name in root[split]}
            print(split, shapes, flush=True)
        subjects = {
            split: set(np.asarray(root[split]["subject_id"]).tolist()) for split in SPLIT_NAMES
        }
    leaks = {
        f"{a}/{b}": len(subjects[a] & subjects[b])
        for index, a in enumerate(SPLIT_NAMES)
        for b in SPLIT_NAMES[index + 1 :]
        if subjects[a] & subjects[b]
    }
    if leaks:
        raise SystemExit(f"subject leakage across splits: {leaks}")
    print(f"wrote {output} ({output.stat().st_size / 2**30:.1f} GiB), no subject leakage", flush=True)


if __name__ == "__main__":
    main()
