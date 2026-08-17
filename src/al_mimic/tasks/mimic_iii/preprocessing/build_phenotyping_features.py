#!/usr/bin/env python3
"""Build the 76-channel phenotyping tensors from raw MIMIC-III event tables.

Stage two of the first-party phenotyping pipeline: streams CHARTEVENTS,
LABEVENTS and OUTPUTEVENTS once, cleans the 17 benchmark variables, discretises
each stay into hourly 76-channel frames (previous-value imputation, one-hot
categoricals, observed masks), z-scores the continuous columns with
train-split statistics, and writes the artifacts ``build_splits`` consumes:

    features/outcome=Phenotyping,T={duration},dt={timestep}/
        Xs.hdf5    X [N, max_steps, 76] float32, s [N, 0], step_mask [N, max_steps]
        IDs.csv    stay order of the X rows
    population/population.hdf5   key Phenotyping_{duration}h, ID + Phenotyping_LABEL
    population/Phenotyping_{duration}h_meta.json

A stay survives into the final cohort when it has at least one measurement
inside [0, LOS] -- the benchmark's "no events in ICU" exclusion. With
``--require-notes`` the stay must additionally carry an extracted note, the
notes-benchmark cohort rule used by the 239-label task.

The event-to-stay attribution follows the executed benchmark path: an event
belongs to a cohort stay when its ICUSTAY_ID matches, or when the event was
charted inside the stay's [INTIME, OUTTIME] window, and only events in
[0, LOS] hours survive.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from .benchmark_labels import (
    benchmark_25_labels,
    ccs_239_labels,
    load_ccs_definitions,
)
from .clean_events import clean_events, load_itemid_map
from .discretize import BenchmarkDiscretizer, BenchmarkNormalizer

EPS = 1e-6
TASK_IDs = ("phenotyping_25", "phenotyping_ccs_239")
OUTCOME_NAME = "Phenotyping"
EVENT_TABLES = ("CHARTEVENTS", "LABEVENTS", "OUTPUTEVENTS")


def _read_notes_stays(path: Path) -> set[int]:
    with h5py.File(path, "r") as handle:
        if "ICUSTAY_ID" not in handle:
            raise SystemExit(f"{path} does not carry an ICUSTAY_ID array")
        return set(np.asarray(handle["ICUSTAY_ID"], dtype=np.int64).tolist())


def _stay_windows(stays: pd.DataFrame) -> dict[int, list[tuple]]:
    """subject -> [(icustay, intime, outtime, los_hours)] for cohort stays."""
    windows: dict[int, list[tuple]] = {}
    for subject, intime, outtime, los, stay in zip(
        stays["SUBJECT_ID"].to_numpy(),
        stays["INTIME"].to_numpy(),
        stays["OUTTIME"].to_numpy(),
        stays["LOS_HOURS"].to_numpy(),
        stays["ICUSTAY_ID"].to_numpy(),
        strict=True,
    ):
        windows.setdefault(int(subject), []).append((int(stay), intime, outtime, float(los)))
    return windows


def accumulate_events(
    mimic_dir: Path,
    stays: pd.DataFrame,
    var_map: pd.DataFrame,
    *,
    chunksize: int,
    progress_every: int = 20,
) -> dict[int, dict[str, dict[float, float | str]]]:
    """One streaming pass over the three event tables into per-stay event maps."""
    windows = _stay_windows(stays)
    subjects = set(windows)
    cohort_stays = set(stays["ICUSTAY_ID"].astype(int))
    itemids = set(var_map["ITEMID"].astype(int))
    events: dict[int, dict[str, dict[float, float | str]]] = {stay: {} for stay in cohort_stays}

    for table in EVENT_TABLES:
        path = mimic_dir / f"{table}.csv"
        print(f"streaming {path.name}...", flush=True)
        usecols = ["SUBJECT_ID", "CHARTTIME", "ITEMID", "VALUE", "VALUEUOM"]
        if table != "LABEVENTS":
            usecols = ["SUBJECT_ID", "ICUSTAY_ID", *usecols[1:]]
        scanned = 0
        chunks = pd.read_csv(
            path,
            usecols=usecols,
            dtype={"VALUE": str, "VALUEUOM": str},
            chunksize=chunksize,
        )
        for number, chunk in enumerate(chunks):
            scanned += len(chunk)
            chunk = chunk[chunk["SUBJECT_ID"].isin(subjects) & chunk["ITEMID"].isin(itemids)]
            if chunk.empty:
                continue
            chunk = chunk.copy()
            chunk["ITEMID"] = chunk["ITEMID"].astype(int)
            if "ICUSTAY_ID" not in chunk:
                chunk["ICUSTAY_ID"] = np.nan
            chunk = chunk[chunk["CHARTTIME"].notna()]
            if chunk.empty:
                continue
            chunk["CHARTTIME"] = pd.to_datetime(chunk["CHARTTIME"])
            chunk = clean_events(chunk, var_map)
            if chunk.empty:
                continue
            _attribute_events(chunk, windows, events)
            if number % progress_every == 0:
                print(f"  scanned {scanned:,} rows, kept stays touched={len(events):,}", flush=True)
        print(f"  {table}: scanned {scanned:,} rows", flush=True)
    return events


def _attribute_events(
    chunk: pd.DataFrame,
    windows: dict[int, list[tuple]],
    events: dict[int, dict[str, dict[float, float | str]]],
) -> None:
    for subject, icustay, charttime, variable, value in zip(
        chunk["SUBJECT_ID"].to_numpy(),
        chunk["ICUSTAY_ID"].to_numpy(),
        chunk["CHARTTIME"].to_numpy(),
        chunk["VARIABLE"].to_numpy(),
        chunk["VALUE"].to_numpy(),
        strict=True,
    ):
        for stay, intime, outtime, los in windows[int(subject)]:
            matches = (
                not pd.isna(icustay) and int(icustay) == stay
            ) or (intime <= charttime <= outtime)
            if not matches:
                continue
            hours = (charttime - intime) / np.timedelta64(1, "s") / 3600.0
            if not -EPS < hours < los + EPS:
                continue
            slot = events[stay].setdefault(str(variable), {})
            previous = slot.get(hours)
            if previous is None or not _keeps_max(previous, value):
                slot[hours] = value


def _keeps_max(previous: float | str, candidate: float | str) -> bool:
    try:
        return float(previous) >= float(candidate)
    except (TypeError, ValueError):
        return str(previous) >= str(candidate)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimic-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--task-id", choices=TASK_IDs, required=True)
    parser.add_argument("--duration", type=float, default=256.0, help="nominal window used in artifact paths")
    parser.add_argument("--timestep", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=256)
    parser.add_argument(
        "--require-notes",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="keep only stays with an extracted note (the notes-benchmark cohort rule)",
    )
    parser.add_argument("--chunksize", type=int, default=5_000_000)
    parser.add_argument("--limit-stays", type=int, default=None)
    args = parser.parse_args()
    if args.require_notes is None:
        args.require_notes = args.task_id == "phenotyping_ccs_239"

    prep = args.data_dir / "prep"
    stays = pd.read_csv(prep / "benchmark_icustays.csv", parse_dates=["INTIME", "OUTTIME"])
    task_dir = args.data_dir / f"features/outcome={OUTCOME_NAME},T={args.duration},dt={args.timestep}"
    population_path = args.data_dir / "population/population.hdf5"
    key = f"{OUTCOME_NAME}_{args.duration}h"
    meta_path = args.data_dir / "population" / f"{key}_meta.json"
    for target in (task_dir / "Xs.hdf5", task_dir / "IDs.csv", meta_path):
        if target.exists():
            raise SystemExit(f"{target} already exists; remove it or rebuild into a fresh working directory")
    if population_path.is_file():
        with pd.HDFStore(population_path, mode="r") as store:
            if f"/{key}" in store.keys():
                raise SystemExit(
                    f"{population_path} already contains {key}; remove that key or "
                    "use a fresh working directory"
                )

    if args.limit_stays is not None:
        stays = stays.iloc[: args.limit_stays].copy()
    if args.require_notes:
        notes_path = task_dir / "notes.hdf5"
        if not notes_path.is_file():
            raise SystemExit(
                f"{notes_path} is missing; run extract_notes --protocol all_stay_chronological first"
            )
        with_notes = _read_notes_stays(notes_path)
        stays = stays[stays["ICUSTAY_ID"].isin(with_notes)].copy()
        print(f"stays with an extracted note: {len(stays):,}", flush=True)

    events = accumulate_events(args.mimic_dir, stays, load_itemid_map(), chunksize=args.chunksize)
    populated = [int(stay) for stay in sorted(events) if events[stay]]
    print(
        f"stays with measurements in [0, LOS]: {len(populated):,} / {len(stays):,}",
        flush=True,
    )
    stays = stays[stays["ICUSTAY_ID"].isin(populated)].sort_values("ICUSTAY_ID").reset_index(drop=True)

    discretizer = BenchmarkDiscretizer(timestep=args.timestep, max_steps=args.max_steps)
    normalizer = BenchmarkNormalizer(discretizer.layout.continuous_columns)
    windows = {
        int(row.ICUSTAY_ID): (row.INTIME, row.OUTTIME, float(row.LOS_HOURS), row.partition)
        for row in stays.itertuples(index=False)
    }
    for stay, (_, _, los, partition) in windows.items():
        if partition != "train":
            continue
        frame, _ = discretizer.transform(events[stay], los, max_steps=discretizer.bin_count(los))
        normalizer.feed(frame)
    normalizer.finalize()
    print(
        "normalizer fitted on train rows:",
        f"{normalizer.state()['rows']:,} steps over "
        f"{sum(1 for v in windows.values() if v[3] == 'train'):,} stays",
        flush=True,
    )

    diagnoses = pd.read_csv(prep / "all_diagnoses.csv", dtype={"ICD9_CODE": str})
    if args.task_id == "phenotyping_25":
        labels, label_names = benchmark_25_labels(stays, diagnoses, load_ccs_definitions())
    else:
        labels, label_names = ccs_239_labels(stays, diagnoses, load_ccs_definitions())

    count = len(stays)
    task_dir.mkdir(parents=True, exist_ok=True)
    (args.data_dir / "population").mkdir(parents=True, exist_ok=True)
    with h5py.File(task_dir / "Xs.hdf5", "w") as handle:
        x = handle.create_dataset("X", shape=(count, args.max_steps, 76), dtype=np.float32)
        static = handle.create_dataset("s", shape=(count, 0), dtype=np.float32)
        step_mask = handle.create_dataset("step_mask", shape=(count, args.max_steps), dtype=bool)
        for row, stay in enumerate(stays["ICUSTAY_ID"].to_numpy()):
            frame, mask = discretizer.transform(events[int(stay)], windows[int(stay)][2])
            normalized = normalizer.transform(frame)
            padded = np.zeros((args.max_steps, 76), dtype=np.float32)
            padded[: len(normalized)] = normalized
            padded_mask = np.zeros(args.max_steps, dtype=bool)
            padded_mask[: len(mask)] = mask
            x[row] = padded
            step_mask[row] = padded_mask
            static[row] = np.empty((0,), dtype=np.float32)
            if row % 2000 == 0:
                print(f"  discretised {row:,}/{count:,}", flush=True)
    pd.DataFrame({"ID": stays["ICUSTAY_ID"].to_numpy()}).to_csv(task_dir / "IDs.csv", index=False)

    pd.DataFrame(
        {"ID": stays["ICUSTAY_ID"].to_numpy(), f"{OUTCOME_NAME}_LABEL": list(labels)}
    ).to_hdf(population_path, key=key, mode="a", format="fixed")

    positives = labels.sum(axis=0)
    meta = {
        "task_id": args.task_id,
        "outcome": OUTCOME_NAME,
        "duration": args.duration,
        "timestep": args.timestep,
        "max_steps": args.max_steps,
        "n": int(count),
        "label_names": list(label_names),
        "positive_counts": {name: int(positives[position]) for position, name in enumerate(label_names)},
        "split_counts": {
            str(partition): int(count)
            for partition, count in stays["partition"].value_counts().items()
        },
        "require_notes": bool(args.require_notes),
        "normalizer": normalizer.state(),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {task_dir / 'Xs.hdf5'} ({(task_dir / 'Xs.hdf5').stat().st_size / 2**30:.2f} GiB), "
        f"{population_path} [{key}], labels={labels.shape[1]}",
        flush=True,
    )


if __name__ == "__main__":
    main()
