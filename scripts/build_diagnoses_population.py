#!/usr/bin/env python3
"""Build Yang-Wu-style Diagnoses_48h population + ICD-9 top-3 multi-hot labels.

This is NOT published by FIDDLE-experiments. It follows the EMNLP 2021 protocol:
MetaVision ICU stays with LOS >= 48h, notes in [0, 48], ICD-9 three-digit groups.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


def icd9_top3(code: str) -> str | None:
    if not isinstance(code, str) or not code:
        return None
    code = code.strip().upper()
    if not code:
        return None
    # keep leading V/E then digits; take first 3 alphanumeric diagnosis group chars
    if code[0] in {"V", "E"}:
        body = code[1:]
        digits = "".join(ch for ch in body if ch.isdigit())
        if len(digits) < 2:
            return None
        return code[0] + digits[:2] if code[0] == "V" else code[0] + digits[:3]
    digits = "".join(ch for ch in code if ch.isdigit())
    if len(digits) < 3:
        return None
    return digits[:3]


def earliest_note_offset(mimic_dir: Path, stays: pd.DataFrame, chunksize: int) -> pd.Series:
    """Hours from ICU admission to each stay's first non-negative note.

    Streamed because NOTEEVENTS is 4 GB on disk and several times that in memory;
    a stay has a note inside [0, T] exactly when this offset is at most T, so one
    pass answers the question for every candidate window.
    """
    keys = stays[["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "INTIME"]]
    earliest: dict[int, float] = {}
    scanned = 0
    for chunk in pd.read_csv(
        mimic_dir / "NOTEEVENTS.csv",
        usecols=["SUBJECT_ID", "HADM_ID", "CHARTTIME", "ISERROR"],
        parse_dates=["CHARTTIME"],
        chunksize=chunksize,
    ):
        scanned += len(chunk)
        chunk = chunk[chunk["ISERROR"].isnull() & chunk["CHARTTIME"].notnull()]
        if chunk.empty:
            continue
        merged = keys.merge(chunk.drop(columns="ISERROR"), on=["SUBJECT_ID", "HADM_ID"], how="inner")
        if merged.empty:
            continue
        offset = (merged["CHARTTIME"] - merged["INTIME"]).dt.total_seconds() / 3600.0
        merged = merged.assign(offset=offset)
        merged = merged[merged["offset"] >= 0.0]
        for stay, value in merged.groupby("ICUSTAY_ID")["offset"].min().items():
            previous = earliest.get(stay)
            if previous is None or value < previous:
                earliest[stay] = float(value)
        print(f"  scanned {scanned:,} note rows, stays touched={len(earliest):,}", flush=True)
    return pd.Series(earliest, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mimic-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=48.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument(
        "--require-notes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "keep only stays with a note inside the window. Dropping the requirement "
            "admits patients who have no note yet, which at a 12h window is 27%% of the "
            "cohort and is genuine modality missingness rather than simulated"
        ),
    )
    args = parser.parse_args()

    mimic = args.mimic_dir
    data = args.data_dir
    pop_dir = data / "population"
    pop_dir.mkdir(parents=True, exist_ok=True)

    icus = pd.read_csv(data / "prep" / "icustays_MV.csv", parse_dates=["INTIME", "OUTTIME"])
    print("MetaVision stays:", len(icus))

    # LOS >= observation window (hours)
    los_h = (icus["OUTTIME"] - icus["INTIME"]).dt.total_seconds() / 3600.0
    icus = icus.loc[los_h >= args.duration].copy()
    print(f"LOS>={args.duration}h:", len(icus))

    # Notes in [0, T]
    first_note = earliest_note_offset(mimic, icus, args.chunksize)
    offsets = icus["ICUSTAY_ID"].map(first_note)
    has_notes = offsets.notna() & (offsets <= args.duration)
    print(f"with notes in window: {int(has_notes.sum())} / {len(icus)}")
    if args.require_notes:
        icus = icus[has_notes].copy()
    else:
        icus = icus.copy()
        print(f"keeping {int((~has_notes).sum())} stays whose note has not been charted yet")

    # ICD-9 diagnosis groups for the hospital admission
    dx = pd.read_csv(mimic / "DIAGNOSES_ICD.csv", usecols=["HADM_ID", "ICD9_CODE"])
    dx["group"] = dx["ICD9_CODE"].astype(str).map(icd9_top3)
    dx = dx.dropna(subset=["group"])
    # restrict to cohort admissions
    dx = dx[dx["HADM_ID"].isin(set(icus["HADM_ID"]))]
    # keep groups appearing in at least 1 stay; Yang-Wu reports 1042 groups after their filtering
    group_counts = Counter(dx["group"])
    # take top frequency groups until we approach paper size if needed; default keep all then trim
    all_groups = sorted(group_counts)
    print("raw ICD9 top3 groups:", len(all_groups))

    # Prefer exactly 1042 most frequent groups if more exist (paper uses 1042)
    if len(all_groups) > 1042:
        keep = {g for g, _ in group_counts.most_common(1042)}
    else:
        keep = set(all_groups)
    groups = sorted(keep)
    group_index = {g: i for i, g in enumerate(groups)}
    print("kept groups:", len(groups))

    dx = dx[dx["group"].isin(keep)]
    label_rows = []
    for hadm, sub in dx.groupby("HADM_ID"):
        vec = np.zeros(len(groups), dtype=np.uint8)
        for g in set(sub["group"]):
            vec[group_index[g]] = 1
        label_rows.append((hadm, vec))
    label_map = dict(label_rows)

    records = []
    labels = []
    for row in icus.itertuples(index=False):
        vec = label_map.get(row.HADM_ID)
        if vec is None or vec.sum() == 0:
            continue
        records.append(row.ICUSTAY_ID)
        labels.append(vec)
    labels_arr = np.stack(labels)
    print("final cohort:", len(records), "label_dim:", labels_arr.shape[1], "pos_rate:", float(labels_arr.mean()))

    pop = pd.DataFrame({"ID": records, f"Diagnoses_LABEL": list(labels_arr)})
    # FIDDLE population CSV expects ID + LABEL columns for binary tasks; for multi-label store hdf5
    csv_path = pop_dir / f"Diagnoses_{args.duration}h.csv"
    # keep a lightweight CSV of IDs for FIDDLE population filter
    pd.DataFrame({"ID": records}).to_csv(csv_path, index=False)
    print("wrote", csv_path)

    hdf_path = pop_dir / "population.hdf5"
    # Yang-Wu expects columns ID + Diagnoses_LABEL (object/array)
    df_hdf = pd.DataFrame({"ID": records, "Diagnoses_LABEL": list(labels_arr)})
    key = f"Diagnoses_{args.duration}h"
    df_hdf.to_hdf(hdf_path, key=key, mode="a", format="fixed")
    print("wrote", hdf_path, key)

    meta = {
        "n": len(records),
        "label_dim": int(labels_arr.shape[1]),
        "groups": groups,
        "duration": args.duration,
    }
    import json

    # Scoped by duration so building a second observation window cannot silently
    # overwrite the group vocabulary an already-materialised cohort depends on.
    meta_path = pop_dir / f"Diagnoses_{args.duration}h_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("wrote", meta_path)
    print("done")


if __name__ == "__main__":
    main()
