#!/usr/bin/env python3
"""Dry-run cohort sizing for candidate observation windows. Writes nothing.

Mirrors the inclusion logic of build_diagnoses_population.py so the cost of a
FIDDLE re-run at a new duration can be estimated before committing to it.

NOTEEVENTS is streamed in chunks: the agent shell runs under a 2 GB cgroup, so
the 4 GB CSV cannot be loaded whole. A single pass records, per ICU stay, the
earliest non-negative note offset; a stay then has a note inside [0, T] exactly
when that offset is <= T.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from build_diagnoses_population import earliest_note_offset, icd9_top3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mimic-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--durations", type=float, nargs="+", default=[48.0, 24.0, 12.0])
    parser.add_argument("--chunksize", type=int, default=100_000)
    args = parser.parse_args()

    icus = pd.read_csv(
        args.data_dir / "prep" / "icustays_MV.csv", parse_dates=["INTIME", "OUTTIME"]
    )
    icus["los_h"] = (icus["OUTTIME"] - icus["INTIME"]).dt.total_seconds() / 3600.0
    print("MetaVision stays:", len(icus), flush=True)

    print("streaming NOTEEVENTS...", flush=True)
    first_note = earliest_note_offset(args.mimic_dir, icus, args.chunksize)
    print("stays with any non-negative note:", len(first_note), flush=True)

    dx = pd.read_csv(args.mimic_dir / "DIAGNOSES_ICD.csv", usecols=["HADM_ID", "ICD9_CODE"])
    dx["group"] = dx["ICD9_CODE"].astype(str).map(icd9_top3)
    dx = dx.dropna(subset=["group"])

    icus["first_note_h"] = icus["ICUSTAY_ID"].map(first_note)

    rows = []
    for duration in args.durations:
        pool = icus[icus["los_h"] >= duration]
        n_los = len(pool)
        pool = pool[pool["first_note_h"].notna() & (pool["first_note_h"] <= duration)]
        n_notes = len(pool)

        scoped = dx[dx["HADM_ID"].isin(set(pool["HADM_ID"]))]
        counts = Counter(scoped["group"])
        keep = (
            {g for g, _ in counts.most_common(1042)} if len(counts) > 1042 else set(counts)
        )
        scoped = scoped[scoped["group"].isin(keep)]
        per_hadm = scoped.groupby("HADM_ID")["group"].nunique()
        labels = pool["HADM_ID"].map(per_hadm).fillna(0)
        final = pool[labels > 0]
        rows.append(
            {
                "T(h)": duration,
                "LOS>=T": n_los,
                "+notes in [0,T]": n_notes,
                "final cohort": len(final),
                "label groups": len(keep),
                "labels/stay": round(float(final["HADM_ID"].map(per_hadm).mean()), 1),
                "N*L rows": len(final) * int(duration),
            }
        )

    print()
    print(pd.DataFrame(rows).to_string(index=False))
    print("\ncurrent 48h baseline actually built: cohort=10258, N*L=492,384", flush=True)


if __name__ == "__main__":
    main()
