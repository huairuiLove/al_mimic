#!/usr/bin/env python
"""Assemble Yang-Wu splits.hdf5/with_notes from FIDDLE X/s + MIMIC notes."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np


def attach_subject_ids(root: Path, feat: Path) -> None:
    """Upgrade older Yang-Wu splits after verifying their exact row order."""
    import pandas as pd

    labels = pd.read_hdf(
        root / "data/fiddle_processed/population/population.hdf5",
        "Diagnoses_48.0h",
    ).rename(columns={"ID": "ICUSTAY_ID", "Diagnoses_LABEL": "LABEL"})
    notes = pd.read_hdf(feat / "notes.hdf5", "notes")
    subjects = pd.read_csv(
        root / "data/fiddle_processed/prep/icustays_MV.csv",
        usecols=["ICUSTAY_ID", "SUBJECT_ID", "partition"],
    )
    rows = labels.merge(notes, on="ICUSTAY_ID", how="left")
    rows = rows[rows["input_ids"].notnull()].reset_index(drop=True)
    rows = rows.reset_index().merge(subjects, on="ICUSTAY_ID", how="left").set_index("index")

    split_path = feat / "splits.hdf5"
    with h5py.File(split_path, "a") as handle:
        group = handle["with_notes"]
        for split in ("train", "val", "test"):
            split_rows = rows.loc[rows["partition"] == split]
            expected_labels = np.stack(split_rows["LABEL"].to_numpy())
            actual_labels = np.asarray(group[split]["label"])
            if not np.array_equal(expected_labels, actual_labels):
                raise RuntimeError(
                    f"refusing to attach SUBJECT_ID: {split} label order does not match"
                )
            subject_ids = split_rows["SUBJECT_ID"].to_numpy(dtype=np.int64)
            if "SUBJECT_ID" in group[split]:
                if not np.array_equal(np.asarray(group[split]["SUBJECT_ID"]), subject_ids):
                    raise RuntimeError(f"existing {split}/SUBJECT_ID values do not match")
            else:
                group[split].create_dataset("SUBJECT_ID", data=subject_ids)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="repository root containing data/ and third_party/",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    import sparse

    sys.path.insert(0, str(root / "third_party" / "yang_wu_mimic"))

    feat = root / "data/fiddle_processed/features/outcome=Diagnoses,T=48.0,dt=1.0"
    s_link = feat / "s.npz"
    if not s_link.exists():
        s_link.symlink_to("S.npz")

    # Build Xs.hdf5 in float32 to stay under disk/cgroup pressure
    xs_path = feat / "Xs.hdf5"
    if not xs_path.is_file():
        print("Building Xs.hdf5 (float32)...", flush=True)
        X = sparse.load_npz(feat / "X.npz").todense().astype(np.float32, copy=False)
        s = sparse.load_npz(feat / "s.npz").todense().astype(np.float32, copy=False)
        print("X", X.shape, "s", s.shape, flush=True)
        with h5py.File(xs_path, "w") as hf:
            hf.create_dataset("X", data=X, compression="gzip", compression_opts=1)
            hf.create_dataset("s", data=s, compression="gzip", compression_opts=1)
        del X, s
        print("Wrote", xs_path, "size", xs_path.stat().st_size, flush=True)

    from data_module import MimicDataModule

    # Prefer local ClinicalBERT vocab if present; Yang-Wu tokenize() hardcodes
    # bert-base-uncased — leave that behavior intact for protocol fidelity.
    dm = MimicDataModule(
        mimic_dir=str(root / "mimic-iii-clinical-database-1.4"),
        data_dir=str(root / "data/fiddle_processed"),
        task="Diagnoses",
        duration=48.0,
        timestep=1.0,
        notes=True,
        discrete=False,
        merge=False,
        batch_size=8,
        num_workers=4,
    )
    # Skip xs_hdf5 inside prepare_data (already built as float32)
    print("Preparing notes + splits...", flush=True)
    if not (feat / "notes.hdf5").is_file():
        dm.note_hdf5()
    dm.split_hdf5()
    attach_subject_ids(root, feat)
    print("DONE splits:", feat / "splits.hdf5", flush=True)
    with h5py.File(feat / "splits.hdf5", "r") as hf:
        g = hf["with_notes"]
        for split in ("train", "val", "test"):
            keys = list(g[split].keys())
            n = g[split]["label"].shape[0]
            print(split, "n=", n, "keys=", keys, "X=", g[split]["X"].shape, "label=", g[split]["label"].shape, flush=True)


if __name__ == "__main__":
    main()
