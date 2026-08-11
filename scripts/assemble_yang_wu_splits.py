#!/usr/bin/env python
"""Assemble Yang-Wu splits.hdf5/with_notes from FIDDLE X/s + MIMIC notes."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import h5py
import numpy as np
import sparse


def main() -> None:
    root = Path("/root/autodl-tmp/al_mimic-master")
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
    print("DONE splits:", feat / "splits.hdf5", flush=True)
    with h5py.File(feat / "splits.hdf5", "r") as hf:
        g = hf["with_notes"]
        for split in ("train", "val", "test"):
            keys = list(g[split].keys())
            n = g[split]["label"].shape[0]
            print(split, "n=", n, "keys=", keys, "X=", g[split]["X"].shape, "label=", g[split]["label"].shape, flush=True)


if __name__ == "__main__":
    main()
