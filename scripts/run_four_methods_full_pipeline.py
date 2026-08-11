#!/usr/bin/env python
"""Rebuild notes+splits with ClinicalBERT/512, validate, then run four methods."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
os.environ.setdefault("PYTHONUNBUFFERED", "1")

ROOT = Path("/root/autodl-tmp/al_mimic-master")
FEAT = ROOT / "data/fiddle_processed/features/outcome=Diagnoses,T=48.0,dt=1.0"
DEST = ROOT / "data/yang_wu_mimic/features/outcome=Diagnoses,T=48.0,dt=1.0"


def rebuild_splits() -> None:
    sys.path.insert(0, str(ROOT / "third_party" / "yang_wu_mimic"))
    import numpy as np
    from data_module import MimicDataModule

    DEST.mkdir(parents=True, exist_ok=True)
    for name in ("notes.hdf5", "splits.hdf5"):
        p = FEAT / name
        if p.exists():
            p.unlink()

    dm = MimicDataModule(
        mimic_dir=str(ROOT / "mimic-iii-clinical-database-1.4"),
        data_dir=str(ROOT / "data/fiddle_processed"),
        task="Diagnoses",
        duration=48.0,
        timestep=1.0,
        notes=True,
        discrete=False,
        merge=False,
        batch_size=8,
        num_workers=4,
    )
    print("note_feats...", flush=True)
    notes = dm.note_feats()
    shapes = {tuple(np.asarray(x).shape) for x in notes["input_ids"]}
    mx = max(int(np.asarray(x).max()) for x in notes["input_ids"])
    print("notes", notes.shape, "id_shapes", shapes, "max_id", mx, flush=True)
    if shapes != {(512,)}:
        raise SystemExit(f"expected all token rows shape (512,), got {shapes}")
    if mx >= 28996:
        raise SystemExit(f"token id {mx} exceeds ClinicalBERT vocab")
    notes.to_hdf(str(FEAT / "notes.hdf5"), key="notes", mode="w")
    print("split_hdf5...", flush=True)
    dm.split_hdf5()
    print("SPLITS_DONE", flush=True)

    import h5py

    with h5py.File(FEAT / "splits.hdf5", "r") as hf:
        g = hf["with_notes"]
        for split in ("train", "val", "test"):
            print(split, {k: g[split][k].shape for k in g[split]}, flush=True)

    DEST.mkdir(parents=True, exist_ok=True)
    link = DEST / "splits.hdf5"
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(FEAT / "splits.hdf5")


def main() -> None:
    os.chdir(ROOT)
    rebuild_splits()
    print("=== validate-data ===", flush=True)
    subprocess.check_call(
        [sys.executable, "-u", "main.py", "validate-data", "--config", "configs/mimic_comal.yaml"],
        cwd=ROOT,
    )
    print("=== four methods ===", flush=True)
    subprocess.check_call(["bash", "scripts/run_four_methods.sh", "active"], cwd=ROOT)
    print("FOUR_METHODS_DONE", flush=True)


if __name__ == "__main__":
    main()
