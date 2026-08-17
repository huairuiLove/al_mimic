"""Clinical-note artifact I/O shared by MIMIC-III preprocessing steps."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

TOKEN_FIELDS = ("input_ids", "token_type_ids", "attention_mask")


def read_notes(path: str | Path) -> pd.DataFrame:
    """Read streamed HDF5 tensors or the legacy pandas ``notes`` frame."""
    artifact = Path(path)
    with h5py.File(artifact, "r") as handle:
        streamed = "ICUSTAY_ID" in handle and all(field in handle for field in TOKEN_FIELDS)
        if streamed:
            frame = pd.DataFrame({"ICUSTAY_ID": np.asarray(handle["ICUSTAY_ID"], dtype=np.int64)})
            for field in TOKEN_FIELDS:
                frame[field] = list(np.asarray(handle[field], dtype=np.int64))
            return frame
    return pd.read_hdf(artifact, "notes")


__all__ = ["TOKEN_FIELDS", "read_notes"]
