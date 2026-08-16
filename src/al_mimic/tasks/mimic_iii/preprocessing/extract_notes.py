#!/usr/bin/env python
"""Build notes.hdf5 for a cohort without loading NOTEEVENTS into memory.

Reproduces the Yang-Wu note protocol -- keep the latest note per
(ICU stay, CATEGORY/DESCRIPTION), join a stay's notes with newlines in sorted
variable order, then tokenize to a fixed 512-token window -- but streams the
4 GB CSV in chunks instead of reading it whole. MimicDataModule.note_feats()
materialises every note column at once, which needs tens of gigabytes; this
keeps only the notes that survive the per-variable latest-wins rule.

Output is the same frame MimicDataModule.split() reads: one row per stay with
ICUSTAY_ID, input_ids, token_type_ids and attention_mask.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import numpy as np
import pandas as pd

NOTE_COLUMNS = ["SUBJECT_ID", "HADM_ID", "CHARTTIME", "ISERROR", "CATEGORY", "DESCRIPTION", "TEXT"]


def clean(text: str) -> str:
    """The Yang-Wu preprocessing chain, applied per note rather than per column."""
    text = text.replace("\n", " ").replace("\r", " ").strip().lower()
    text = re.sub("\\[(.*?)]", "", text)
    text = re.sub("[0-9]+\\.", "", text)
    text = re.sub("dr\\.", "doctor", text)
    text = re.sub("m\\.d\\.", "md", text)
    text = re.sub("admission date:", "", text)
    text = re.sub("discharge date:", "", text)
    text = re.sub("--|__|==", "", text)
    return text


def collect_latest_notes(
    mimic_dir: Path,
    stays: pd.DataFrame,
    duration: float,
    chunksize: int,
    max_chars: int | None,
) -> dict[int, dict[str, tuple[float, int, str]]]:
    """Latest note per (stay, variable), streamed over NOTEEVENTS."""
    keys = stays[["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "INTIME"]]
    kept: dict[int, dict[str, tuple[float, int, str]]] = {}
    scanned = 0
    for chunk in pd.read_csv(
        mimic_dir / "NOTEEVENTS.csv",
        usecols=NOTE_COLUMNS,
        parse_dates=["CHARTTIME"],
        chunksize=chunksize,
    ):
        scanned += len(chunk)
        chunk = chunk[chunk["ISERROR"].isnull() & chunk["CHARTTIME"].notnull()]
        if chunk.empty:
            continue
        # pandas keeps a global row index across chunks, so ties between notes
        # charted at the same minute resolve to source order on every run.
        chunk = chunk.assign(_row=chunk.index)
        merged = keys.merge(chunk, on=["SUBJECT_ID", "HADM_ID"], how="inner")
        if merged.empty:
            continue
        time = (merged["CHARTTIME"] - merged["INTIME"]).dt.total_seconds() / 3600.0
        merged = merged.assign(TIME=time)
        merged = merged[(merged["TIME"] >= 0.0) & (merged["TIME"] <= duration)]
        if merged.empty:
            continue
        variable = merged["CATEGORY"].astype(str) + "/" + merged["DESCRIPTION"].astype(str)
        merged = merged.assign(VARNAME=variable)
        for stay, name, moment, row, text in zip(
            merged["ICUSTAY_ID"].to_numpy(),
            merged["VARNAME"].to_numpy(),
            merged["TIME"].to_numpy(),
            merged["_row"].to_numpy(),
            merged["TEXT"].fillna(" ").to_numpy(),
            strict=True,
        ):
            slot = kept.setdefault(int(stay), {})
            previous = slot.get(name)
            if previous is not None and (previous[0], -previous[1]) >= (moment, -int(row)):
                continue
            body = clean(str(text))
            slot[name] = (float(moment), int(row), body if max_chars is None else body[:max_chars])
        print(f"  scanned {scanned:,} note rows, stays with notes={len(kept):,}", flush=True)
    return kept


def join_texts(kept: dict[int, dict[str, tuple[float, int, str]]]) -> pd.DataFrame:
    """One text per stay, variables joined in sorted order as the protocol requires."""
    rows = [(stay, "\n".join(slot[name][2] for name in sorted(slot))) for stay, slot in sorted(kept.items())]
    return pd.DataFrame(rows, columns=["ICUSTAY_ID", "TEXT"])


def tokenize_to_hdf5(
    frame: pd.DataFrame,
    output: Path,
    vocab_dir: Path,
    max_tokens: int,
    batch_size: int,
) -> None:
    """Tokenize and stream straight to disk, so peak memory does not track cohort size.

    Holding the token tensors in memory costs three arrays of N x 512 on top of
    the joined text and the tokenizer's own import of torch. Writing each batch
    into preallocated HDF5 datasets keeps the footprint flat instead.

    Stored as int32: the largest id in the ClinicalBERT vocabulary is 28995, and
    the loader casts to int64 on read anyway, so this halves both memory and file
    size at no cost.
    """
    import h5py
    from transformers import BertTokenizer

    vocab_dir = Path(vocab_dir).expanduser().resolve()
    if not (vocab_dir / "vocab.txt").is_file():
        raise FileNotFoundError(f"ClinicalBERT tokenizer artifact must contain vocab.txt: {vocab_dir}")
    tokenizer = BertTokenizer.from_pretrained(vocab_dir, local_files_only=True)
    tokenizer.model_max_length = max_tokens
    print(f"tokenizer: {vocab_dir} (vocab {tokenizer.vocab_size})", flush=True)

    total = len(frame)
    texts = frame["TEXT"]
    with h5py.File(output, "w") as handle:
        handle.create_dataset("ICUSTAY_ID", data=frame["ICUSTAY_ID"].to_numpy().astype(np.int64))
        blocks = {
            name: handle.create_dataset(name, shape=(total, max_tokens), dtype=np.int32)
            for name in ("input_ids", "token_type_ids", "attention_mask")
        }
        for start in range(0, total, batch_size):
            stop = min(start + batch_size, total)
            encoded = tokenizer(
                texts.iloc[start:stop].tolist(),
                padding="max_length",
                truncation=True,
                max_length=max_tokens,
                return_token_type_ids=True,
                return_attention_mask=True,
            )
            for name, block in blocks.items():
                block[start:stop] = np.asarray(encoded[name], dtype=np.int32)
            if (start // batch_size) % 25 == 0:
                print(f"  tokenized {stop:,}/{total:,}", flush=True)
        largest = int(handle["input_ids"][:, :].max()) if total else 0
    if largest >= 28996:
        raise SystemExit(f"token id {largest} exceeds the ClinicalBERT vocabulary")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mimic-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--task", default="Diagnoses")
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--timestep", type=float, default=1.0)
    parser.add_argument(
        "--vocab-dir",
        type=Path,
        required=True,
        help="first-party materialized ClinicalBERT tokenizer artifact directory",
    )
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--chunksize", type=int, default=50_000)
    parser.add_argument("--tokenize-batch", type=int, default=64)
    parser.add_argument(
        "--max-chars",
        type=int,
        default=None,
        help="truncate each note; only affects text beyond the token window",
    )
    parser.add_argument(
        "--limit-stays",
        type=int,
        default=None,
        help="restrict the cohort, for validating the pipeline on a constrained host",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    task_dir = args.data_dir / f"features/outcome={args.task},T={args.duration},dt={args.timestep}"
    task_dir.mkdir(parents=True, exist_ok=True)
    output = args.output or task_dir / "notes.hdf5"

    population = pd.read_hdf(args.data_dir / "population/population.hdf5", f"{args.task}_{args.duration}h")
    cohort = population["ID"].astype(np.int64)
    if args.limit_stays is not None:
        cohort = cohort.iloc[: args.limit_stays]
    print(f"cohort stays: {len(cohort):,}", flush=True)

    icus = pd.read_csv(args.mimic_dir / "ICUSTAYS.csv", parse_dates=["INTIME"])
    stays = icus[icus["ICUSTAY_ID"].isin(set(cohort))][["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "INTIME"]]
    print(f"resolved INTIME for {len(stays):,} stays", flush=True)

    kept = collect_latest_notes(args.mimic_dir, stays, args.duration, args.chunksize, args.max_chars)
    per_stay = np.array([len(slot) for slot in kept.values()])
    print(
        f"stays with >=1 note: {len(kept):,} / {len(stays):,}; "
        f"notes per stay mean={per_stay.mean():.1f} max={per_stay.max()}",
        flush=True,
    )

    frame = join_texts(kept)
    kept.clear()
    lengths = frame["TEXT"].str.len()
    print(
        f"joined text chars: median={int(lengths.median()):,} p95={int(lengths.quantile(0.95)):,} "
        f"max={int(lengths.max()):,}",
        flush=True,
    )

    tokenize_to_hdf5(frame, output, args.vocab_dir, args.max_tokens, args.tokenize_batch)
    print(
        f"wrote {output} rows={len(frame):,} ({output.stat().st_size / 2**20:.0f} MiB)",
        flush=True,
    )


if __name__ == "__main__":
    main()
