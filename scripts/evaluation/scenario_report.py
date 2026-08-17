#!/usr/bin/env python
"""Predict whether a scenario leaves room for acquisition to matter, before any GPU time.

Reports, per scenario, the prior-only R@K on the test split and the fraction of
positive mass that survives. A scenario whose prior already sits near the
achievable ceiling cannot separate acquisition strategies no matter which method
is run, so this is the cheap gate in front of the two-arm resolution test.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import numpy as np

from al_mimic.tasks.mimic_iii.config import load_config
from al_mimic.tasks.mimic_iii.data import YangWuFeatureStore
from al_mimic.tasks.mimic_iii.scenarios import empty_note


def prior_recall_at_k(train_labels: np.ndarray, test_labels: np.ndarray, k: int) -> float:
    """Recall@k of the constant ranking induced by train label frequency."""
    evaluable = test_labels.sum(axis=1) > 0
    truth = test_labels[evaluable]
    order = np.argsort(-train_labels.mean(axis=0))[:k]
    return float((truth[:, order].sum(axis=1) / truth.sum(axis=1)).mean())


def report(config_path: Path, k: int) -> dict[str, object]:
    config = load_config(config_path)
    store = YangWuFeatureStore(config, validate=False)
    train = store.indices("train")
    test = store.indices("test")
    labels = store.labels

    row = {
        "config": config_path.name,
        "scenario": store.scenario.name,
        "labels": store.label_count,
        f"prior R@{k}": round(prior_recall_at_k(labels[train], labels[test], k), 4),
        "train rows": int(train.size),
        "positives/visit": round(float(labels[train].sum(axis=1).mean()), 1),
    }
    missing = store.scenario.notes_missing
    if missing is not None:
        row["notes withheld"] = f"{100 * missing[train].mean():.0f}% train"
        withheld_positives = labels[train][missing[train]].sum(axis=1).mean()
        present_positives = labels[train][~missing[train]].sum(axis=1).mean()
        row["withheld vs present positives"] = f"{withheld_positives:.1f} vs {present_positives:.1f}"
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", nargs="+", type=Path)
    parser.add_argument("--k", type=int, default=30)
    args = parser.parse_args()

    note = empty_note(512)
    print(
        f"withheld-note tensors: input_ids[:3]={note['input_ids'][:3].tolist()} "
        f"attention_mask sum={int(note['attention_mask'].sum())}\n",
        flush=True,
    )

    import pandas as pd

    rows = [report(path, args.k) for path in args.configs]
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
