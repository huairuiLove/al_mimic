#!/usr/bin/env python
"""CPU pre-flight for the mixup training path.

Runs one short round on a handful of labelled rows so a crash surfaces in
seconds instead of hours into a queued GPU job. This is a smoke check, not an
experiment: it mutates an in-memory copy of the config and writes nothing.
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mimic_comal.config import load_config
from mimic_comal.multimodal_data import YangWuFeatureStore
from mimic_comal.multimodal_training import train_multimodal_round


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mimic_random_mixup.yaml")
    parser.add_argument("--rows", type=int, default=6)
    args = parser.parse_args()

    config = load_config(args.config)
    print("strategy:", config["active_learning"]["strategy"], "mixup:", config.get("mixup"))

    # Shrink the round so a forward/backward pass fits comfortably on CPU.
    config["training"]["epochs"] = 1
    config["training"]["optimizer_steps_per_round"] = 1
    config["training"]["batch_size"] = args.rows
    config["training"]["num_workers"] = 0
    config["training"]["pin_memory"] = False
    config["training"]["eval_batch_size"] = args.rows

    store = YangWuFeatureStore(config, validate=False)
    labeled = store.indices("train")[: args.rows]
    print("labelled rows:", labeled.tolist())
    positives = store.labels[labeled].sum(axis=1)
    print("positives per row:", positives.tolist())

    trained = train_multimodal_round(store, labeled, config, torch.device("cpu"))
    history = trained.history
    print("classifier_loss:", history["classifier_loss"])
    for key in ("mixup_loss", "mixup_anchor_positive_mean", "mixup_mixed_positive_mean"):
        if key in history:
            print(f"{key}:", history[key])

    if config.get("mixup", {}).get("enabled"):
        anchor = history["mixup_anchor_positive_mean"][-1]
        mixed = history["mixup_mixed_positive_mean"][-1]
        if not np.isfinite(history["mixup_loss"][-1]):
            raise SystemExit("mixup loss is not finite")
        if mixed <= anchor:
            raise SystemExit(
                f"mixup failed to add positive mass: anchor={anchor:.2f} mixed={mixed:.2f}"
            )
        print(f"positive mass {anchor:.2f} -> {mixed:.2f} per mixed anchor")
    print("PREFLIGHT_OK")


if __name__ == "__main__":
    main()
