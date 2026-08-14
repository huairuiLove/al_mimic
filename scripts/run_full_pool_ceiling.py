#!/usr/bin/env python
"""Train once on the full train pool of a scenario (local ceiling, single seed).

Uses the same Yang-Wu BertEncoder and training hyper-parameters as the AL arms
but labels every train visit, giving the upper bound a budgeted arm is measured
against. Without it a recall number means nothing: on the official cohort the
ceiling is 0.5418 and a 35% budget already reaches 0.534, so the whole prize is
0.008 and no acquisition strategy can show an effect above test noise.

Takes the config of the arm whose scenario should be measured, so each scenario
gets its own ceiling rather than being compared against the official one.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import numpy as np
import torch

from mimic_comal.config import load_config
from mimic_comal.metrics import multilabel_metrics
from mimic_comal.multimodal_data import YangWuFeatureStore
from mimic_comal.multimodal_training import (
    collect_classifier_outputs,
    train_multimodal_round,
)
from mimic_comal.runtime import configure_runtime


def _host(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().float().cpu().numpy()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/mimic_random.yaml",
        help="arm config whose scenario and training knobs the ceiling should match",
    )
    parser.add_argument("--name", default=None, help="experiment directory name")
    args = parser.parse_args()

    # A random-strategy config keeps the round from building CoMAL/MoDIS/MoSAIC
    # heads that the ceiling never queries with.
    config = load_config(args.config)
    config["experiment"]["name"] = args.name or f"{config['experiment']['name']}_full_pool"
    configure_runtime(config)

    output_dir = (
        Path(config["experiment"]["output_root"]) / config["experiment"]["name"]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)

    device = torch.device(str(config["training"]["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA required for the formal ceiling run")

    store = YangWuFeatureStore(config, validate=True)
    train_indices = store.indices("train")
    val_indices = store.indices("val")
    test_indices = store.indices("test")
    print(
        f"FULL_POOL scenario={store.scenario.name} n_train={train_indices.size} "
        f"n_val={val_indices.size} n_test={test_indices.size} labels={store.label_count}",
        flush=True,
    )
    print("scenario:", json.dumps(store.scenario.summary()), flush=True)

    start = time.perf_counter()
    trained = train_multimodal_round(store, train_indices, config, device)
    print(f"training_sec={trained.timings}", flush=True)

    validation = collect_classifier_outputs(
        trained.classifier, store, val_indices, config, device, return_tokens=False
    )
    test = collect_classifier_outputs(
        trained.classifier, store, test_indices, config, device, return_tokens=False
    )
    validation_metrics = multilabel_metrics(
        _host(validation["labels"]), _host(validation["probabilities"])
    )
    test_metrics = multilabel_metrics(
        _host(test["labels"]), _host(test["probabilities"])
    )
    print("validation", validation_metrics, flush=True)
    print("test", test_metrics, flush=True)

    torch.save(
        {
            "classifier": {
                key: value.detach().cpu()
                for key, value in trained.classifier.state_dict().items()
            },
            "labeled_count": int(train_indices.size),
            "target_model": "Yang-Wu BertEncoder (EMNLP 2021)",
            "protocol": "full official train pool, single seed",
            "classifier_epochs": int(config["training"]["epochs"]),
            "seed": int(config["training"]["seed"]),
            "model_seed": int(config["model"]["seed"]),
        },
        output_dir / "checkpoints" / "final.pt",
    )
    np.savez(
        output_dir / "final_predictions.npz",
        validation_labels=_host(validation["labels"]),
        validation_probabilities=_host(validation["probabilities"]),
        test_labels=_host(test["labels"]),
        test_probabilities=_host(test["probabilities"]),
    )
    result = {
        "protocol": "full official train pool ceiling",
        "strategy": "full_pool",
        "scenario": store.scenario.summary(),
        "seed": int(config["training"]["seed"]),
        "model_seed": int(config["model"]["seed"]),
        "labeled_count": int(train_indices.size),
        "labeled_fraction_of_train": 1.0,
        "validation": validation_metrics,
        "test": test_metrics,
        "training_history": trained.history,
        "timing": trained.timings | {"total_wall_sec": time.perf_counter() - start},
        "data_usage": {
            "official_train_pool": int(train_indices.size),
            "official_validation_evaluation_only": int(val_indices.size),
            "official_test_evaluation_only": int(test_indices.size),
            "final_labeled": int(train_indices.size),
            "final_fraction_of_train": 1.0,
        },
    }
    _write_json(output_dir / "final_metrics.json", result)
    _write_json(output_dir / "resolved_config.json", config)
    print("FULL_POOL_DONE", json.dumps(test_metrics), flush=True)


if __name__ == "__main__":
    main()
