#!/usr/bin/env python3
"""CPU/GPU microbench for adapter hot paths. Never touches external jobs."""

from __future__ import annotations

import argparse
import time

import torch

from mimic_comal.model import positive_similarity_thresholds, supervised_contrastive_loss
from mimic_comal.training import train_round


def _timed(fn, repeats: int) -> float:
    for _ in range(2):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("cuda requested but unavailable")

    torch.manual_seed(0)
    features = torch.randn(96, 50, 64, device=device)
    labels = (torch.rand(96, 50, device=device) > 0.85).float()
    contrastive_s = _timed(
        lambda: supervised_contrastive_loss(features, labels, anchor_chunk_size=2048),
        args.repeats,
    )
    print(f"contrastive_ms={contrastive_s * 1000:.3f}")

    latents = torch.randn(4000, 50, 64, device=device)
    labeled = (torch.rand(4000, 50, device=device) > 0.9).float()
    prototypes = torch.randn(51, 64, device=device)
    threshold_s = _timed(
        lambda: positive_similarity_thresholds(latents, labeled, prototypes),
        max(5, args.repeats // 2),
    )
    print(f"threshold_ms={threshold_s * 1000:.3f}")

    if device.type == "cpu":
        import numpy as np

        config = {
            "model": {"hidden_dims": [512, 256], "dropout": 0.0},
            "comal": {"label_dim": 64, "prototype_dim": 64, "anchor_chunk_size": 512},
            "training": {
                "device": "cpu",
                "precision": "fp32",
                "batch_size": 1024,
                "comal_batch_size": 64,
                "eval_batch_size": 2048,
                "epochs": 1,
                "comal_epochs": 1,
                "num_workers": 0,
                "pin_memory": False,
                "gpu_resident_features": False,
                "fused_optimizer": False,
            },
        }
        x = np.random.randn(8000, 256).astype("float16")
        y = (np.random.rand(8000, 50) > 0.9).astype("float32")
        start = time.perf_counter()
        trained = train_round(x, y, list(range(3000)), config, device)
        print(f"train_round_sec={time.perf_counter() - start:.3f} timings={trained.timings}")


if __name__ == "__main__":
    main()
