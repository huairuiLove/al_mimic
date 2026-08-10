"""Diagnosis metrics used by Yang and Wu (EMNLP 2021)."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np


def top_k_recall(
    labels: np.ndarray, probabilities: np.ndarray, k_values: Iterable[int]
) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int8)
    probabilities = np.asarray(probabilities, dtype=np.float32)
    if labels.shape != probabilities.shape or labels.ndim != 2:
        raise ValueError("labels and probabilities must have the same [samples, labels] shape")
    positives = labels.sum(axis=1)
    if np.any(positives == 0):
        raise ValueError("top-k recall is undefined for visits without positive labels")
    order = np.argsort(-probabilities, axis=1, kind="stable")
    result: dict[str, float] = {}
    for requested_k in k_values:
        k = min(int(requested_k), labels.shape[1])
        hits = np.take_along_axis(labels, order[:, :k], axis=1).sum(axis=1)
        result[f"recall_at_{requested_k}"] = float(np.mean(hits / positives))
    return result


def multilabel_metrics(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float | None = None
) -> dict[str, Any]:
    """Return only the Recall@10/20/30 metrics reported for Diagnoses."""
    del threshold
    return top_k_recall(labels, probabilities, (10, 20, 30))
