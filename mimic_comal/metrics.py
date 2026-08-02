"""Multi-label metrics used by the original CoMAL reproduction."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


def _safe_macro_auprc(labels: np.ndarray, probabilities: np.ndarray) -> tuple[float, list[float | None]]:
    values: list[float | None] = []
    for column in range(labels.shape[1]):
        if np.unique(labels[:, column]).size < 2:
            values.append(None)
        else:
            values.append(float(average_precision_score(labels[:, column], probabilities[:, column])))
    valid = [value for value in values if value is not None]
    return (float(np.mean(valid)) if valid else 0.0), values


def multilabel_metrics(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5
) -> dict[str, Any]:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = probabilities >= threshold
    macro_auprc, per_label = _safe_macro_auprc(labels, probabilities)
    metrics: dict[str, Any] = {
        "auprc_micro": float(average_precision_score(labels, probabilities, average="micro")),
        "auprc_macro": macro_auprc,
        "f1_micro": float(f1_score(labels, predictions, average="micro", zero_division=0)),
        "f1_macro": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "precision_micro": float(precision_score(labels, predictions, average="micro", zero_division=0)),
        "recall_micro": float(recall_score(labels, predictions, average="micro", zero_division=0)),
        "per_label_auprc": per_label,
    }
    try:
        metrics["auroc_micro"] = float(roc_auc_score(labels, probabilities, average="micro"))
    except ValueError:
        metrics["auroc_micro"] = None
    label_sums = labels.sum(axis=1)
    for k in (1, 3, 5):
        actual_k = min(k, labels.shape[1])
        top = np.argpartition(-probabilities, kth=actual_k - 1, axis=1)[:, :actual_k]
        hits = np.take_along_axis(labels, top, axis=1).sum(axis=1)
        metrics[f"precision_at_{k}"] = float(np.mean(hits / actual_k))
        metrics[f"ndcg_proxy_at_{k}"] = float(np.mean(hits / np.minimum(np.maximum(label_sums, 1), actual_k)))
    return metrics
