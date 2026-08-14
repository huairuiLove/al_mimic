"""Metrics for the registered native multi-label MIMIC-III tasks."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from .tasks import task_spec


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


def ranking_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    """Compute label-balanced and pooled ranking metrics with explicit coverage."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    labels = np.asarray(labels, dtype=np.int8)
    probabilities = np.asarray(probabilities, dtype=np.float32)
    if labels.shape != probabilities.shape or labels.ndim != 2:
        raise ValueError("labels and probabilities must have the same [samples, labels] shape")
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities must be finite")

    positives = labels.sum(axis=0)
    negatives = labels.shape[0] - positives
    auprc_mask = positives > 0
    auroc_mask = auprc_mask & (negatives > 0)
    if not auprc_mask.any():
        raise ValueError("AUPRC is undefined because the split has no positive labels")
    if not auroc_mask.any():
        raise ValueError("AUROC is undefined because no label has both classes")

    return {
        "macro_auprc": float(
            average_precision_score(
                labels[:, auprc_mask], probabilities[:, auprc_mask], average="macro"
            )
        ),
        "micro_auprc": float(average_precision_score(labels, probabilities, average="micro")),
        "macro_auroc": float(
            roc_auc_score(labels[:, auroc_mask], probabilities[:, auroc_mask], average="macro")
        ),
        "micro_auroc": float(roc_auc_score(labels, probabilities, average="micro")),
        "metric_label_coverage": {
            "total": int(labels.shape[1]),
            "auprc": int(auprc_mask.sum()),
            "auroc": int(auroc_mask.sum()),
        },
    }


def task_multilabel_metrics(
    config: dict[str, Any], labels: np.ndarray, probabilities: np.ndarray
) -> dict[str, Any]:
    spec = task_spec(config)
    if spec.task_id == "icd9_diagnoses":
        return multilabel_metrics(labels, probabilities)
    calculated = ranking_metrics(labels, probabilities)
    selected = {name: calculated[name] for name in spec.metrics}
    selected["metric_label_coverage"] = calculated["metric_label_coverage"]
    return selected
