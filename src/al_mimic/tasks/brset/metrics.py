"""Thresholded and ranking metrics for BRSET multi-label diagnosis."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .data import LABEL_COLUMNS


def _validate(labels: np.ndarray, probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int8)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if labels.shape != probabilities.shape or labels.ndim != 2:
        raise ValueError("labels and probabilities must have the same [N,C] shape")
    if labels.shape[1] != len(LABEL_COLUMNS):
        raise ValueError(f"BRSET metrics require {len(LABEL_COLUMNS)} labels")
    if not np.isin(labels, (0, 1)).all() or not np.isfinite(probabilities).all():
        raise ValueError("BRSET labels must be binary and probabilities finite")
    return labels, probabilities


def fit_f1_thresholds(labels: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    """Fit the paper's per-label F1 grid on validation data only."""
    labels, probabilities = _validate(labels, probabilities)
    grid = np.linspace(0.0, 1.0, 26)
    thresholds = np.full(labels.shape[1], 0.5, dtype=np.float64)
    for column in range(labels.shape[1]):
        if np.unique(labels[:, column]).size < 2:
            continue
        scores = np.asarray(
            [
                f1_score(labels[:, column], probabilities[:, column] >= threshold, zero_division=0)
                for threshold in grid
            ]
        )
        best = np.flatnonzero(scores == scores.max())
        thresholds[column] = grid[best[np.argmin(np.abs(grid[best] - 0.5))]]
    return thresholds.astype(np.float32)


def _binary_details(target: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    true_negative = int(np.logical_and(target == 0, predicted == 0).sum())
    false_positive = int(np.logical_and(target == 0, predicted == 1).sum())
    false_negative = int(np.logical_and(target == 1, predicted == 0).sum())
    specificity = true_negative / max(true_negative + false_positive, 1)
    negative_predictive_value = true_negative / max(true_negative + false_negative, 1)
    return {
        "specificity": float(specificity),
        "negative_predictive_value": float(negative_predictive_value),
    }


def multilabel_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray,
) -> dict[str, Any]:
    labels, probabilities = _validate(labels, probabilities)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    if thresholds.shape != (labels.shape[1],):
        raise ValueError("thresholds must have one value per BRSET label")
    predictions = probabilities >= thresholds[None, :]
    per_label: dict[str, dict[str, float | int | None]] = {}
    auc_values: list[float] = []
    auprc_values: list[float] = []
    for column, name in enumerate(LABEL_COLUMNS):
        target = labels[:, column]
        predicted = predictions[:, column]
        auc = None
        if np.unique(target).size == 2:
            auc = float(roc_auc_score(target, probabilities[:, column]))
            auc_values.append(auc)
        auprc = None
        if int(target.sum()) > 0:
            auprc = float(average_precision_score(target, probabilities[:, column]))
            auprc_values.append(auprc)
        per_label[name] = {
            "support": int(target.sum()),
            "threshold": float(thresholds[column]),
            "auroc": auc,
            "auprc": auprc,
            "accuracy": float(accuracy_score(target, predicted)),
            "precision": float(precision_score(target, predicted, zero_division=0)),
            "recall": float(recall_score(target, predicted, zero_division=0)),
            "f1": float(f1_score(target, predicted, zero_division=0)),
        } | _binary_details(target, predicted)
    return {
        "macro_auroc": float(np.mean(auc_values)),
        "micro_auroc": float(roc_auc_score(labels.reshape(-1), probabilities.reshape(-1))),
        "macro_auprc": float(np.mean(auprc_values)),
        "micro_auprc": float(average_precision_score(labels.reshape(-1), probabilities.reshape(-1))),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(labels, predictions, average="micro", zero_division=0)),
        "macro_precision": float(precision_score(labels, predictions, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(labels, predictions, average="macro", zero_division=0)),
        "subset_accuracy": float(accuracy_score(labels, predictions)),
        "hamming_loss": float(hamming_loss(labels, predictions)),
        "per_label": per_label,
    }
