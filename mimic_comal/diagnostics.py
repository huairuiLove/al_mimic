"""Dataset and round-level diagnostics adapted from the CXR experiment harness."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


def expected_calibration_error(labels: np.ndarray, probabilities: np.ndarray, bins: int = 15) -> float:
    truth = labels.reshape(-1)
    confidence = probabilities.reshape(-1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidence >= lower) & (confidence < upper if upper < 1 else confidence <= upper)
        if mask.any():
            error += float(mask.mean()) * abs(float(confidence[mask].mean()) - float(truth[mask].mean()))
    return error


def build_round_diagnostics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    latents: np.ndarray,
    prototypes: torch.Tensor,
    label_names: tuple[str, ...],
    *,
    acquisition_scores: np.ndarray | None = None,
) -> dict[str, Any]:
    """Check prototype behavior, calibration, and score usefulness."""
    latent = F.normalize(torch.from_numpy(latents).float(), dim=-1)
    proto = F.normalize(prototypes.detach().cpu().float(), dim=-1)
    positive_similarity = torch.einsum("nld,ld->nl", latent, proto[:-1]).numpy()
    background_similarity = torch.einsum("nld,d->nl", latent, proto[-1]).numpy()
    pos_mask = labels >= 0.5
    neg_mask = ~pos_mask
    errors = np.not_equal(probabilities >= 0.5, labels >= 0.5).mean(axis=1)
    report: dict[str, Any] = {
        "samples": int(labels.shape[0]),
        "prototype_diagnostics": {
            "positive_own_prototype_similarity": float(positive_similarity[pos_mask].mean())
            if pos_mask.any()
            else None,
            "negative_own_prototype_similarity": float(positive_similarity[neg_mask].mean())
            if neg_mask.any()
            else None,
            "negative_background_similarity": float(background_similarity[neg_mask].mean())
            if neg_mask.any()
            else None,
            "positive_vs_background_margin": float(
                (positive_similarity - background_similarity)[pos_mask].mean()
            )
            if pos_mask.any()
            else None,
        },
        "calibration": {
            "ece": expected_calibration_error(labels, probabilities),
            "mean_confidence": float(np.maximum(probabilities, 1 - probabilities).mean()),
            "mean_sample_label_error": float(errors.mean()),
        },
    }
    if acquisition_scores is not None and acquisition_scores.size == errors.size:
        correlation = spearmanr(acquisition_scores, errors).statistic
        report["acquisition_diagnostics"] = {
            "spearman_score_vs_label_error": float(correlation) if np.isfinite(correlation) else None,
            "error_auroc": float(roc_auc_score(errors > np.median(errors), acquisition_scores))
            if np.unique(errors > np.median(errors)).size == 2
            else None,
        }
    positives = labels.sum(axis=0)
    rare_threshold = max(5, int(np.quantile(positives, 0.25)))
    rare: dict[str, Any] = {}
    for index, name in enumerate(label_names):
        if positives[index] <= rare_threshold and np.unique(labels[:, index]).size == 2:
            rare[name] = {
                "positives": int(positives[index]),
                "auprc": float(average_precision_score(labels[:, index], probabilities[:, index])),
            }
    report["rare_label_diagnostics"] = {"threshold": rare_threshold, "labels": rare}
    return report


def acquisition_summary(components: dict[str, np.ndarray], selected_positions: np.ndarray) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    selected_set = set(int(value) for value in selected_positions)
    for name, values in components.items():
        selected = np.asarray([value for index, value in enumerate(values) if index in selected_set])
        summary[name] = {
            "pool_mean": float(np.mean(values)),
            "pool_std": float(np.std(values)),
            "selected_mean": float(np.mean(selected)) if selected.size else None,
            "selected_min": float(np.min(selected)) if selected.size else None,
            "selected_max": float(np.max(selected)) if selected.size else None,
        }
    return summary
