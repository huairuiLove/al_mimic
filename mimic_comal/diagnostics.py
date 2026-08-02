"""Dataset and round-level diagnostics adapted from the CXR experiment harness."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


def expected_calibration_error(labels: np.ndarray, probabilities: np.ndarray, bins: int = 15) -> float:
    truth = np.asarray(labels, dtype=np.float64).reshape(-1)
    confidence = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    # Vectorized ECE: one digitize pass instead of a Python bin loop.
    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_ids = np.clip(np.digitize(confidence, edges[1:-1], right=False), 0, bins - 1)
    counts = np.bincount(bin_ids, minlength=bins).astype(np.float64)
    conf_sums = np.bincount(bin_ids, weights=confidence, minlength=bins)
    truth_sums = np.bincount(bin_ids, weights=truth, minlength=bins)
    active = counts > 0
    if not np.any(active):
        return 0.0
    weights = counts[active] / confidence.size
    gaps = np.abs(conf_sums[active] / counts[active] - truth_sums[active] / counts[active])
    return float(np.sum(weights * gaps))


def build_round_diagnostics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    latents: np.ndarray,
    prototypes: torch.Tensor,
    label_names: tuple[str, ...],
    *,
    acquisition_scores: np.ndarray | None = None,
    prototype_similarities: np.ndarray | torch.Tensor | None = None,
) -> dict[str, Any]:
    """Check prototype behavior, calibration, and score usefulness."""
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    if prototype_similarities is not None:
        # Prefer model-emitted similarities (evaluation metric), shape [N, L, L+1] or [N, L+1].
        sims = (
            prototype_similarities.detach().float().cpu().numpy()
            if isinstance(prototype_similarities, torch.Tensor)
            else np.asarray(prototype_similarities)
        )
        if sims.ndim == 3:
            # [N, L, L+1] -> own-label diagonal for positives/negatives, last column background.
            index = np.arange(sims.shape[1])
            positive_similarity = sims[:, index, index]
            background_similarity = sims[:, :, -1]
        else:
            # Fallback: if only pooled [N, L+1], reconstruct via latents below.
            positive_similarity = None
            background_similarity = None
    else:
        positive_similarity = None
        background_similarity = None
    if positive_similarity is None or background_similarity is None:
        if isinstance(latents, torch.Tensor):
            latent = F.normalize(latents.detach().float(), dim=-1)
            proto = F.normalize(prototypes.detach().to(device=latent.device).float(), dim=-1)
            positive_similarity = torch.einsum("nld,ld->nl", latent, proto[:-1]).detach().cpu().numpy()
            background_similarity = torch.einsum("nld,d->nl", latent, proto[-1]).detach().cpu().numpy()
        else:
            latent = F.normalize(torch.as_tensor(latents).float(), dim=-1)
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
            "mean_prototype_similarity": float(np.asarray(positive_similarity).mean()),
            "mean_background_similarity": float(np.asarray(background_similarity).mean()),
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
    selected_positions = np.asarray(selected_positions, dtype=np.int64)
    for name, values in components.items():
        values = np.asarray(values)
        selected = values[selected_positions] if selected_positions.size else values[:0]
        summary[name] = {
            "pool_mean": float(np.mean(values)),
            "pool_std": float(np.std(values)),
            "selected_mean": float(np.mean(selected)) if selected.size else None,
            "selected_min": float(np.min(selected)) if selected.size else None,
            "selected_max": float(np.max(selected)) if selected.size else None,
        }
    return summary
