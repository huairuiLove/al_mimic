"""MoDIS disagreement, instability, and sufficiency-penalized acquisition."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn

from al_mimic.utils.fusion import fuse_token_batches, probabilities_from_fused

from .intervene import interpolate_modality_token
from .probes import MoDISProbeState, probe_probabilities


@dataclass(frozen=True)
class InstabilityResult:
    per_modality: torch.Tensor
    combined: torch.Tensor
    flip_curves: list[list[float]]
    monotonicity_violation_rate: list[float]


@dataclass(frozen=True)
class MoDISAcquisitionResult:
    selected_positions: torch.Tensor
    disagreement: torch.Tensor
    instability: torch.Tensor
    dominance: torch.Tensor
    sufficiency_penalty: torch.Tensor
    combined: torch.Tensor
    diagnostics: dict[str, Any]


def _binary_entropy(probabilities: torch.Tensor, epsilon: float = 1e-7) -> torch.Tensor:
    values = probabilities.float().clamp(epsilon, 1.0 - epsilon)
    return -(values * values.log() + (1.0 - values) * (1.0 - values).log())


def quantile_thresholds(probabilities: torch.Tensor, prevalence: torch.Tensor) -> torch.Tensor:
    """Column-wise linear quantiles at ``1 - prevalence`` without label loops."""
    if probabilities.ndim != 2 or prevalence.shape != probabilities.shape[1:]:
        raise ValueError("probabilities and prevalence must have shapes [N,C] and [C]")
    if probabilities.shape[0] == 0:
        raise ValueError("candidate probabilities cannot be empty")
    sorted_values = probabilities.float().sort(dim=0).values
    positions = (1.0 - prevalence.float().clamp(0.0, 1.0)) * (probabilities.shape[0] - 1)
    lower = positions.floor().long()
    upper = positions.ceil().long()
    columns = torch.arange(probabilities.shape[1], device=probabilities.device)
    lower_values = sorted_values[lower, columns]
    upper_values = sorted_values[upper, columns]
    return lower_values + (positions - lower.float()) * (upper_values - lower_values)


def modality_thresholds(probabilities: torch.Tensor, prevalence: torch.Tensor) -> torch.Tensor:
    if probabilities.ndim != 3:
        raise ValueError("modality probabilities must have shape [N,M,C]")
    return torch.stack(
        [
            quantile_thresholds(probabilities[:, modality], prevalence)
            for modality in range(probabilities.shape[1])
        ]
    )


def decision_support(
    fused_probabilities: torch.Tensor,
    modality_probabilities: torch.Tensor,
    fused_thresholds: torch.Tensor,
    probe_thresholds: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    fused_positive = fused_probabilities >= fused_thresholds
    modality_positive = modality_probabilities >= probe_thresholds[None, :, :]
    support = fused_positive | modality_positive.any(dim=1)
    empty = ~support.any(dim=1)
    if empty.any():
        fallback = fused_probabilities.argmax(dim=1)
        support = support.clone()
        support[empty, fallback[empty]] = True
    return fused_positive, modality_positive, support


def generalized_js_disagreement(
    modality_probabilities: torch.Tensor,
    label_weights: torch.Tensor,
    support: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized Bernoulli JSD per label and its support-set mean per sample."""
    if modality_probabilities.ndim != 3:
        raise ValueError("modality probabilities must have shape [N,M,C]")
    if label_weights.shape != modality_probabilities.shape[1:]:
        raise ValueError("label_weights must have shape [M,C]")
    if support.shape != (modality_probabilities.shape[0], modality_probabilities.shape[2]):
        raise ValueError("support must have shape [N,C]")
    weights = label_weights.float()
    mixture = (modality_probabilities.float() * weights[None, :, :]).sum(dim=1)
    per_label = _binary_entropy(mixture) - (
        _binary_entropy(modality_probabilities) * weights[None, :, :]
    ).sum(dim=1)
    per_label = per_label.clamp_min(0.0)
    disagreement = (per_label * support).sum(dim=1) / support.sum(dim=1).clamp_min(1)
    return disagreement, per_label


def modality_sufficiency(
    fused_positive: torch.Tensor,
    modality_positive: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return max single-modality Jaccard agreement and all modality agreements."""
    if (
        modality_positive.shape[0] != fused_positive.shape[0]
        or modality_positive.shape[2] != fused_positive.shape[1]
    ):
        raise ValueError("decision tensors have incompatible shapes")
    fused = fused_positive[:, None, :]
    intersection = (modality_positive & fused).sum(dim=2).float()
    union = (modality_positive | fused).sum(dim=2).float()
    jaccard = torch.where(union > 0, intersection / union, torch.zeros_like(union))
    return jaccard.max(dim=1).values, jaccard


def _average_rank_fraction(values: torch.Tensor) -> torch.Tensor:
    """Tie-aware ascending ranks in ``(0, 1]``."""
    flat = values.float().reshape(-1)
    if flat.numel() == 0:
        return flat
    order = torch.argsort(flat, stable=True)
    sorted_values = flat[order]
    _unique, inverse, counts = torch.unique_consecutive(
        sorted_values, return_inverse=True, return_counts=True
    )
    ends = counts.cumsum(dim=0).float()
    starts = ends - counts.float() + 1.0
    average = (starts + ends) * 0.5 / flat.numel()
    sorted_ranks = average[inverse]
    ranks = torch.empty_like(sorted_ranks)
    ranks[order] = sorted_ranks
    return ranks.reshape(values.shape)


def _spearman(left: torch.Tensor, right: torch.Tensor) -> float | None:
    left_rank = _average_rank_fraction(left).double().reshape(-1)
    right_rank = _average_rank_fraction(right).double().reshape(-1)
    left_centered = left_rank - left_rank.mean()
    right_centered = right_rank - right_rank.mean()
    denominator = torch.linalg.vector_norm(left_centered) * torch.linalg.vector_norm(right_centered)
    if float(denominator) <= 1e-12:
        return None
    return float((left_centered * right_centered).sum() / denominator)


@torch.inference_mode()
def critical_instability(
    classifier: nn.Module,
    tokens: torch.Tensor,
    prototypes: torch.Tensor,
    base_decisions: torch.Tensor,
    thresholds: torch.Tensor,
    modality_weights: torch.Tensor,
    *,
    grid_k: int,
    bisect_steps: int,
    fusion_batch_size: int,
) -> InstabilityResult:
    """Estimate equation (7) by streaming one modality/grid point at a time."""
    if grid_k < 1:
        raise ValueError("grid_k must be positive")
    if bisect_steps < 0:
        raise ValueError("bisect_steps must be non-negative")
    if tokens.ndim != 3 or prototypes.shape != tokens.shape[1:]:
        raise ValueError("tokens and prototypes have incompatible shapes")
    if base_decisions.shape != (tokens.shape[0], thresholds.numel()):
        raise ValueError("base decisions and thresholds have incompatible shapes")
    if modality_weights.shape != (tokens.shape[1],):
        raise ValueError("modality_weights must have shape [M]")

    per_modality = torch.zeros(tokens.shape[0], tokens.shape[1], dtype=torch.float32, device=tokens.device)
    flip_curves: list[list[float]] = []
    violations: list[float] = []
    for modality in range(tokens.shape[1]):
        changed_grid = torch.zeros(tokens.shape[0], grid_k, dtype=torch.bool, device=tokens.device)
        ever_changed = torch.zeros(tokens.shape[0], dtype=torch.bool, device=tokens.device)
        alpha_star = torch.zeros(tokens.shape[0], dtype=torch.float32, device=tokens.device)
        for grid_index in range(grid_k):
            alpha = 1.0 - (grid_index + 1) / grid_k
            intervened = interpolate_modality_token(tokens, prototypes, modality, alpha)
            fused = fuse_token_batches(classifier, intervened, batch_size=fusion_batch_size)
            decisions = probabilities_from_fused(classifier, fused) >= thresholds
            changed = (decisions != base_decisions).any(dim=1)
            changed_grid[:, grid_index] = changed
            first_change = changed & ~ever_changed
            alpha_star[first_change] = float(alpha)
            ever_changed |= changed

        if bisect_steps and ever_changed.any():
            active = ever_changed.nonzero(as_tuple=False).squeeze(1)
            lower = alpha_star.index_select(0, active)
            upper = (lower + 1.0 / grid_k).clamp_max(1.0)
            active_tokens = tokens.index_select(0, active)
            active_base = base_decisions.index_select(0, active)
            for _ in range(bisect_steps):
                midpoint = (lower + upper) * 0.5
                intervened = interpolate_modality_token(active_tokens, prototypes, modality, midpoint)
                fused = fuse_token_batches(classifier, intervened, batch_size=fusion_batch_size)
                decisions = probabilities_from_fused(classifier, fused) >= thresholds
                changed = (decisions != active_base).any(dim=1)
                lower = torch.where(changed, midpoint, lower)
                upper = torch.where(changed, upper, midpoint)
            alpha_star[active] = lower

        per_modality[:, modality] = alpha_star
        flip_curves.append([float(value) for value in changed_grid.float().mean(dim=0)])
        nonmonotonic = (changed_grid[:, 1:].int() < changed_grid[:, :-1].int()).any(dim=1)
        violations.append(float(nonmonotonic.float().mean()))
    combined = (per_modality * modality_weights[None, :]).sum(dim=1)
    return InstabilityResult(
        per_modality=per_modality,
        combined=combined,
        flip_curves=flip_curves,
        monotonicity_violation_rate=violations,
    )


def _component_quantiles(values: torch.Tensor) -> dict[str, float]:
    levels = torch.tensor([0.0, 0.25, 0.5, 0.75, 0.9, 1.0], device=values.device)
    quantiles = torch.quantile(values.float(), levels)
    return {
        name: float(value) for name, value in zip(("min", "q25", "median", "q75", "q90", "max"), quantiles)
    }


def _maximum_tie_fraction(values: torch.Tensor) -> float:
    if values.numel() == 0:
        return 0.0
    _unique, counts = torch.unique(values, return_counts=True)
    return float(counts.max().float() / values.numel())


def _disagreement_dominance_histogram(
    disagreement: torch.Tensor, dominance: torch.Tensor, num_modalities: int
) -> list[list[int]]:
    bins = 10
    normalized_disagreement = disagreement / max(math.log(num_modalities), 1e-12)
    disagreement_bin = (normalized_disagreement * bins).long().clamp(0, bins - 1)
    dominance_bin = (dominance * bins).long().clamp(0, bins - 1)
    counts = torch.bincount(disagreement_bin * bins + dominance_bin, minlength=bins * bins)
    return counts.reshape(bins, bins).detach().cpu().tolist()


def _geometric_rank_score(
    components: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    beta: tuple[float, float, float],
) -> torch.Tensor:
    ranks = tuple(_average_rank_fraction(component).clamp_min(1e-12) for component in components)
    log_score = torch.zeros_like(ranks[0])
    for exponent, rank in zip(beta, ranks):
        if exponent:
            log_score = log_score + float(exponent) * rank.log()
    return log_score.exp()


@torch.inference_mode()
def acquire_modis(
    classifier: nn.Module,
    probe_state: MoDISProbeState,
    candidates: dict[str, torch.Tensor],
    *,
    query_size: int,
    config: dict[str, Any],
    initial_prevalence: torch.Tensor | None = None,
    candidate_metadata: Mapping[str, np.ndarray] | None = None,
    comparison_scores: Mapping[str, torch.Tensor] | None = None,
) -> MoDISAcquisitionResult:
    """Score a candidate pool according to equations (3)-(13)."""
    if not callable(getattr(classifier, "fuse_from_tokens", None)):
        raise ValueError("MoDIS requires a classifier with fuse_from_tokens")
    required = {"probabilities", "modality_tokens"}
    if not required.issubset(candidates):
        raise ValueError("candidates must include probabilities and modality_tokens")
    probabilities = candidates["probabilities"].float()
    tokens = candidates["modality_tokens"].float()
    if probabilities.shape[0] == 0:
        raise ValueError("candidate pool cannot be empty")
    modis_cfg = config.get("modis", {})
    beta_values = tuple(float(value) for value in modis_cfg.get("beta", [1.0, 1.0, 1.0]))
    if len(beta_values) != 3:
        raise ValueError("modis.beta must contain three values")
    beta = (beta_values[0], beta_values[1], beta_values[2])
    fusion_batch_size = int(modis_cfg.get("fusion_batch_size", 4096))
    probe_batch_size = int(modis_cfg.get("probe_eval_batch_size", fusion_batch_size))

    probe_probs = probe_probabilities(probe_state.probes, tokens, batch_size=probe_batch_size)
    fused_threshold = quantile_thresholds(probabilities, probe_state.labeled_prevalence)
    probe_threshold = modality_thresholds(probe_probs, probe_state.labeled_prevalence)
    fused_positive, modality_positive, support = decision_support(
        probabilities,
        probe_probs,
        fused_threshold,
        probe_threshold,
    )
    disagreement, _per_label_js = generalized_js_disagreement(
        probe_probs,
        probe_state.statistics.label_weights,
        support,
    )
    dominance, modality_jaccard = modality_sufficiency(fused_positive, modality_positive)
    sufficiency_penalty = 1.0 - dominance

    screen_score = _average_rank_fraction(disagreement) * _average_rank_fraction(sufficiency_penalty)
    requested_workset = int(modis_cfg.get("workset_size", 5000))
    workset_count = min(probabilities.shape[0], max(int(query_size), max(1, requested_workset)))
    screen_order = torch.argsort(screen_score, descending=True, stable=True)
    work_positions = screen_order[:workset_count]
    work_instability = critical_instability(
        classifier,
        tokens.index_select(0, work_positions),
        probe_state.prototypes,
        fused_positive.index_select(0, work_positions),
        fused_threshold,
        probe_state.statistics.modality_weights,
        grid_k=int(modis_cfg.get("grid_k", 8)),
        bisect_steps=int(modis_cfg.get("bisect_steps", 0)),
        fusion_batch_size=fusion_batch_size,
    )
    instability = torch.zeros(probabilities.shape[0], dtype=torch.float32, device=probabilities.device)
    instability[work_positions] = work_instability.combined

    combined = _geometric_rank_score((disagreement, instability, sufficiency_penalty), beta)
    if workset_count < probabilities.shape[0]:
        work_mask = torch.zeros(probabilities.shape[0], dtype=torch.bool, device=probabilities.device)
        work_mask[work_positions] = True
        combined = combined.masked_fill(~work_mask, 0.0)
    selection_count = min(int(query_size), workset_count)
    ranked_work = torch.argsort(combined.index_select(0, work_positions), descending=True, stable=True)
    selected_positions = work_positions.index_select(0, ranked_work[:selection_count])

    tail_start = max(0, math.floor(0.9 * workset_count))
    screen_rank = torch.empty_like(screen_order)
    screen_rank[screen_order] = torch.arange(screen_order.numel(), device=screen_order.device)
    selected_screen_ranks = screen_rank.index_select(0, selected_positions)
    boundary_fraction = (
        float((selected_screen_ranks >= tail_start).float().mean()) if selected_screen_ranks.numel() else 0.0
    )
    prediction_entropy = _binary_entropy(probabilities).mean(dim=1)
    work_disagreement = disagreement.index_select(0, work_positions)
    work_sufficiency_penalty = sufficiency_penalty.index_select(0, work_positions)
    work_combined = combined.index_select(0, work_positions)
    pairwise_correlations = {
        "disagreement_vs_instability": _spearman(work_disagreement, work_instability.combined),
        "disagreement_vs_sufficiency_penalty": _spearman(work_disagreement, work_sufficiency_penalty),
        "instability_vs_sufficiency_penalty": _spearman(work_instability.combined, work_sufficiency_penalty),
    }
    baseline_correlations: dict[str, float | None] = {
        "fusion_prediction_entropy": _spearman(
            work_combined, prediction_entropy.index_select(0, work_positions)
        ),
        "paper_comal": None,
        "mosaic_synergy": None,
    }
    if comparison_scores:
        for name, score in comparison_scores.items():
            if score.numel() == combined.numel():
                baseline_correlations[str(name)] = _spearman(
                    work_combined, score.index_select(0, work_positions)
                )

    metadata_correlations: dict[str, float | None] = {}
    if candidate_metadata:
        for name, values in candidate_metadata.items():
            metadata = torch.as_tensor(np.asarray(values, dtype=np.float32), device=dominance.device)
            if metadata.numel() == dominance.numel():
                metadata_correlations[str(name)] = _spearman(dominance, metadata)

    prevalence_ratio = None
    if initial_prevalence is not None and initial_prevalence.shape == probe_state.labeled_prevalence.shape:
        prevalence_ratio = (
            (probe_state.labeled_prevalence / initial_prevalence.to(probabilities.device).clamp_min(1e-8))
            .detach()
            .cpu()
            .tolist()
        )
    diagnostics = {
        "criteria": {
            "disagreement_quantiles": _component_quantiles(disagreement),
            "disagreement_max_tie_fraction": _maximum_tie_fraction(disagreement),
            "instability_quantiles_on_workset": _component_quantiles(work_instability.combined),
            "instability_zero_fraction_on_workset": float((work_instability.combined == 0).float().mean()),
            "instability_zero_fraction_by_modality": [
                float(value) for value in (work_instability.per_modality == 0).float().mean(dim=0)
            ],
            "dominance_one_fraction": float((dominance == 1).float().mean()),
            "dominance_quantiles": _component_quantiles(dominance),
            "dominance_max_tie_fraction": _maximum_tie_fraction(dominance),
            "modality_jaccard_mean": [float(value) for value in modality_jaccard.mean(dim=0)],
        },
        "redundancy": {
            "instability_metric_scope": "screened_workset",
            "pairwise_spearman": pairwise_correlations,
            "baseline_spearman": baseline_correlations,
            "disagreement_dominance_histogram": _disagreement_dominance_histogram(
                disagreement, dominance, tokens.shape[1]
            ),
        },
        "sufficiency_metadata_spearman": metadata_correlations,
        "probe_reliability": probe_state.diagnostics,
        "intervention": {
            "grid_k": int(modis_cfg.get("grid_k", 8)),
            "bisect_steps": int(modis_cfg.get("bisect_steps", 0)),
            "flip_curves_by_modality": work_instability.flip_curves,
            "monotonicity_violation_rate_by_modality": (work_instability.monotonicity_violation_rate),
            "instability_vs_text_modality_spearman": _spearman(
                work_instability.combined, work_instability.per_modality[:, 0]
            ),
        },
        "thresholds": {
            "fused": fused_threshold.detach().cpu().tolist(),
            "probes": probe_threshold.detach().cpu().tolist(),
            "mean_predicted_cardinality": float(fused_positive.sum(dim=1).float().mean()),
            "predicted_to_labeled_cardinality_ratio": float(
                fused_positive.sum(dim=1).float().mean() / max(probe_state.labeled_cardinality, 1e-8)
            ),
            "labeled_prevalence_to_initial_ratio": prevalence_ratio,
        },
        "screening": {
            "candidate_count": int(probabilities.shape[0]),
            "workset_size": workset_count,
            "selected_in_workset_bottom_decile_fraction": boundary_fraction,
            "exact": workset_count == probabilities.shape[0],
        },
        "beta": list(beta),
    }
    return MoDISAcquisitionResult(
        selected_positions=selected_positions,
        disagreement=disagreement,
        instability=instability,
        dominance=dominance,
        sufficiency_penalty=sufficiency_penalty,
        combined=combined,
        diagnostics=diagnostics,
    )
