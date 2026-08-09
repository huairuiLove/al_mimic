"""Two-stage MoSAIC acquisition with exact three-modality lattice evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from mimic_comal.model import MultimodalFusionClassifier

from .design import FisherDesign
from .intervene import coalition_masks, intervene_tokens
from .lattice import LatticeDecomposition, decompose_lattice
from .tokens import fuse_token_batches, probabilities_from_fused


@dataclass(frozen=True)
class MosaicAcquisitionResult:
    selected_positions: torch.Tensor
    combined: torch.Tensor
    additive: torch.Tensor
    synergy: torch.Tensor
    total_gain: torch.Tensor
    diagnostics: dict[str, Any]


def _rank_union(
    design_bound: torch.Tensor,
    interaction_proxy: torch.Tensor,
    *,
    design_size: int,
    synergy_size: int,
) -> tuple[torch.Tensor, dict[str, int]]:
    count = int(design_bound.numel())
    design_count = min(max(int(design_size), 1), count)
    synergy_count = min(max(int(synergy_size), 1), count)
    design_top = torch.topk(design_bound, design_count, sorted=False).indices
    synergy_top = torch.topk(interaction_proxy, synergy_count, sorted=False).indices
    union = torch.unique(torch.cat((design_top, synergy_top)), sorted=True)
    design_membership = torch.isin(union, design_top)
    synergy_membership = torch.isin(union, synergy_top)
    certificate = {
        "design_branch": int(design_membership.sum()),
        "synergy_branch": int(synergy_membership.sum()),
        "intersection": int((design_membership & synergy_membership).sum()),
        "union": int(union.numel()),
    }
    return union, certificate


def _closure_samples(
    classifier: MultimodalFusionClassifier,
    labeled_tokens: torch.Tensor,
    *,
    count: int,
    rng: np.random.Generator,
    lambda_alpha: float | None,
    fusion_batch_size: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    if count <= 0:
        return None, None
    base_indices = rng.integers(0, labeled_tokens.shape[0], size=count)
    base = labeled_tokens.index_select(
        0, torch.as_tensor(base_indices, dtype=torch.long, device=labeled_tokens.device)
    )
    chimeras = intervene_tokens(
        base,
        labeled_tokens,
        torch.zeros(labeled_tokens.shape[1], dtype=torch.bool, device=labeled_tokens.device),
        partners=1,
        rng=rng,
        lambda_alpha=lambda_alpha,
    )[:, 0]
    fused = fuse_token_batches(classifier, chimeras, batch_size=fusion_batch_size)
    return fused, probabilities_from_fused(classifier, fused)


def _counterfactual_features(
    classifier: MultimodalFusionClassifier,
    work_tokens: torch.Tensor,
    partner_tokens: torch.Tensor,
    *,
    partners: int,
    rng: np.random.Generator,
    lambda_alpha: float | None,
    fusion_batch_size: int,
) -> torch.Tensor:
    masks = coalition_masks(int(work_tokens.shape[1])).to(work_tokens.device)
    coalition_features = []
    for mask in masks:
        intervened = intervene_tokens(
            work_tokens,
            partner_tokens,
            mask,
            partners=partners,
            rng=rng,
            lambda_alpha=lambda_alpha,
        )
        coalition_features.append(
            fuse_token_batches(classifier, intervened, batch_size=fusion_batch_size)
        )
    return torch.stack(coalition_features, dim=1)


def _lattice_values(
    design: FisherDesign,
    classifier: MultimodalFusionClassifier,
    counterfactual_features: torch.Tensor,
    *,
    value_batch_size: int,
) -> torch.Tensor:
    shape = counterfactual_features.shape
    flat = counterfactual_features.reshape(-1, shape[-1])
    values = torch.empty(flat.shape[0], dtype=torch.float32, device=flat.device)
    step = max(1, int(value_batch_size))
    for start in range(0, int(flat.shape[0]), step):
        stop = min(start + step, int(flat.shape[0]))
        batch = flat[start:stop]
        probabilities = probabilities_from_fused(classifier, batch)
        values[start:stop] = design.marginal_gain(batch, probabilities)
    values = values.reshape(shape[:-1]).mean(dim=-1)
    # The empty coalition is a pool-level zero point rather than an x-specific MC draw.
    values[:, 0] = values[:, 0].mean()
    return values


def _score_workset(
    design: FisherDesign,
    classifier: MultimodalFusionClassifier,
    counterfactual_features: torch.Tensor,
    *,
    eta: float,
    value_batch_size: int,
) -> tuple[torch.Tensor, LatticeDecomposition]:
    values = _lattice_values(
        design,
        classifier,
        counterfactual_features,
        value_batch_size=value_batch_size,
    )
    return values, decompose_lattice(values, num_modalities=3, eta=eta)


@torch.inference_mode()
def acquire_mosaic(
    classifier: MultimodalFusionClassifier,
    labeled: dict[str, torch.Tensor],
    reference: dict[str, torch.Tensor],
    candidates: dict[str, torch.Tensor],
    *,
    query_size: int,
    config: dict[str, Any],
    seed: int,
) -> MosaicAcquisitionResult:
    """Run MoSAIC and return candidate-relative positions in greedy order."""
    if not isinstance(classifier, MultimodalFusionClassifier):
        raise ValueError("MoSAIC requires MultimodalFusionClassifier")
    mosaic_cfg = config.get("mosaic", {})
    eta = float(mosaic_cfg.get("eta", 0.25))
    if not 0.0 <= eta <= 1.0:
        raise ValueError("mosaic.eta must be in [0, 1]")
    partners = max(1, int(mosaic_cfg.get("partners", 4)))
    lambda_value = mosaic_cfg.get("lambda_alpha", None)
    lambda_alpha = None if lambda_value in (None, "", 0, 0.0) else float(lambda_value)
    fusion_batch_size = int(mosaic_cfg.get("fusion_batch_size", 4096))
    value_batch_size = int(mosaic_cfg.get("value_batch_size", 4096))
    rng = np.random.default_rng(int(seed))

    closure_features, closure_probabilities = _closure_samples(
        classifier,
        labeled["modality_tokens"],
        count=int(mosaic_cfg.get("mixup_closure_samples", 0)),
        rng=rng,
        lambda_alpha=lambda_alpha,
        fusion_batch_size=fusion_batch_size,
    )
    design = FisherDesign.build(
        labeled["features"],
        labeled["probabilities"],
        reference["features"],
        reference["probabilities"],
        reference["labels"],
        damping=float(mosaic_cfg.get("damping", 0.01)),
        closure_features=closure_features,
        closure_probabilities=closure_probabilities,
    )
    bound = design.upper_bound(candidates["features"], candidates["probabilities"])
    token_mean = candidates["modality_tokens"].mean(dim=1)
    interaction_proxy = 1.0 - F.cosine_similarity(candidates["features"], token_mean, dim=1)
    work_positions, screening = _rank_union(
        bound,
        interaction_proxy,
        design_size=max(int(query_size), int(mosaic_cfg.get("workset_size", 5000))),
        synergy_size=int(mosaic_cfg.get("synergy_workset_size", 2500)),
    )
    work_tokens = candidates["modality_tokens"].index_select(0, work_positions)
    counterfactual = _counterfactual_features(
        classifier,
        work_tokens,
        candidates["modality_tokens"],
        partners=partners,
        rng=rng,
        lambda_alpha=lambda_alpha,
        fusion_batch_size=fusion_batch_size,
    )
    values, decomposition = _score_workset(
        design,
        classifier,
        counterfactual,
        eta=eta,
        value_batch_size=value_batch_size,
    )
    initial_values = values
    initial_decomposition = decomposition
    selected_work: list[int] = []
    selected_mask = torch.zeros(work_positions.numel(), dtype=torch.bool, device=work_positions.device)
    selection_count = min(int(query_size), int(work_positions.numel()))
    deflation_steps = min(
        selection_count,
        max(0, int(mosaic_cfg.get("deflation_steps", selection_count))),
    )
    for step in range(selection_count):
        current_scores = decomposition.score.masked_fill(selected_mask, float("-inf"))
        chosen = int(current_scores.argmax())
        selected_work.append(chosen)
        selected_mask[chosen] = True
        if step + 1 >= deflation_steps or step + 1 >= selection_count:
            if step + 1 < selection_count:
                remaining = torch.topk(
                    current_scores.masked_fill(selected_mask, float("-inf")),
                    k=selection_count - step - 1,
                    sorted=True,
                ).indices.tolist()
                selected_work.extend(int(value) for value in remaining)
            break
        candidate_position = work_positions[chosen]
        design.deflate(
            candidates["features"][candidate_position],
            candidates["probabilities"][candidate_position],
        )
        values, decomposition = _score_workset(
            design,
            classifier,
            counterfactual,
            eta=eta,
            value_batch_size=value_batch_size,
        )

    selected_work_tensor = torch.as_tensor(selected_work, dtype=torch.long, device=work_positions.device)
    selected_positions = work_positions.index_select(0, selected_work_tensor)
    candidate_count = int(candidates["features"].shape[0])
    floor = float(initial_decomposition.score.min()) - 1.0

    def expand(work_values: torch.Tensor) -> torch.Tensor:
        full = torch.full((candidate_count,), floor, dtype=torch.float32, device=work_values.device)
        full[work_positions] = work_values.float()
        return full

    combined = expand(initial_decomposition.score)
    higher_order = initial_decomposition.interactions[:, [3, 5, 6, 7]].abs().amax(dim=1)
    diagnostics = {
        "screening": screening,
        "partners": partners,
        "coalitions": 8,
        "workset_size": int(work_positions.numel()),
        "deflation_steps": deflation_steps,
        "empty_value_mean": float(values[:, 0].mean()),
        "higher_order_abs_mean": float(higher_order.mean()),
        "linearized_warning": bool(float(higher_order.mean()) < float(mosaic_cfg.get("synergy_epsilon", 1e-8))),
    }
    return MosaicAcquisitionResult(
        selected_positions=selected_positions,
        combined=combined,
        additive=expand(initial_decomposition.additive),
        synergy=expand(initial_decomposition.synergy),
        total_gain=expand(initial_values[:, -1] - initial_values[:, 0]),
        diagnostics=diagnostics,
    )
