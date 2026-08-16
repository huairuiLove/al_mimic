"""On-manifold modality-level mixup interventions."""

from __future__ import annotations

import numpy as np
import torch


def coalition_masks(num_modalities: int) -> torch.Tensor:
    if num_modalities < 1:
        raise ValueError("num_modalities must be positive")
    masks = [
        [(coalition >> modality) & 1 == 1 for modality in range(num_modalities)]
        for coalition in range(1 << num_modalities)
    ]
    return torch.tensor(masks, dtype=torch.bool)


def intervene_tokens(
    base_tokens: torch.Tensor,
    partner_pool: torch.Tensor,
    keep_mask: torch.Tensor,
    *,
    partners: int,
    rng: np.random.Generator,
    lambda_alpha: float | None = None,
) -> torch.Tensor:
    """Keep coalition modalities and independently mix the remaining modalities."""
    if base_tokens.ndim != 3 or partner_pool.ndim != 3:
        raise ValueError("base_tokens and partner_pool must have shape [N,M,D]")
    if base_tokens.shape[1:] != partner_pool.shape[1:]:
        raise ValueError("base and partner modality shapes must match")
    if keep_mask.shape != (base_tokens.shape[1],):
        raise ValueError("keep_mask must have one entry per modality")
    partner_count = max(1, int(partners))
    result = base_tokens[:, None, :, :].expand(-1, partner_count, -1, -1).clone()
    for modality in range(base_tokens.shape[1]):
        if bool(keep_mask[modality]):
            continue
        sampled = rng.integers(0, partner_pool.shape[0], size=(base_tokens.shape[0], partner_count))
        sampled_index = torch.as_tensor(sampled, dtype=torch.long, device=partner_pool.device)
        replacement = (
            partner_pool[:, modality]
            .index_select(0, sampled_index.reshape(-1))
            .reshape(base_tokens.shape[0], partner_count, -1)
        )
        if lambda_alpha is None or float(lambda_alpha) <= 0.0:
            mixed = replacement
        else:
            lambdas = rng.beta(
                float(lambda_alpha),
                float(lambda_alpha),
                size=(base_tokens.shape[0], partner_count, 1),
            )
            mixing = torch.as_tensor(lambdas, dtype=base_tokens.dtype, device=base_tokens.device)
            original = base_tokens[:, None, modality, :]
            mixed = (1.0 - mixing) * original + mixing * replacement
        result[:, :, modality, :] = mixed
    return result
