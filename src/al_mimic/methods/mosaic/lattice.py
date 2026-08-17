"""Exact modality information-gain lattice for small modality counts."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LatticeDecomposition:
    interactions: torch.Tensor
    additive: torch.Tensor
    synergy: torch.Tensor
    score: torch.Tensor


def mobius_inversion(values: torch.Tensor, *, num_modalities: int) -> torch.Tensor:
    expected = 1 << int(num_modalities)
    if values.ndim != 2 or values.shape[1] != expected:
        raise ValueError(f"values must have shape [N,{expected}]")
    interactions = torch.zeros_like(values)
    for coalition in range(expected):
        subset = coalition
        while True:
            sign = -1.0 if ((coalition.bit_count() - subset.bit_count()) % 2) else 1.0
            interactions[:, coalition] += sign * values[:, subset]
            if subset == 0:
                break
            subset = (subset - 1) & coalition
    return interactions


def decompose_lattice(
    values: torch.Tensor,
    *,
    num_modalities: int,
    eta: float,
) -> LatticeDecomposition:
    interactions = mobius_inversion(values, num_modalities=num_modalities)
    singleton_indices = [1 << modality for modality in range(num_modalities)]
    additive = interactions[:, singleton_indices].sum(dim=1)
    higher_order = [coalition for coalition in range(1 << num_modalities) if coalition.bit_count() >= 2]
    synergy = interactions[:, higher_order].sum(dim=1)
    score = synergy.clamp_min(0.0) + float(eta) * additive
    return LatticeDecomposition(interactions, additive, synergy, score)
