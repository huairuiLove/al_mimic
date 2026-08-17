"""Shared orchestration primitives for active-learning task runners."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from al_mimic.methods.api import AcquisitionResult


def initial_selection(candidate_ids: Sequence[Any], size: int, seed: int) -> list[Any]:
    if size < 1 or size > len(candidate_ids):
        raise ValueError("initial selection size is outside the query pool")
    generator = np.random.default_rng(seed)
    positions = generator.choice(len(candidate_ids), size=size, replace=False)
    return sorted((candidate_ids[int(position)] for position in positions), key=str)


def validate_acquisition(
    result: AcquisitionResult,
    candidate_ids: Sequence[Any],
    query_size: int,
) -> None:
    candidates = tuple(candidate_ids)
    if len(result.selected_ids) != query_size or len(result.selected_positions) != query_size:
        raise ValueError("method did not return the exact query budget")
    if len(set(result.selected_ids)) != query_size:
        raise ValueError("method selected query IDs must be unique")
    if len(set(result.selected_positions)) != query_size:
        raise ValueError("method selected positions must be unique")
    positioned = tuple(candidates[position] for position in result.selected_positions)
    if positioned != result.selected_ids:
        raise ValueError("method selected IDs do not match candidate-relative positions")


def numeric_score_summary(
    scores: Mapping[str, Any], selected_positions: Sequence[int]
) -> dict[str, dict[str, float]]:
    import torch

    positions = torch.as_tensor(tuple(selected_positions), dtype=torch.long)
    summary: dict[str, dict[str, float]] = {}
    for name, values in scores.items():
        if not isinstance(values, torch.Tensor) or values.ndim != 1:
            continue
        selected = values.index_select(0, positions.to(values.device))
        summary[str(name)] = {
            "pool_mean": float(values.float().mean()),
            "selected_mean": float(selected.float().mean()),
        }
    return summary


__all__ = ["initial_selection", "numeric_score_summary", "validate_acquisition"]
