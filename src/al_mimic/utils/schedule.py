"""Task-independent active-learning budget schedules."""

from __future__ import annotations

import math


def labeled_schedule(
    pool_size: int,
    *,
    rounds: int = 6,
    initial_fraction: float = 0.10,
    query_fraction: float = 0.05,
) -> list[int]:
    """Return exact cumulative query-unit targets with half-up rounding."""
    if pool_size < 1 or rounds < 1:
        raise ValueError("pool_size and rounds must be positive")
    if not 0 < initial_fraction <= 1 or query_fraction < 0:
        raise ValueError("active-learning fractions must be positive and bounded")
    targets = [
        int(math.floor(pool_size * (initial_fraction + query_fraction * index) + 0.5))
        for index in range(rounds)
    ]
    if targets[0] < 1 or targets[-1] > pool_size:
        raise ValueError("active-learning fractions exceed the query pool")
    if any(right <= left for left, right in zip(targets, targets[1:])):
        raise ValueError("active-learning fractions do not produce an increasing schedule")
    return targets


__all__ = ["labeled_schedule"]
