from __future__ import annotations

import numpy as np

from mimic_comal.runner import ActiveLearningExperiment, _initial_indices, labeled_schedule


class _RandomOnly:
    """Exercise the random control without constructing the CUDA experiment."""

    seed = 17
    _acquire_random = ActiveLearningExperiment._acquire_random


def test_official_train_split_gets_exact_six_round_fraction_schedule() -> None:
    schedule = labeled_schedule(7147)
    assert schedule == [715, 1072, 1429, 1787, 2144, 2501]
    assert len(schedule) == 6


def test_initial_sample_is_reproducible_and_label_blind() -> None:
    train = np.arange(7147, dtype=np.int64)
    first = _initial_indices(train, 715, 17)
    second = _initial_indices(train, 715, 17)
    assert first == second
    assert len(first) == len(set(first)) == 715


def test_random_control_draws_the_exact_unique_budget_from_the_pool() -> None:
    candidates = np.arange(1000, 7000, dtype=np.int64)
    queries, meta = _RandomOnly._acquire_random(_RandomOnly(), candidates, 357, 0)
    assert len(queries) == len(set(queries)) == 357
    assert set(queries) <= set(candidates.tolist())
    assert meta["candidate_count"] == candidates.size
    assert queries == sorted(queries)


def test_random_control_is_reproducible_but_varies_across_rounds() -> None:
    candidates = np.arange(1000, 7000, dtype=np.int64)
    first, _ = _RandomOnly._acquire_random(_RandomOnly(), candidates, 357, 0)
    repeat, _ = _RandomOnly._acquire_random(_RandomOnly(), candidates, 357, 0)
    later, _ = _RandomOnly._acquire_random(_RandomOnly(), candidates, 357, 3)
    assert first == repeat
    assert first != later
