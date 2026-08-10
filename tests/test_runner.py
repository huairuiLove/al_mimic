from __future__ import annotations

import numpy as np

from mimic_comal.runner import _initial_indices, labeled_schedule


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
