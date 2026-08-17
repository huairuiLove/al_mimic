"""Resume helpers of the MIMIC-III active-learning loop."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from al_mimic.tasks.mimic_iii.runner import (
    load_round_progress,
    write_round_progress,
)

SCHEDULE = [718, 1077, 1436]


def _seed_progress(checkpoint_dir: Path, *, rounds_done: int = 2) -> None:
    initial = list(range(SCHEDULE[0]))
    labeled = list(range(SCHEDULE[rounds_done - 1]))
    for index in range(rounds_done):
        write_round_progress(
            checkpoint_dir,
            round_index=index,
            record={"round_index": index},
            labeled=labeled,
            initial=initial,
            schedule=SCHEDULE,
            strategy="comal",
            seed=17,
            rounds=len(SCHEDULE),
        )


def test_resume_restores_labeled_state_and_records(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    _seed_progress(checkpoint_dir, rounds_done=2)
    resumed = load_round_progress(
        checkpoint_dir,
        resume=True,
        schedule=SCHEDULE,
        strategy="comal",
        seed=17,
        rounds=3,
        train_indices=np.arange(5000),
    )
    assert resumed is not None
    assert resumed.rounds_done == 2
    assert resumed.labeled == list(range(1077))
    assert resumed.initial.tolist() == list(range(718))
    assert [record["round_index"] for record in resumed.records] == [0, 1]


@pytest.mark.parametrize(
    ("override", "message_part"),
    [
        ({"strategy": "modis"}, "strategy"),
        ({"seed": 42}, "seed"),
        ({"schedule": [700, 1000, 1300]}, "labeled schedule"),
        ({"rounds": 9}, "rounds"),
    ],
)
def test_resume_refuses_protocol_mismatch(tmp_path: Path, override: dict, message_part: str) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    _seed_progress(checkpoint_dir, rounds_done=2)
    expected = {
        "schedule": SCHEDULE,
        "strategy": "comal",
        "seed": 17,
        "rounds": 3,
    } | override
    with pytest.raises(ValueError, match=f"resume refused.*{message_part}"):
        load_round_progress(
            checkpoint_dir,
            resume=True,
            train_indices=np.arange(5000),
            **expected,
        )


def test_fresh_start_purges_stale_artifacts(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    _seed_progress(checkpoint_dir, rounds_done=2)
    (checkpoint_dir / "round_000.pt").touch()
    (checkpoint_dir / "final.pt").touch()
    (checkpoint_dir / "unrelated.txt").touch()
    resumed = load_round_progress(
        checkpoint_dir,
        resume=False,
        schedule=SCHEDULE,
        strategy="comal",
        seed=17,
        rounds=3,
        train_indices=np.arange(5000),
    )
    assert resumed is None
    assert not (checkpoint_dir / "progress.json").exists()
    assert not (checkpoint_dir / "round_000.pt").exists()
    assert not (checkpoint_dir / "final.pt").exists()
    assert not any(checkpoint_dir.glob("round_*_record.json"))
    assert (checkpoint_dir / "unrelated.txt").exists()


def test_resume_without_progress_starts_fresh(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    resumed = load_round_progress(
        checkpoint_dir,
        resume=True,
        schedule=SCHEDULE,
        strategy="comal",
        seed=17,
        rounds=3,
        train_indices=np.arange(5000),
    )
    assert resumed is None


def test_resume_refuses_labels_outside_train_pool(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    _seed_progress(checkpoint_dir, rounds_done=2)
    with pytest.raises(ValueError, match="outside the train pool"):
        load_round_progress(
            checkpoint_dir,
            resume=True,
            schedule=SCHEDULE,
            strategy="comal",
            seed=17,
            rounds=3,
            train_indices=np.arange(1000),
        )
