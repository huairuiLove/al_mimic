from __future__ import annotations

import numpy as np
import pytest

from mimic_comal.scenarios import (
    CLS_TOKEN_ID,
    PAD_TOKEN_ID,
    SEP_TOKEN_ID,
    empty_note,
    scenario_from_config,
)


def _cohort(n_train: int = 300, n_eval: int = 100, n_labels: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Labels with a deliberate head: column j appears with probability ~1/(j+1)."""
    rng = np.random.default_rng(7)
    total = n_train + 2 * n_eval
    prevalence = 1.0 / (np.arange(n_labels) + 1.0)
    labels = (rng.random((total, n_labels)) < prevalence).astype(np.float32)
    # Guarantee every visit carries at least one diagnosis, as the artifact does.
    labels[labels.sum(axis=1) == 0, 0] = 1.0
    split_names = np.concatenate(
        [
            np.full(n_train, "train", dtype=object),
            np.full(n_eval, "val", dtype=object),
            np.full(n_eval, "test", dtype=object),
        ]
    )
    return labels, split_names


def test_official_config_leaves_the_cohort_untouched() -> None:
    labels, split_names = _cohort()
    spec = scenario_from_config({}, labels=labels, split_names=split_names)
    assert not spec.active
    assert spec.label_count == labels.shape[1]
    assert spec.select_labels(labels) is labels
    assert spec.note_is_missing(0) is False


def test_dropping_the_head_removes_the_most_frequent_train_labels() -> None:
    labels, split_names = _cohort()
    config = {"scenario": {"label_subset": {"drop_top_k": 5}}}
    spec = scenario_from_config(config, labels=labels, split_names=split_names)

    assert spec.label_count == labels.shape[1] - 5
    train_frequency = labels[split_names == "train"].sum(axis=0)
    dropped = sorted(set(range(labels.shape[1])) - set(spec.label_columns.tolist()))
    assert dropped == sorted(np.argsort(-train_frequency)[:5].tolist())
    assert spec.select_labels(labels).shape == (labels.shape[0], labels.shape[1] - 5)


def test_label_ranking_ignores_evaluation_rows() -> None:
    """A head label must be chosen on train evidence alone, never on test rows."""
    labels, split_names = _cohort()
    poisoned = labels.copy()
    rare = int(np.argmin(labels[split_names == "train"].sum(axis=0)))
    poisoned[split_names != "train", rare] = 1.0

    config = {"scenario": {"label_subset": {"drop_top_k": 5}}}
    baseline = scenario_from_config(config, labels=labels, split_names=split_names)
    poisoned_spec = scenario_from_config(config, labels=poisoned, split_names=split_names)
    assert baseline.label_columns.tolist() == poisoned_spec.label_columns.tolist()


def test_minimum_train_positives_prunes_unlearnable_columns() -> None:
    labels, split_names = _cohort()
    config = {"scenario": {"label_subset": {"min_train_positives": 20}}}
    spec = scenario_from_config(config, labels=labels, split_names=split_names)

    train_frequency = labels[split_names == "train"].sum(axis=0)
    assert spec.label_columns.size < labels.shape[1]
    assert (train_frequency[spec.label_columns] >= 20).all()


def test_dropping_every_label_is_rejected() -> None:
    labels, split_names = _cohort()
    config = {"scenario": {"label_subset": {"drop_top_k": labels.shape[1]}}}
    with pytest.raises(ValueError, match="drop every"):
        scenario_from_config(config, labels=labels, split_names=split_names)


def test_missing_notes_hits_the_requested_rate_in_each_split() -> None:
    labels, split_names = _cohort()
    config = {"scenario": {"missing_notes": {"rate": 0.4, "seed": 3}}}
    spec = scenario_from_config(config, labels=labels, split_names=split_names)

    for split in ("train", "val", "test"):
        rows = split_names == split
        assert spec.notes_missing[rows].sum() == round(0.4 * rows.sum())


def test_missing_notes_respects_the_split_allowlist() -> None:
    labels, split_names = _cohort()
    config = {"scenario": {"missing_notes": {"rate": 0.4, "splits": ["train"], "seed": 3}}}
    spec = scenario_from_config(config, labels=labels, split_names=split_names)

    assert spec.notes_missing[split_names == "train"].any()
    assert not spec.notes_missing[split_names != "train"].any()


def test_missing_notes_is_reproducible_and_seed_sensitive() -> None:
    labels, split_names = _cohort()
    base = {"rate": 0.3, "bias": "label_sparse"}
    first = scenario_from_config(
        {"scenario": {"missing_notes": base | {"seed": 1}}},
        labels=labels,
        split_names=split_names,
    )
    repeat = scenario_from_config(
        {"scenario": {"missing_notes": base | {"seed": 1}}},
        labels=labels,
        split_names=split_names,
    )
    other = scenario_from_config(
        {"scenario": {"missing_notes": base | {"seed": 2}}},
        labels=labels,
        split_names=split_names,
    )
    assert np.array_equal(first.notes_missing, repeat.notes_missing)
    assert not np.array_equal(first.notes_missing, other.notes_missing)


def test_label_sparse_bias_concentrates_missingness_on_sparse_visits() -> None:
    labels, split_names = _cohort()
    counts = labels.sum(axis=1)
    biased = scenario_from_config(
        {"scenario": {"missing_notes": {"rate": 0.4, "bias": "label_sparse", "seed": 5}}},
        labels=labels,
        split_names=split_names,
    )
    uniform = scenario_from_config(
        {"scenario": {"missing_notes": {"rate": 0.4, "bias": "uniform", "seed": 5}}},
        labels=labels,
        split_names=split_names,
    )
    assert counts[biased.notes_missing].mean() < counts[uniform.notes_missing].mean()
    assert counts[biased.notes_missing].mean() < counts[~biased.notes_missing].mean()


def test_careunit_bias_requires_the_subgroup_array() -> None:
    labels, split_names = _cohort()
    config = {"scenario": {"missing_notes": {"rate": 0.3, "bias": "careunit", "careunit": 1}}}
    with pytest.raises(ValueError, match="careunit_code"):
        scenario_from_config(config, labels=labels, split_names=split_names)


def test_careunit_bias_favours_the_named_unit() -> None:
    labels, split_names = _cohort()
    rng = np.random.default_rng(11)
    careunits = rng.integers(0, 3, size=labels.shape[0])
    spec = scenario_from_config(
        {
            "scenario": {
                "missing_notes": {
                    "rate": 0.3,
                    "bias": "careunit",
                    "careunit": 1,
                    "strength": 9.0,
                    "seed": 4,
                }
            }
        },
        labels=labels,
        split_names=split_names,
        careunit_codes=careunits,
    )
    share_in_unit = (careunits[spec.notes_missing] == 1).mean()
    assert share_in_unit > (careunits == 1).mean()


def test_cohort_missingness_is_honoured_without_any_scenario() -> None:
    """A rebuilt cohort carries real gaps; they are part of the setting, not noise."""
    labels, split_names = _cohort()
    available = np.ones(labels.shape[0], dtype=bool)
    available[::4] = False

    spec = scenario_from_config(
        {}, labels=labels, split_names=split_names, notes_available=available
    )
    assert spec.active
    assert np.array_equal(spec.notes_missing, ~available)
    assert spec.summary()["missing_notes"]["source"] == "natural only"


def test_configured_rate_is_a_target_that_real_gaps_count_towards() -> None:
    labels, split_names = _cohort()
    available = np.ones(labels.shape[0], dtype=bool)
    train = split_names == "train"
    # A quarter of the train pool already has no note.
    available[np.flatnonzero(train)[::4]] = False

    spec = scenario_from_config(
        {"scenario": {"missing_notes": {"rate": 0.4, "seed": 2}}},
        labels=labels,
        split_names=split_names,
        notes_available=available,
    )
    assert spec.notes_missing[train].sum() == round(0.4 * train.sum())
    # Every genuinely absent note stays absent.
    assert spec.notes_missing[~available].all()
    accounting = spec.summary()["missing_notes"]["per_split"]["train"]
    assert accounting["natural"] > 0
    assert accounting["synthetic"] > 0


def test_a_cohort_already_past_the_target_gets_no_synthetic_withholding() -> None:
    labels, split_names = _cohort()
    available = np.ones(labels.shape[0], dtype=bool)
    train = np.flatnonzero(split_names == "train")
    available[train[: int(0.6 * train.size)]] = False

    spec = scenario_from_config(
        {"scenario": {"missing_notes": {"rate": 0.4, "splits": ["train"], "seed": 2}}},
        labels=labels,
        split_names=split_names,
        notes_available=available,
    )
    assert spec.summary()["missing_notes"]["per_split"]["train"]["synthetic"] == 0
    assert np.array_equal(spec.notes_missing, ~available)


def test_unknown_bias_is_rejected() -> None:
    labels, split_names = _cohort()
    config = {"scenario": {"missing_notes": {"rate": 0.3, "bias": "nonsense"}}}
    with pytest.raises(ValueError, match="bias must be one of"):
        scenario_from_config(config, labels=labels, split_names=split_names)


def test_rate_of_one_is_rejected() -> None:
    labels, split_names = _cohort()
    config = {"scenario": {"missing_notes": {"rate": 1.0}}}
    with pytest.raises(ValueError, match="below 1.0"):
        scenario_from_config(config, labels=labels, split_names=split_names)


def test_withheld_note_stays_a_valid_bert_sequence() -> None:
    note = empty_note(512)
    assert note["input_ids"][0] == CLS_TOKEN_ID
    assert note["input_ids"][1] == SEP_TOKEN_ID
    assert (note["input_ids"][2:] == PAD_TOKEN_ID).all()
    # An all-zero mask would divide by zero inside attention softmax.
    assert note["attention_mask"].sum() == 2
    assert not note["token_type_ids"].any()


def test_withheld_note_needs_room_for_both_special_tokens() -> None:
    with pytest.raises(ValueError, match="CLS"):
        empty_note(1)


def test_summary_records_what_the_scenario_changed() -> None:
    labels, split_names = _cohort()
    config = {
        "scenario": {
            "name": "combined",
            "label_subset": {"drop_top_k": 5},
            "missing_notes": {"rate": 0.4, "bias": "label_sparse", "seed": 8},
        }
    }
    summary = scenario_from_config(config, labels=labels, split_names=split_names).summary()

    assert summary["name"] == "combined"
    assert summary["label_count"] == labels.shape[1] - 5
    assert summary["label_subset"]["dropped_labels"] == 5
    assert summary["label_subset"]["dropped_positive_mass"] > 0.0
    assert summary["missing_notes"]["bias"] == "label_sparse"
    assert pytest.approx(summary["notes_missing_rate"], abs=0.01) == 0.4
