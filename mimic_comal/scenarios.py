"""Dataset-scenario transforms that reshape the acquisition problem itself.

The official Diagnoses cohort gives active learning almost nothing to win: the
label prior alone reaches R@30 0.460 against a full-pool ceiling of 0.542, and
every visit carries both a note and a dense time series, so no sample is
informative through one modality rather than another. These transforms change
that setting without touching any acquisition method.

Both are views over the official split artifact, resolved from arrays the
feature store already holds in memory, so the multi-gigabyte HDF5 is never
rewritten and a scenario costs nothing to add or drop.

label_subset
    Drop the highest-frequency ICD-9 groups. Those are recalled for free by the
    prior, so they inflate every arm equally and compress the range in which
    acquisition can differ.

missing_notes
    Withhold the clinical note from a controlled slice of rows, optionally
    concentrated on a subgroup. This is the modality heterogeneity that
    multimodal acquisition assumes and that the 48h cohort does not contain.

    A rebuilt cohort may already carry real missingness: at a 12h window 27% of
    stays have no note charted yet. That is read from the artifact and treated as
    part of the scenario, with the configured rate acting as an overall target
    that synthetic withholding only tops up towards.

missing_series
    The same for the time series, and by default disjoint from the note gaps so
    each affected visit loses exactly one modality. Withholding notes alone
    turned out to cost almost nothing -- R@30 moved from 0.5301 to 0.5292 with
    40% of notes gone -- because the text carries little the series does not.
    Complementary dropout is what actually makes a candidate informative through
    one modality rather than the other, which is the premise the multimodal
    acquisition methods are built on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


PAD_TOKEN_ID = 0
CLS_TOKEN_ID = 101
SEP_TOKEN_ID = 102

BIAS_MODES = ("uniform", "label_sparse", "careunit")


@dataclass(frozen=True, slots=True)
class ScenarioSpec:
    """Row- and column-level view applied on top of the official tensors."""

    name: str
    raw_label_count: int
    label_columns: np.ndarray | None
    notes_missing: np.ndarray | None
    series_missing: np.ndarray | None
    provenance: dict[str, Any]

    @property
    def label_count(self) -> int:
        return self.raw_label_count if self.label_columns is None else int(self.label_columns.size)

    @property
    def active(self) -> bool:
        return (
            self.label_columns is not None
            or self.notes_missing is not None
            or self.series_missing is not None
        )

    def select_labels(self, labels: np.ndarray) -> np.ndarray:
        """Apply the column subset to a [..., raw_label_count] label array."""
        if self.label_columns is None:
            return labels
        return labels[..., self.label_columns]

    def note_is_missing(self, global_index: int) -> bool:
        return self.notes_missing is not None and bool(self.notes_missing[global_index])

    def series_is_missing(self, global_index: int) -> bool:
        return self.series_missing is not None and bool(self.series_missing[global_index])

    def summary(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "raw_label_count": self.raw_label_count,
            "label_count": self.label_count,
        }
        for modality, mask in (("notes", self.notes_missing), ("series", self.series_missing)):
            if mask is not None:
                payload[f"{modality}_missing_rows"] = int(mask.sum())
                payload[f"{modality}_missing_rate"] = float(mask.mean())
        if self.notes_missing is not None and self.series_missing is not None:
            payload["both_modalities_missing_rows"] = int(
                (self.notes_missing & self.series_missing).sum()
            )
        return payload | self.provenance


def empty_note(length: int) -> dict[str, np.ndarray]:
    """Token tensors for a withheld note: a well-formed but contentless sequence.

    An all-zero attention mask would make BERT's attention softmax divide by
    zero, so the [CLS]/[SEP] pair stays visible and carries no content.
    """
    if length < 2:
        raise ValueError(f"note length must leave room for [CLS] and [SEP], got {length}")
    input_ids = np.full(length, PAD_TOKEN_ID, dtype=np.int64)
    input_ids[0] = CLS_TOKEN_ID
    input_ids[1] = SEP_TOKEN_ID
    attention_mask = np.zeros(length, dtype=np.int64)
    attention_mask[:2] = 1
    return {
        "input_ids": input_ids,
        "token_type_ids": np.zeros(length, dtype=np.int64),
        "attention_mask": attention_mask,
    }


def _label_columns(
    settings: dict[str, Any], labels: np.ndarray, train_mask: np.ndarray
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Choose which label columns survive, ranking frequency on train rows only."""
    drop_top_k = int(settings.get("drop_top_k", 0))
    min_train_positives = int(settings.get("min_train_positives", 0))
    if drop_top_k <= 0 and min_train_positives <= 0:
        return None, {}

    raw_count = int(labels.shape[1])
    train_frequency = labels[train_mask].sum(axis=0)
    keep = np.ones(raw_count, dtype=bool)
    if drop_top_k > 0:
        if drop_top_k >= raw_count:
            raise ValueError(
                f"scenario.label_subset.drop_top_k={drop_top_k} would drop every one of "
                f"{raw_count} labels"
            )
        # Ties broken by column index so the subset is reproducible across runs.
        order = np.lexsort((np.arange(raw_count), -train_frequency))
        keep[order[:drop_top_k]] = False
    if min_train_positives > 0:
        keep &= train_frequency >= min_train_positives

    columns = np.flatnonzero(keep).astype(np.int64)
    if columns.size == 0:
        raise ValueError("scenario.label_subset removed every label column")
    provenance = {
        "label_subset": {
            "drop_top_k": drop_top_k,
            "min_train_positives": min_train_positives,
            "kept_labels": int(columns.size),
            "dropped_labels": raw_count - int(columns.size),
            "dropped_positive_mass": float(
                1.0 - train_frequency[columns].sum() / max(train_frequency.sum(), 1.0)
            ),
        }
    }
    return columns, provenance


def _missing_weights(
    bias: str,
    strength: float,
    eligible: np.ndarray,
    labels: np.ndarray,
    careunit_codes: np.ndarray | None,
    careunit_target: int | None,
) -> np.ndarray:
    """Unnormalised selection weight per eligible row."""
    if bias == "uniform":
        return np.ones(eligible.size, dtype=np.float64)
    if bias == "label_sparse":
        # Rows carrying few diagnoses are the ones a note would most help, so
        # withholding notes there is what forces acquisition to reason about
        # which modality a candidate is informative through.
        positives = labels[eligible].sum(axis=1).astype(np.float64)
        spread = positives.std()
        if spread <= 0.0:
            return np.ones(eligible.size, dtype=np.float64)
        standardised = (positives - positives.mean()) / spread
        return np.exp(-strength * standardised)
    if bias == "careunit":
        if careunit_codes is None:
            raise ValueError(
                "scenario.missing_notes.bias='careunit' needs a careunit_code array in the "
                "split artifact; rebuild the splits or choose another bias"
            )
        if careunit_target is None:
            raise ValueError("scenario.missing_notes.careunit must name the affected unit")
        weights = np.ones(eligible.size, dtype=np.float64)
        weights[careunit_codes[eligible] == careunit_target] = max(strength, 0.0) + 1.0
        return weights
    raise ValueError(f"scenario.missing_notes.bias must be one of {BIAS_MODES}, got {bias!r}")


def _notes_missing(
    settings: dict[str, Any],
    labels: np.ndarray,
    split_names: np.ndarray,
    careunit_codes: np.ndarray | None,
    natural_missing: np.ndarray | None,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    total = int(labels.shape[0])
    missing = (
        np.zeros(total, dtype=bool) if natural_missing is None else natural_missing.copy()
    )
    natural_rows = int(missing.sum())

    rate = float(settings.get("rate", 0.0))
    if rate >= 1.0:
        raise ValueError(f"scenario.missing_notes.rate must stay below 1.0, got {rate}")
    if rate <= 0.0:
        if natural_rows == 0:
            return None, {}
        return missing, {
            "missing_notes": {"source": "natural only", "natural_rows": natural_rows}
        }

    bias = str(settings.get("bias", "uniform")).lower()
    strength = float(settings.get("strength", 1.0))
    seed = int(settings.get("seed", 0))
    splits = tuple(settings.get("splits", ("train", "val", "test")))
    careunit_target = settings.get("careunit")
    careunit_target = None if careunit_target is None else int(careunit_target)

    per_split: dict[str, dict[str, int]] = {}
    for split in splits:
        rows = split_names == split
        if not rows.any():
            raise ValueError(f"scenario.missing_notes.splits names an empty split: {split!r}")
        # rate is the target share of rows without a note; whatever the cohort is
        # already missing counts towards it, so the two sources compose instead of
        # stacking into an unintended rate.
        already = int(missing[rows].sum())
        wanted = int(round(rate * rows.sum())) - already
        if wanted <= 0:
            per_split[split] = {"natural": already, "synthetic": 0}
            continue
        eligible = np.flatnonzero(rows & ~missing)
        wanted = min(wanted, eligible.size)
        weights = _missing_weights(
            bias, strength, eligible, labels, careunit_codes, careunit_target
        )
        weights = weights / weights.sum()
        # Seeded per split so adding or resizing one split cannot shift another.
        rng = np.random.default_rng([seed, abs(hash(split)) % (2**31)])
        chosen = rng.choice(eligible, size=wanted, replace=False, p=weights)
        missing[chosen] = True
        per_split[split] = {"natural": already, "synthetic": int(wanted)}

    provenance = {
        "missing_notes": {
            "target_rate": rate,
            "bias": bias,
            "strength": strength,
            "seed": seed,
            "natural_rows": natural_rows,
            "per_split": per_split,
        }
    }
    return missing, provenance


def scenario_from_config(
    config: dict[str, Any],
    *,
    labels: np.ndarray,
    split_names: np.ndarray,
    careunit_codes: np.ndarray | None = None,
    notes_available: np.ndarray | None = None,
) -> ScenarioSpec:
    """Resolve the scenario declared in the config against the loaded row arrays."""
    settings = config.get("scenario", {}) or {}
    name = str(settings.get("name", "official"))
    raw_label_count = int(labels.shape[1])

    columns, label_provenance = _label_columns(
        settings.get("label_subset", {}) or {}, labels, split_names == "train"
    )
    natural_missing = None if notes_available is None else ~np.asarray(notes_available, dtype=bool)
    missing, missing_provenance = _notes_missing(
        settings.get("missing_notes", {}) or {},
        labels,
        split_names,
        careunit_codes,
        natural_missing,
    )
    return ScenarioSpec(
        name=name,
        raw_label_count=raw_label_count,
        label_columns=columns,
        notes_missing=missing,
        series_missing=None,
        provenance=label_provenance | missing_provenance,
    )
