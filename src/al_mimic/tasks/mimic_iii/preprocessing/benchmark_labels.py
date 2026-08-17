"""First-party phenotyping label construction over HCUP CCS 2015 groups.

Both MIMIC-III phenotyping tasks label one ICU stay with the CCS groups of its
hospital admission's ICD-9 codes:

``benchmark_25``
    The 25 acute-care groups the MIMIC-III Benchmark marks
    ``use_in_benchmark`` in its CCS definitions, in alphabetical group order.
``ccs_239``
    The notes-benchmark paper's stated rule: every CCS group that occurs in
    at least 30 episodes of the final cohort, ordered by HCUP CCS id then
    name. The paper reports 172 groups from this rule but never enumerates
    them, and its released code selects only the 25 ``use_in_benchmark``
    groups; on this repository's cohort the stated rule yields 239, which is
    the width the builder enforces.

The definitions YAML is a checked-in materialised copy of the benchmark's
``hcup_ccs_2015_definitions.yaml``; the mapping is code-exact, including codes
that only differ by trailing digits.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

RESOURCES = Path(__file__).with_name("resources")
DEFINITIONS_YAML = RESOURCES / "hcup_ccs_2015_definitions.yaml"
DEFAULT_MINIMUM_EPISODES = 30
EXPECTED_CCS_LABELS = 239


def load_ccs_definitions(path: str | Path = DEFINITIONS_YAML) -> dict[str, dict]:
    definitions = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(definitions, dict) or not definitions:
        raise ValueError(f"CCS definitions file is empty or malformed: {path}")
    for group, definition in definitions.items():
        if "codes" not in definition or "id" not in definition:
            raise ValueError(f"CCS group {group!r} lacks 'codes' or 'id' in {path}")
    return definitions


def _code_maps(definitions: dict[str, dict]) -> tuple[dict[str, str], dict[str, int], dict[str, bool]]:
    code_to_group: dict[str, str] = {}
    group_ids: dict[str, int] = {}
    use_in_benchmark: dict[str, bool] = {}
    for group, definition in definitions.items():
        group = str(group)
        group_ids[group] = int(definition["id"])
        use_in_benchmark[group] = bool(definition.get("use_in_benchmark", False))
        for code in definition["codes"]:
            code = str(code)
            previous = code_to_group.get(code)
            if previous is not None and previous != group:
                raise ValueError(f"ICD-9 code {code!r} maps to both {previous!r} and {group!r}")
            code_to_group[code] = group
    duplicate_ids = [group_id for group_id, count in Counter(group_ids.values()).items() if count > 1]
    if duplicate_ids:
        raise ValueError(f"duplicate HCUP CCS ids in definitions: {duplicate_ids}")
    return code_to_group, group_ids, use_in_benchmark


def attach_ccs_groups(diagnoses: pd.DataFrame, definitions: dict[str, dict]) -> pd.DataFrame:
    """Add HCUP_CCS_2015 and USE_IN_BENCHMARK columns to an ICD-9 diagnoses frame."""
    code_to_group, _, use_in_benchmark = _code_maps(definitions)
    groups = diagnoses["ICD9_CODE"].astype(str).map(code_to_group)
    # Series.map yields NaN (not None) for unmapped codes; collapse both.
    return diagnoses.assign(
        HCUP_CCS_2015=groups,
        USE_IN_BENCHMARK=groups.map(
            lambda group: None if group is None or pd.isna(group) else use_in_benchmark[group]
        ),
    )


def benchmark_label_names(definitions: dict[str, dict]) -> list[str]:
    """The 25 benchmark groups, alphabetically as the benchmark orders them."""
    _, _, use_in_benchmark = _code_maps(definitions)
    return sorted(group for group, used in use_in_benchmark.items() if used)


def _stay_groups(diagnoses: pd.DataFrame, allowed: set[str] | None) -> dict[int, set[str]]:
    grouped: dict[int, set[str]] = {}
    rows = diagnoses[diagnoses["HCUP_CCS_2015"].notna()]
    if allowed is not None:
        rows = rows[rows["HCUP_CCS_2015"].isin(allowed)]
    for stay, group in zip(rows["ICUSTAY_ID"].to_numpy(), rows["HCUP_CCS_2015"].to_numpy(), strict=True):
        grouped.setdefault(int(stay), set()).add(str(group))
    return grouped


def benchmark_25_labels(
    stays: pd.DataFrame, diagnoses: pd.DataFrame, definitions: dict[str, dict]
) -> tuple[np.ndarray, list[str]]:
    """Multi-hot 25-label matrix aligned with ``stays`` row order."""
    names = benchmark_label_names(definitions)
    index = {name: position for position, name in enumerate(names)}
    grouped = _stay_groups(diagnoses, allowed=set(names))
    labels = np.zeros((len(stays), len(names)), dtype=np.uint8)
    for row, stay in enumerate(stays["ICUSTAY_ID"].to_numpy()):
        for group in grouped.get(int(stay), ()):
            labels[row, index[group]] = 1
    return labels, names


def ccs_239_labels(
    stays: pd.DataFrame,
    diagnoses: pd.DataFrame,
    definitions: dict[str, dict],
    *,
    minimum_episodes: int = DEFAULT_MINIMUM_EPISODES,
    expected_labels: int = EXPECTED_CCS_LABELS,
) -> tuple[np.ndarray, list[str]]:
    """Multi-hot CCS label matrix under the >=30-episode rule (239 labels)."""
    _, group_ids, _ = _code_maps(definitions)
    grouped = _stay_groups(diagnoses, allowed=None)
    counts = Counter(group for groups in grouped.values() for group in groups)
    names = [
        group
        for group in sorted(counts, key=lambda name: (group_ids[name], name))
        if counts[group] >= minimum_episodes
    ]
    if len(names) != expected_labels:
        raise ValueError(
            f"paper rule selected {len(names)} CCS labels, expected {expected_labels}; "
            "verify the cohort reproduces the authors' filtered population"
        )
    index = {name: position for position, name in enumerate(names)}
    labels = np.zeros((len(stays), len(names)), dtype=np.uint8)
    for row, stay in enumerate(stays["ICUSTAY_ID"].to_numpy()):
        for group in grouped.get(int(stay), ()):
            if group in index:
                labels[row, index[group]] = 1
    return labels, names
