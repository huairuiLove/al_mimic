"""Registered native multi-label tasks for the MIMIC-III runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class MimicTaskSpec:
    task_id: str
    display_name: str
    label_count: int
    label_format: str
    query_unit: str
    primary_metric: str
    metrics: tuple[str, ...]
    source_repositories: tuple[tuple[str, str], ...]


TASKS: dict[str, MimicTaskSpec] = {
    "icd9_diagnoses": MimicTaskSpec(
        task_id="icd9_diagnoses",
        display_name="MIMIC-III 48h three-digit ICD-9 diagnosis prediction",
        label_count=915,
        label_format="icd9_top3_multihot",
        query_unit="icu_stay",
        primary_metric="recall_at_30",
        metrics=("recall_at_10", "recall_at_20", "recall_at_30"),
        source_repositories=(("https://github.com/emnlp-mimic/mimic", "upstream"),),
    ),
    "phenotyping_25": MimicTaskSpec(
        task_id="phenotyping_25",
        display_name="MIMIC-III acute-care phenotyping (25 labels)",
        label_count=25,
        label_format="mimic3_benchmark_acute_care_multihot",
        query_unit="icu_stay",
        primary_metric="macro_auprc",
        metrics=("macro_auprc", "micro_auprc", "macro_auroc", "micro_auroc"),
        source_repositories=(
            (
                "https://github.com/YerevaNN/mimic3-benchmarks",
                "ea0314c7cbd369f62e2237ace6f683740f867e3a",
            ),
            (
                "https://github.com/kingrc15/multimodal-clinical-pretraining",
                "655c26a23880950cc270df5681b981e6869e26df",
            ),
        ),
    ),
    "phenotyping_ccs_172": MimicTaskSpec(
        task_id="phenotyping_ccs_172",
        display_name="MIMIC-III HCUP CCS phenotyping (172 labels)",
        label_count=172,
        label_format="hcup_ccs_2015_multihot_172",
        query_unit="icu_stay",
        primary_metric="macro_auprc",
        metrics=("macro_auprc", "micro_auprc", "macro_auroc", "micro_auroc"),
        source_repositories=(
            (
                "https://github.com/amoldwin/notes_benchmark",
                "fa378b828fb1f832635c4259c3dff97ab81bd19d",
            ),
        ),
    ),
}


def task_id_from_config(config: dict[str, Any]) -> str:
    task_id = str(config.get("task", {}).get("id", "icd9_diagnoses")).strip().lower()
    if task_id not in TASKS:
        raise ValueError(f"task.id must be one of {sorted(TASKS)}, got {task_id!r}")
    return task_id


def task_spec(config: dict[str, Any]) -> MimicTaskSpec:
    return TASKS[task_id_from_config(config)]


def task_manifest(config: dict[str, Any]) -> dict[str, Any]:
    spec = task_spec(config)
    return {
        "id": spec.task_id,
        "display_name": spec.display_name,
        "native_multilabel": True,
        "query_unit": spec.query_unit,
        "label_count": spec.label_count,
        "label_format": spec.label_format,
        "primary_metric": spec.primary_metric,
        "metrics": list(spec.metrics),
        "source_repositories": [
            {"url": url, "revision": revision} for url, revision in spec.source_repositories
        ],
    }
