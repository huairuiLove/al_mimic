"""First-party BRSET task plugin for repository orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import load_config
from .data import audit_prepared, audit_source, prepare_data
from .runner import BrsetActiveLearningExperiment


@dataclass(frozen=True, slots=True)
class BrsetTaskPlugin:
    """Stable BRSET task surface; acquisition behavior remains method-owned."""

    task_id: str = "brset"
    display_name: str = "BRSET multimodal retinal diagnosis"
    label_count: int = 13
    query_unit: str = "patient"
    actions: tuple[str, ...] = (
        "prepare",
        "validate-data",
        "active",
        "full-data",
        "hardware",
    )
    supported_methods: tuple[str, ...] = (
        "random",
        "comal",
        "modis",
        "mosaic",
    )
    capabilities: tuple[str, ...] = (
        "multilabel_probabilities",
        "modality_tokens",
        "token_fusion",
        "label_prototypes",
        "reference_labels",
    )

    def load_config(self, path: str | Path) -> dict[str, Any]:
        return load_config(path)

    def manifest(self, config: dict[str, Any]) -> dict[str, Any]:
        preprocessing = config.get("preprocessing", {})
        dataset = config.get("dataset", {})
        return {
            "task_id": self.task_id,
            "dataset_version": str(dataset.get("version", "")),
            "label_count": self.label_count,
            "query_unit": self.query_unit,
            "split_protocol": str(preprocessing.get("split_protocol", "")),
            "capabilities": self.capabilities,
        }

    def audit(self, config: dict[str, Any], *, prepared: bool = True) -> dict[str, Any]:
        audit = audit_prepared(config) if prepared else audit_source(config)
        return asdict(audit)

    def prepare(self, config: dict[str, Any]) -> dict[str, Any]:
        return prepare_data(config)

    def runner(self, config: dict[str, Any]) -> BrsetActiveLearningExperiment:
        return BrsetActiveLearningExperiment(config)

    def run(self, config: dict[str, Any]) -> dict[str, Any]:
        return self.runner(config).run()

    def run_full_data(self, config: dict[str, Any]) -> dict[str, Any]:
        return self.runner(config).run_full_data()

    def execute(self, action: str, config: dict[str, Any], **options: Any) -> dict[str, Any]:
        if action == "prepare":
            return self.prepare(config)
        if action == "validate-data":
            return self.audit(config, prepared=True)
        if action == "active":
            return self.run(config)
        if action == "full-data":
            return self.run_full_data(config)
        if action == "hardware":
            from al_mimic.utils.runtime import hardware_report

            return hardware_report(config)
        raise ValueError(f"unsupported BRSET action: {action}")


PLUGIN = BrsetTaskPlugin()


__all__ = ["BrsetTaskPlugin", "PLUGIN"]
