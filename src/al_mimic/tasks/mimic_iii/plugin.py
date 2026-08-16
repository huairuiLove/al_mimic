"""Native MIMIC-III task-family plugin for repository orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .config import load_config
from .data import audit_split_hdf5, prepare_official_artifacts
from .runner import ActiveLearningExperiment
from .tasks import TASKS, task_manifest


@dataclass(frozen=True, slots=True)
class MimicIIITaskPlugin:
    """Stable task-family surface; acquisition remains method-owned."""

    task_id: str = "mimic_iii"
    display_name: str = "MIMIC-III native multi-label tasks"
    task_ids: tuple[str, ...] = tuple(TASKS)
    actions: tuple[str, ...] = (
        "prepare",
        "validate-data",
        "explore",
        "active",
        "full-data",
        "visualize",
        "hardware",
    )
    supported_methods: tuple[str, ...] = (
        "random",
        "comal",
        "mm_comal",
        "modis",
        "mosaic",
    )
    query_unit: str = "icu_stay"
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
        return task_manifest(config)

    def audit(self, config: dict[str, Any]) -> dict[str, Any]:
        return asdict(audit_split_hdf5(config))

    def prepare(
        self,
        config: dict[str, Any],
        output_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        return prepare_official_artifacts(config, output_dir)

    def runner(self, config: dict[str, Any]) -> ActiveLearningExperiment:
        return ActiveLearningExperiment(config)

    def run(self, config: dict[str, Any]) -> dict[str, Any]:
        return self.runner(config).run()

    def run_full_data(self, config: dict[str, Any]) -> dict[str, Any]:
        return self.runner(config).run_full_data()

    def execute(self, action: str, config: dict[str, Any], **options: Any) -> dict[str, Any]:
        if action == "prepare":
            return self.prepare(config, options.get("output_dir"))
        if action == "validate-data":
            return self.audit(config)
        if action == "explore":
            from .visualization import explore_dataset

            return explore_dataset(config, options.get("output_dir"))
        if action == "active":
            return self.run(config)
        if action == "full-data":
            return self.run_full_data(config)
        if action == "visualize":
            from .visualization import visualize_experiment

            directory = options.get("experiment_dir")
            if directory is None:
                from .config import resolve_path

                experiment = config.get("experiment", {})
                directory = resolve_path(config, experiment.get("output_root", "../../../experiments")) / str(
                    experiment.get("name", "mimic_iii")
                )
            return visualize_experiment(directory)
        if action == "hardware":
            from al_mimic.utils.runtime import hardware_report

            return hardware_report(config)
        raise ValueError(f"unsupported MIMIC-III action: {action}")


PLUGIN = MimicIIITaskPlugin()


__all__ = ["MimicIIITaskPlugin", "PLUGIN"]
