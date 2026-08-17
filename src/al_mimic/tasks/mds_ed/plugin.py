"""Native MDS-ED task plugin exposed to repository-level orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .audit import PreparedMemmapAudit, ReleaseAudit, audit_prepared_memmap, audit_release_csv
from .constants import (
    DIAGNOSIS_LABEL_COUNT,
    ECG_CHANNEL_COUNT,
    ECG_SAMPLE_COUNT,
    TEST_FOLDS,
    TRAIN_FOLDS,
    VALIDATION_FOLDS,
)
from .discovery import ReleasePaths, discover_release_inputs
from .prepare import prepare_release


@dataclass(frozen=True, slots=True)
class MdsEdTaskPlugin:
    task_id: str = "mds_ed"
    display_name: str = "MIMIC-IV MDS-ED diagnosis prediction"
    actions: tuple[str, ...] = ("prepare", "validate-data", "train", "hardware")
    capabilities: tuple[str, ...] = ()
    supported_methods: tuple[str, ...] = ()
    query_unit: str = "ecg_study"
    supervised_only: bool = True
    label_count: int = DIAGNOSIS_LABEL_COUNT
    waveform_samples: int = ECG_SAMPLE_COUNT
    waveform_channels: int = ECG_CHANNEL_COUNT
    train_folds: tuple[int, ...] = TRAIN_FOLDS
    validation_folds: tuple[int, ...] = VALIDATION_FOLDS
    test_folds: tuple[int, ...] = TEST_FOLDS

    def discover(self, search_root: str | Path | None = None) -> ReleasePaths:
        return discover_release_inputs(search_root)

    def audit_release(self, csv_path: str | Path) -> ReleaseAudit:
        return audit_release_csv(csv_path)

    def audit_prepared(self, prepared_dir: str | Path, expected_records: int) -> PreparedMemmapAudit:
        return audit_prepared_memmap(prepared_dir, expected_records)

    def prepare(
        self,
        paths: ReleasePaths | None = None,
        **options: Any,
    ) -> dict[str, Any]:
        return prepare_release(paths, **options)

    def train(self, prepared_dir: str | Path, output_dir: str | Path, config=None):
        # Keeping this import inside the method makes release audits usable on
        # machines without the optional supervised-training stack.
        from .training import train_supervised

        return train_supervised(prepared_dir, output_dir, config)

    def load_config(self, path: str | Path) -> dict[str, Any]:
        from al_mimic.utils.config import load_inherited_yaml

        return load_inherited_yaml(path)

    def execute(self, action: str, config: dict[str, Any], **options: Any) -> dict[str, Any]:
        from al_mimic.utils.config import resolve_config_path

        dataset = config.get("dataset", {})
        csv_override = options.get("release_csv")
        prepared_override = options.get("prepared_dir")
        csv_path = csv_override or dataset.get("release_csv")
        prepared_dir = prepared_override or dataset.get("prepared_dir")
        if csv_path is not None and csv_override is None:
            csv_path = resolve_config_path(config, csv_path)
        if prepared_dir is not None and prepared_override is None:
            prepared_dir = resolve_config_path(config, prepared_dir)
        if action == "validate-data":
            if csv_path is None:
                raise ValueError("MDS-ED validation requires dataset.release_csv")
            release = self.audit_release(csv_path)
            result: dict[str, Any] = {"release": release.to_dict()}
            if prepared_dir is not None:
                result["prepared"] = self.audit_prepared(prepared_dir, release.rows).to_dict()
            return result
        if action == "prepare":
            ecg_override = options.get("ecg_root")
            ecg_root = ecg_override or dataset.get("ecg_root")
            if ecg_root is not None and ecg_override is None:
                ecg_root = resolve_config_path(config, ecg_root)
            if csv_path is None or ecg_root is None or prepared_dir is None:
                raise ValueError("MDS-ED preparation requires release_csv, ecg_root, and prepared_dir")
            return self.prepare(
                ReleasePaths(Path(csv_path), Path(ecg_root), Path(prepared_dir)),
                resume=bool(options.get("resume", True)),
            )
        if action == "train":
            from .training import SupervisedTrainingConfig

            if prepared_dir is None:
                raise ValueError("MDS-ED training requires dataset.prepared_dir")
            output_override = options.get("output_dir")
            output_dir = output_override or config.get("experiment", {}).get(
                "output_dir", "../../../experiments/mds_ed/supervised"
            )
            if output_override is None:
                output_dir = resolve_config_path(config, output_dir)
            settings = SupervisedTrainingConfig(**config.get("training", {}))
            return self.train(prepared_dir, output_dir, settings)
        if action == "hardware":
            from al_mimic.utils.runtime import hardware_report

            return hardware_report(config)
        raise ValueError(f"unsupported MDS-ED action: {action}")


PLUGIN = MdsEdTaskPlugin()
