"""Command-line interface for formal multimodal MIMIC-III active learning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .runtime import configure_runtime, hardware_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("prepare", "validate-data", "explore", "features", "active", "visualize", "hardware", "all"),
    )
    parser.add_argument("--config", default="configs/mimic_a800_144c.yaml")
    parser.add_argument("--name", help="override experiment.name")
    parser.add_argument("--output-root", help="override experiment.output_root")
    parser.add_argument("--prepared-dir", help="override dataset.prepared_dir")
    parser.add_argument("--experiment-dir", help="experiment directory for visualize")
    return parser


def _apply_overrides(config: dict, args: argparse.Namespace) -> None:
    experiment = config.setdefault("experiment", {})
    dataset = config.setdefault("dataset", {})
    if args.name:
        experiment["name"] = args.name
    if args.output_root:
        experiment["output_root"] = args.output_root
    if args.prepared_dir:
        dataset["prepared_dir"] = args.prepared_dir


def _validate(config: dict) -> dict:
    from .integrity import assert_original_unchanged
    from .multimodal_data import audit_split_hdf5

    audit = audit_split_hdf5(config)
    return {
        "records": audit.total_samples,
        "labels": audit.label_count,
        "split_counts": audit.split_counts,
        "time_steps": audit.time_steps,
        "time_series_dim": audit.time_series_dim,
        "time_invariant_dim": audit.time_invariant_dim,
        "note_tokens": audit.note_tokens,
        "original_comal_integrity": assert_original_unchanged(Path.cwd()),
    }


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    _apply_overrides(config, args)
    configure_runtime(config)
    command = args.command
    if command == "hardware":
        result = hardware_report(config)
    elif command == "prepare":
        from .multimodal_data import prepare_official_artifacts

        result = prepare_official_artifacts(config)
    elif command == "validate-data":
        result = _validate(config)
    elif command == "explore":
        from .visualization import explore_dataset

        result = explore_dataset(config)
    elif command == "features":
        from .multimodal_data import prepare_official_artifacts

        result = prepare_official_artifacts(config)
    elif command == "active":
        from .runner import ActiveLearningExperiment

        result = ActiveLearningExperiment(config).run()
    elif command == "visualize":
        from .visualization import visualize_experiment

        experiment = config.get("experiment", {})
        directory = args.experiment_dir or str(
            Path(experiment.get("output_root", "experiments")) / str(experiment.get("name", "mimic_comal"))
        )
        result = visualize_experiment(directory)
    else:
        from .multimodal_data import prepare_official_artifacts
        from .runner import ActiveLearningExperiment
        from .visualization import explore_dataset, visualize_experiment

        result = {"prepare": prepare_official_artifacts(config)}
        result["validation"] = _validate(config)
        result["exploration"] = explore_dataset(config)
        result["features"] = prepare_official_artifacts(config)
        result["active"] = ActiveLearningExperiment(config).run()
        result["visualization"] = visualize_experiment(result["active"]["output_dir"])
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
