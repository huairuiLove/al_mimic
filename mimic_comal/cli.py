"""Command-line interface for MIMIC-III CoMAL reproduction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .data import audit_records, load_records, prepare_mimic
from .features import build_features
from .integrity import assert_original_unchanged
from .runner import ActiveLearningExperiment
from .runtime import configure_runtime, hardware_report
from .visualization import explore_dataset, visualize_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("prepare", "validate-data", "explore", "features", "active", "visualize", "hardware", "all"),
    )
    parser.add_argument("--config", default="configs/mimic_comal.yaml")
    parser.add_argument("--name", help="override experiment.name")
    parser.add_argument("--output-root", help="override experiment.output_root")
    parser.add_argument("--device", help="override training.device")
    parser.add_argument("--rounds", type=int, help="override active_learning.rounds")
    parser.add_argument("--prepared-dir", help="override dataset.prepared_dir")
    parser.add_argument("--experiment-dir", help="experiment directory for visualize")
    return parser


def _apply_overrides(config: dict, args: argparse.Namespace) -> None:
    experiment = config.setdefault("experiment", {})
    training = config.setdefault("training", {})
    dataset = config.setdefault("dataset", {})
    if args.name:
        experiment["name"] = args.name
    if args.output_root:
        experiment["output_root"] = args.output_root
    if args.device:
        training["device"] = args.device
        config.setdefault("features", {})["device"] = args.device
    if args.rounds is not None:
        if args.rounds < 1:
            raise ValueError("--rounds must be positive")
        config.setdefault("active_learning", {})["rounds"] = args.rounds
    if args.prepared_dir:
        dataset["prepared_dir"] = args.prepared_dir
        dataset["feature_dir"] = str(Path(args.prepared_dir) / "features")


def _validate(config: dict) -> dict:
    prepared = Path(config.get("dataset", {}).get("prepared_dir", "prepared/mimic_iii"))
    records = load_records(prepared)
    labels = tuple(json.loads((prepared / "labels.json").read_text(encoding="utf-8"))["labels"])
    audit = audit_records(records, labels)
    if audit["group_leakage"]:
        raise RuntimeError("SUBJECT_ID leakage detected between splits")
    return {
        "prepared_dir": str(prepared),
        "records": len(records),
        "labels": len(labels),
        "audit": audit,
        "original_comal_integrity": assert_original_unchanged(Path.cwd()),
    }


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    _apply_overrides(config, args)
    configure_runtime(config)
    command = args.command
    if command == "prepare":
        result = prepare_mimic(config)
    elif command == "validate-data":
        result = _validate(config)
    elif command == "explore":
        result = explore_dataset(config)
    elif command == "features":
        result = build_features(config)
    elif command == "active":
        result = ActiveLearningExperiment(config).run()
    elif command == "visualize":
        experiment = config.get("experiment", {})
        directory = args.experiment_dir or str(
            Path(experiment.get("output_root", "experiments")) / str(experiment.get("name", "mimic_comal"))
        )
        result = visualize_experiment(directory)
    elif command == "hardware":
        result = hardware_report(config)
    else:
        result = {"prepare": prepare_mimic(config)}
        result["validation"] = _validate(config)
        result["exploration"] = explore_dataset(config)
        result["features"] = build_features(config)
        result["active"] = ActiveLearningExperiment(config).run()
        result["visualization"] = visualize_experiment(result["active"]["output_dir"])
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
