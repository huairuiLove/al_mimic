"""Command-line interface for formal BRSET multimodal active learning."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from mimic_comal.runtime import configure_runtime, hardware_report

from .config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "validate-data", "active", "hardware"))
    parser.add_argument("--config", default="configs/brset_comal.yaml")
    parser.add_argument("--name", help="override experiment.name")
    parser.add_argument("--output-root", help="override experiment.output_root")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    if args.name:
        config.setdefault("experiment", {})["name"] = args.name
    if args.output_root:
        config.setdefault("experiment", {})["output_root"] = args.output_root
    configure_runtime(config)
    if args.command == "prepare":
        from .data import prepare_data

        result = prepare_data(config)
    elif args.command == "validate-data":
        from .data import audit_prepared

        result = asdict(audit_prepared(config))
    elif args.command == "hardware":
        result = hardware_report(config)
    else:
        from .runner import BrsetActiveLearningExperiment

        result = BrsetActiveLearningExperiment(config).run()
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
