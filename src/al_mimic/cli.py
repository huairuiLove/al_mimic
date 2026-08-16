"""Unified command-line entry point for all first-party tasks and methods."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from al_mimic.core.capabilities import (
    capability_record,
    require_action,
    require_method,
)
from al_mimic.evaluation import build_matrix, write_matrix
from al_mimic.methods import available_methods, get_method
from al_mimic.tasks.registry import available_tasks, get_task
from al_mimic.utils.io import jsonable
from al_mimic.utils.runtime import configure_runtime

_RUN_ACTIONS = (
    "prepare",
    "validate-data",
    "explore",
    "active",
    "full-data",
    "visualize",
    "train",
    "hardware",
)


def _add_run_parser(subparsers: Any, action: str) -> None:
    parser = subparsers.add_parser(action, help=f"run the {action} task action")
    parser.add_argument("--task", required=True, choices=available_tasks())
    parser.add_argument("--config", type=Path, required=action != "hardware")
    parser.add_argument("--method", choices=available_methods())
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--experiment-dir", type=Path)
    parser.add_argument("--prepared-dir", type=Path)
    parser.add_argument("--release-csv", type=Path)
    parser.add_argument("--ecg-root", type=Path)
    parser.add_argument("--no-resume", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("tasks", help="list registered task plugins")
    subparsers.add_parser("methods", help="list registered acquisition methods")
    subparsers.add_parser("capabilities", help="show task and method compatibility")
    for action in _RUN_ACTIONS:
        _add_run_parser(subparsers, action)
    matrix = subparsers.add_parser("matrix", help="build a cross-experiment metric matrix")
    matrix.add_argument("--experiment", action="append", required=True)
    matrix.add_argument("--output", type=Path, default=Path("experiments/evaluation_matrix.csv"))
    return parser


def _load_config(task_plugin: Any, path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    config = task_plugin.load_config(path)
    if not isinstance(config, dict):
        raise TypeError("task load_config() must return a mapping")
    family = config.get("task", {}).get("family")
    if family is not None and str(family) != str(task_plugin.task_id):
        raise ValueError(
            f"configuration task.family={family!r} does not match --task {task_plugin.task_id!r}"
        )
    return config


def _validate_method(
    task_plugin: Any,
    config: dict[str, Any],
    action: str,
    requested_method: str | None,
) -> Any | None:
    configured = config.get("active_learning", {}).get("strategy")
    if action != "active":
        if requested_method is not None:
            raise ValueError(f"--method is only valid for the active action, not {action!r}")
        return None
    if requested_method is None:
        raise ValueError("active requires an explicit --method")
    if configured is None:
        raise ValueError("active configuration must declare active_learning.strategy")
    normalized = str(configured).strip().lower().replace("-", "_")
    if normalized != requested_method:
        raise ValueError(
            f"--method {requested_method!r} does not match resolved active_learning.strategy={configured!r}"
        )
    plugin = get_method(requested_method)
    require_method(task_plugin, plugin)
    return plugin


def _capability_matrix() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    methods = available_methods()
    for task_name in available_tasks():
        task = get_task(task_name)
        record = capability_record(task)
        supported = set(record["methods"])
        record["method_support"] = {method: method in supported for method in methods}
        rows.append(record)
    return rows


def dispatch(args: argparse.Namespace) -> Any:
    if args.command == "tasks":
        return [capability_record(get_task(name)) for name in available_tasks()]
    if args.command == "methods":
        return [
            {
                "method_id": name,
                "display_name": str(getattr(get_method(name), "display_name", name)),
                "required_capabilities": list(getattr(get_method(name), "required_capabilities", ())),
            }
            for name in available_methods()
        ]
    if args.command == "capabilities":
        return _capability_matrix()
    if args.command == "matrix":
        rows = build_matrix(args.experiment)
        output = write_matrix(rows, args.output)
        return {"output": str(output), "rows": len(rows)}

    task = get_task(args.task)
    require_action(task, args.command)
    config = _load_config(task, args.config)
    _validate_method(task, config, args.command, args.method)
    if args.command != "hardware":
        configure_runtime(config)
    execute = getattr(task, "execute", None)
    if not callable(execute):
        raise TypeError(f"task {args.task!r} does not define execute(action, config)")
    return execute(
        args.command,
        config,
        output_dir=args.output_dir,
        experiment_dir=args.experiment_dir,
        prepared_dir=args.prepared_dir,
        release_csv=args.release_csv,
        ecg_root=args.ecg_root,
        resume=not args.no_resume,
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = dispatch(args)
    print(json.dumps(jsonable(result), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
