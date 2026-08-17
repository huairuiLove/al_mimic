from __future__ import annotations

import json
from pathlib import Path

import pytest

from al_mimic.cli import build_parser, dispatch
from al_mimic.evaluation import build_matrix, write_matrix
from al_mimic.methods.registry import METHODS, available_methods, get_method

ROOT = Path(__file__).resolve().parents[3]


def test_parser_exposes_subcommands_for_registry_and_run_actions() -> None:
    assert build_parser().parse_args(["tasks"]).command == "tasks"
    assert build_parser().parse_args(["methods"]).command == "methods"
    assert build_parser().parse_args(["capabilities"]).command == "capabilities"

    args = build_parser().parse_args(
        [
            "active",
            "--task",
            "mimic_iii",
            "--method",
            "mosaic",
            "--config",
            "configs/experiments/mimic_iii/mosaic.yaml",
        ]
    )
    assert (args.command, args.task, args.method) == ("active", "mimic_iii", "mosaic")
    assert args.config == Path("configs/experiments/mimic_iii/mosaic.yaml")


def test_hardware_action_does_not_require_a_config() -> None:
    args = build_parser().parse_args(["hardware", "--task", "mds_ed"])
    assert (args.command, args.task, args.config) == ("hardware", "mds_ed", None)


def test_run_parser_requires_config_for_data_actions() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["validate-data", "--task", "mimic_iii"])


def test_registry_methods_are_first_party_plugins() -> None:
    assert set(METHODS) == {"random", "comal", "modis", "modimix", "mosaic"}
    assert set(available_methods()) == set(METHODS)
    for name in METHODS:
        plugin = get_method(name)
        assert plugin.__module__.startswith(f"al_mimic.methods.{name}")
        assert plugin.method_id == name
        assert callable(plugin.acquire)


def test_tasks_dispatch_returns_capability_records() -> None:
    records = dispatch(build_parser().parse_args(["tasks"]))
    by_id = {record["task_id"]: record for record in records}
    assert set(by_id) == {"brset", "mds_ed", "mimic_iii"}
    assert by_id["mimic_iii"]["methods"] == sorted(METHODS)
    assert by_id["mds_ed"]["supervised_only"] is True
    assert by_id["mds_ed"]["methods"] == []


def test_methods_dispatch_reports_required_capabilities() -> None:
    records = dispatch(build_parser().parse_args(["methods"]))
    by_id = {record["method_id"]: record for record in records}
    assert set(by_id) == set(METHODS)
    assert by_id["random"]["required_capabilities"] == []
    assert "modality_tokens" in by_id["mosaic"]["required_capabilities"]


def test_capabilities_dispatch_builds_task_method_matrix() -> None:
    records = dispatch(build_parser().parse_args(["capabilities"]))
    by_id = {record["task_id"]: record for record in records}
    assert by_id["mimic_iii"]["method_support"]["mosaic"] is True
    assert by_id["mimic_iii"]["method_support"]["modimix"] is True
    assert by_id["brset"]["method_support"]["modimix"] is False
    assert by_id["brset"]["method_support"]["random"] is True
    assert all(value is False for value in by_id["mds_ed"]["method_support"].values())


def test_evaluation_matrix_flattens_and_writes_results(tmp_path: Path) -> None:
    experiment = tmp_path / "experiment_a"
    experiment.mkdir()
    (experiment / "final_metrics.json").write_text(
        json.dumps({"strategy": "comal", "test": {"recall_at_30": 0.75}}),
        encoding="utf-8",
    )

    rows = build_matrix([experiment])
    assert rows == [{"experiment": "experiment_a", "strategy": "comal", "test.recall_at_30": 0.75}]
    output = write_matrix(rows, tmp_path / "matrix.csv")
    assert "test.recall_at_30" in output.read_text(encoding="utf-8")
