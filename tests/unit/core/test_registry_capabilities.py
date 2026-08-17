from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from al_mimic.cli import build_parser, dispatch
from al_mimic.core.capabilities import (
    CapabilityError,
    capability_record,
    require_action,
    require_method,
    task_capabilities,
)
from al_mimic.methods.registry import get_method
from al_mimic.tasks.registry import available_tasks, get_task, normalize_task_name

ROOT = Path(__file__).resolve().parents[3]


def _run_isolated(source: str) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    source_path = str(ROOT / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (source_path, environment.get("PYTHONPATH", "")) if value
    )
    return subprocess.run(
        [sys.executable, "-c", source],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )


def test_task_registry_is_lazy_until_a_task_is_requested() -> None:
    completed = _run_isolated(
        "import sys; import al_mimic.tasks as tasks; "
        "loaded = sorted(name for name in sys.modules "
        "if name.startswith('al_mimic.tasks.') and name != 'al_mimic.tasks.registry'); "
        "assert loaded == [], loaded; "
        "assert tasks.available_tasks() == ('brset', 'mds_ed', 'mimic_iii')"
    )
    assert completed.returncode == 0, completed.stderr


def test_task_registry_normalizes_aliases_and_exposes_plugin_contracts() -> None:
    assert normalize_task_name("MIMIC-3") == "mimic_iii"
    assert normalize_task_name("mimic-iv-mds-ed") == "mds_ed"
    assert available_tasks() == ("brset", "mds_ed", "mimic_iii")
    assert get_task("mimic-3").task_id == "mimic_iii"
    assert get_task("mimic-iv-mds-ed").task_id == "mds_ed"
    for task_name in available_tasks():
        plugin = get_task(task_name)
        assert plugin.display_name
        assert plugin.actions
        assert isinstance(capability_record(plugin)["methods"], list)


def test_capabilities_allow_mimic_method_and_reject_mds_method() -> None:
    mimic = get_task("mimic_iii")
    random = get_method("random")
    require_action(mimic, "active")
    require_method(mimic, random)

    mds = get_task("mds_ed")
    capabilities = task_capabilities(mds)
    assert capabilities.supervised_only is True
    assert capabilities.methods == frozenset()
    require_action(mds, "train")
    with pytest.raises(CapabilityError, match="supervised-only"):
        require_method(mds, random)
    with pytest.raises(CapabilityError, match="does not support action"):
        require_action(mds, "active")


def test_cli_dispatch_rejects_mds_active_action_before_loading_method() -> None:
    args = build_parser().parse_args(
        [
            "active",
            "--task",
            "mds_ed",
            "--method",
            "random",
            "--config",
            "configs/experiments/mds_ed/diagnoses.yaml",
        ]
    )
    with pytest.raises(CapabilityError, match="does not support action"):
        dispatch(args)
