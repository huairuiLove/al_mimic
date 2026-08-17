from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest

from al_mimic.methods.registry import available_methods, get_method

ROOT = Path(__file__).resolve().parents[3]
SOURCE_ROOT = ROOT / "src" / "al_mimic" / "methods"
METHOD_NAMES = {"random", "comal", "modis", "modimix", "mosaic"}


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


def test_registry_is_lazy_until_a_method_is_requested() -> None:
    completed = _run_isolated(
        "import sys; import al_mimic.methods as methods; "
        "loaded = sorted(name for name in sys.modules "
        "if name.startswith('al_mimic.methods.') and name != 'al_mimic.methods.registry'); "
        "assert loaded == [], loaded; assert methods.available_methods() == "
        "('comal', 'modimix', 'modis', 'mosaic', 'random')"
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("method_name", sorted(METHOD_NAMES))
def test_each_method_imports_without_sibling_or_task_modules(method_name: str) -> None:
    completed = _run_isolated(
        f"import importlib, sys; importlib.import_module('al_mimic.methods.{method_name}'); "
        f"siblings = {METHOD_NAMES!r} - {{{method_name!r}}}; "
        "loaded = set(sys.modules); "
        "assert not any(name.startswith('al_mimic.tasks') for name in loaded); "
        "assert not any(any(name == 'al_mimic.methods.' + sibling or "
        "name.startswith('al_mimic.methods.' + sibling + '.') for sibling in siblings) "
        "for name in loaded), sorted(loaded)"
    )
    assert completed.returncode == 0, completed.stderr


def test_method_sources_have_no_legacy_or_cross_method_imports() -> None:
    for path in SOURCE_ROOT.rglob("*.py"):
        relative = path.relative_to(SOURCE_ROOT)
        owner = relative.parts[0] if relative.parts[0] in METHOD_NAMES else None
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            imported: list[str] = []
            if isinstance(node, ast.Import):
                imported = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported = [node.module]
            for module in imported:
                root = module.split(".", 1)[0]
                assert root not in {"methods", "mimic_comal", "modis", "mosaic"}
                assert not module.startswith("al_mimic.tasks")
                prefix = "al_mimic.methods."
                if owner and module.startswith(prefix):
                    target = module.removeprefix(prefix).split(".", 1)[0]
                    assert target not in METHOD_NAMES - {owner}


def test_registry_loads_uniform_plugin_api_and_aliases() -> None:
    assert set(available_methods()) == METHOD_NAMES
    for method_name in METHOD_NAMES:
        plugin = get_method(method_name)
        assert plugin.method_id == method_name
        assert plugin.display_name
        assert isinstance(plugin.required_capabilities, tuple)
        assert isinstance(plugin.required_context_fields, tuple)
        assert callable(plugin.acquire)


def test_only_stateful_methods_expose_fit_hooks() -> None:
    assert callable(get_method("comal").fit)
    assert callable(get_method("comal").prepare_context)
    assert callable(get_method("modis").fit)
    assert callable(get_method("modimix").fit)
    assert not hasattr(get_method("random"), "fit")
    assert not hasattr(get_method("mosaic"), "fit")


def test_random_plugin_is_reproducible_and_accepts_attribute_context() -> None:
    class Context:
        candidate_ids = (10, 11, 12, 13)
        query_size = 2
        seed = 5
        round_index = 3

    plugin = get_method("random")
    first = plugin.acquire(Context())
    second = plugin.acquire(Context())
    assert first == second
    assert len(first.selected_ids) == 2
    assert len(set(first.selected_positions)) == 2
    assert first.diagnostics["seed"] == 8
