from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = ROOT / "src" / "al_mimic"
TASK_ROOT = SOURCE_ROOT / "tasks"
METHOD_ROOT = SOURCE_ROOT / "methods"
CORE_ROOT = SOURCE_ROOT / "core"
TASK_NAMES = {"brset", "mds_ed", "mimic_iii"}
METHOD_NAMES = {"random", "comal", "mm_comal", "modis", "mosaic"}
FORBIDDEN_SOURCE_TERMS = ("thirdparty", "third_party", "comal-main", "mds-ed-main")


def _imported_modules(path: Path) -> list[tuple[str, int]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend((alias.name, node.lineno) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            imports.append((module, node.lineno))
    return imports


def _resolve_relative(module: str, level: int, package: tuple[str, ...]) -> str:
    if level < 1 or level > len(package) + 1:
        return module
    prefix = package[: len(package) - level + 1]
    return ".".join(prefix + ((module,) if module else ()))


def _resolved_imports(path: Path) -> list[tuple[str, int]]:
    relative_parts = path.relative_to(SOURCE_ROOT).with_suffix("").parts
    package = ("al_mimic",) + relative_parts[:-1]
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend((alias.name, node.lineno) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            imports.append((_resolve_relative(module, node.level, package), node.lineno))
    return imports


def test_core_has_no_concrete_task_or_method_dependencies() -> None:
    for path in CORE_ROOT.rglob("*.py"):
        for module, line in _resolved_imports(path):
            assert not module.startswith("al_mimic.tasks."), (path, line, module)
            if module.startswith("al_mimic.methods."):
                assert module == "al_mimic.methods.api", (path, line, module)


def test_task_families_do_not_import_each_other_or_concrete_methods() -> None:
    for task_name in TASK_NAMES:
        task_root = TASK_ROOT / task_name
        for path in task_root.rglob("*.py"):
            for module, line in _resolved_imports(path):
                if module.startswith("al_mimic.tasks."):
                    target = module.removeprefix("al_mimic.tasks.").split(".", 1)[0]
                    assert target == task_name, (path, line, module)
                if module.startswith("al_mimic.methods."):
                    assert module in {"al_mimic.methods.api", "al_mimic.methods.registry"}, (
                        path,
                        line,
                        module,
                    )


def test_method_families_do_not_import_each_other_or_tasks() -> None:
    for method_name in METHOD_NAMES:
        method_root = METHOD_ROOT / method_name
        for path in method_root.rglob("*.py"):
            for module, line in _resolved_imports(path):
                assert not module.startswith("al_mimic.tasks"), (path, line, module)
                if module.startswith("al_mimic.methods."):
                    target = module.removeprefix("al_mimic.methods.").split(".", 1)[0]
                    assert target in {method_name, "api"}, (path, line, module)


def test_runtime_python_sources_have_no_external_checkout_markers() -> None:
    for path in SOURCE_ROOT.rglob("*.py"):
        text = path.read_text(encoding="utf-8").lower()
        for term in FORBIDDEN_SOURCE_TERMS:
            assert term not in text, (path, term)


def test_source_imports_have_no_external_checkout_modules() -> None:
    for path in SOURCE_ROOT.rglob("*.py"):
        for module, line in _resolved_imports(path):
            lowered = module.lower()
            assert not any(term in lowered for term in FORBIDDEN_SOURCE_TERMS), (
                path,
                line,
                module,
            )


@pytest.mark.parametrize("name", ["NOTICE", "LICENSE"])
def test_non_python_dependency_notices_are_outside_runtime_scan(name: str) -> None:
    assert (SOURCE_ROOT / "tasks" / "mds_ed" / name).is_file()
