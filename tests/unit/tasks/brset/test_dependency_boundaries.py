from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
SOURCE_ROOT = ROOT / "src" / "al_mimic" / "tasks" / "brset"
LEGACY_ROOTS = {"brset_al", "methods", "mimic_comal", "modis", "mosaic"}


def _imported_modules(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    return imported


def test_brset_sources_do_not_import_legacy_or_sibling_tasks() -> None:
    for path in SOURCE_ROOT.rglob("*.py"):
        for module in _imported_modules(path):
            assert module.split(".", 1)[0] not in LEGACY_ROOTS, (path, module)
            assert not (
                module.startswith("al_mimic.tasks.") and not module.startswith("al_mimic.tasks.brset")
            ), (path, module)
            if module.startswith("al_mimic.methods"):
                assert module in {"al_mimic.methods", "al_mimic.methods.api"}, (path, module)


def test_importing_brset_plugin_does_not_load_sibling_tasks_or_legacy_modules() -> None:
    environment = dict(os.environ)
    source_path = str(ROOT / "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (source_path, environment.get("PYTHONPATH", "")) if value
    )
    source = (
        "import sys; import al_mimic.tasks.brset as brset; "
        "assert brset.PLUGIN.task_id == 'brset'; loaded = set(sys.modules); "
        "forbidden = ('brset_al', 'mimic_comal', 'methods', 'modis', 'mosaic'); "
        "assert not any(name == root or name.startswith(root + '.') "
        "for name in loaded for root in forbidden), sorted(loaded); "
        "assert not any(name.startswith('al_mimic.tasks.mimic_iii') or "
        "name.startswith('al_mimic.tasks.mds_ed') for name in loaded), sorted(loaded)"
    )

    completed = subprocess.run(
        [sys.executable, "-c", source],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
