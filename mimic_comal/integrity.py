"""Integrity guard for the untouched upstream CoMAL source tree."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def source_integrity(root: str | Path = ".") -> dict[str, Any]:
    root = Path(root).resolve()
    baseline_path = root / "original_comal_sha256.json"
    if not baseline_path.is_file():
        raise FileNotFoundError(f"original CoMAL hash manifest not found: {baseline_path}")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    actual: dict[str, str | None] = {}
    for relative_path in baseline:
        path = root / relative_path
        if not path.is_file():
            actual[relative_path] = None
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        actual[relative_path] = digest.hexdigest()
    unexpected = sorted(
        str(path.relative_to(root))
        for path in (root / "CoMAL-main").glob("*.py")
        if str(path.relative_to(root)) not in baseline
    )
    changed = {
        path: {"expected": baseline[path], "actual": actual[path]}
        for path in baseline
        if actual[path] != baseline[path]
    }
    return {
        "policy": "adapter-only; original CoMAL files are read-only",
        "verified": not changed and not unexpected,
        "changed_or_missing": changed,
        "unexpected_python_files": unexpected,
        "files": actual,
    }


_INTEGRITY_CACHE: dict[str, dict[str, Any]] = {}


def assert_original_unchanged(root: str | Path = ".") -> dict[str, Any]:
    key = str(Path(root).resolve())
    cached = _INTEGRITY_CACHE.get(key)
    if cached is not None:
        return cached
    report = source_integrity(root)
    if not report["verified"]:
        raise RuntimeError(
            "CoMAL-main integrity check failed; restore upstream files and edit only the adapter"
        )
    _INTEGRITY_CACHE[key] = report
    return report
