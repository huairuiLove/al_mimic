"""Input discovery for native MDS-ED workflows."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


@dataclass(frozen=True, slots=True)
class ReleasePaths:
    csv_path: Path
    ecg_root: Path
    output_dir: Path

    def __iter__(self):
        """Allow legacy tuple unpacking without losing named fields."""
        yield self.csv_path
        yield self.ecg_root
        yield self.output_dir


def _first_file(candidates: list[Path], description: str) -> Path:
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"could not find {description}; checked: " + ", ".join(str(path) for path in candidates)
    )


def _first_directory(candidates: list[Path], description: str) -> Path:
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    archives = [path for path in candidates if path.is_file()]
    if archives:
        raise ValueError(f"{description} must be a fully extracted directory, not an archive: {archives[0]}")
    raise FileNotFoundError(
        f"could not find {description}; checked: " + ", ".join(str(path) for path in candidates)
    )


def discover_release_inputs(
    search_root: str | Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> ReleasePaths:
    """Discover explicit environment paths or common first-party data layouts."""
    env = os.environ if environ is None else environ
    root = Path(search_root or Path.cwd()).expanduser()
    csv_candidates = [
        *(Path(env["MDSED_CSV"]).expanduser() for _ in (0,) if env.get("MDSED_CSV")),
        root / "mds_ed.csv",
        root / "data" / "mds_ed.csv",
        root / "dataset" / "raw" / "mds-ed-1.0.0" / "mds_ed.csv",
    ]
    ecg_candidates = [
        *(Path(env["MDSED_ECG_PATH"]).expanduser() for _ in (0,) if env.get("MDSED_ECG_PATH")),
        root / "mimic-iv-ecg_1.0",
        root / "data" / "mimic-iv-ecg_1.0",
        root / "dataset" / "raw" / "mimic-iv-ecg-1.0",
    ]
    output = Path(env.get("MDSED_OUTPUT_DIR", root / "dataset" / "prepared" / "mds_ed"))
    return ReleasePaths(
        csv_path=_first_file(csv_candidates, "the MDS-ED release CSV").resolve(),
        ecg_root=_first_directory(ecg_candidates, "the extracted MIMIC-IV-ECG release").resolve(),
        output_dir=output.expanduser().resolve(),
    )
