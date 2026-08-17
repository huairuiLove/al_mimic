"""End-to-end native preparation for the official MDS-ED release."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any

from .audit import audit_prepared_memmap, audit_release_csv
from .discovery import ReleasePaths, discover_release_inputs
from .ecg import discover_ecg_records, prepare_ecg_records
from .memmap import build_prepared_memmap
from .tabular import fit_tabular_transform, write_tabular_chunks


def read_release_study_ids(csv_path: str | Path) -> set[int]:
    """Stream study identifiers without loading the release table."""
    path = Path(csv_path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if "general_study_id" not in (reader.fieldnames or ()):
            raise ValueError("MDS-ED release CSV is missing general_study_id")
        studies: set[int] = set()
        for row_number, row in enumerate(reader, start=2):
            try:
                studies.add(int(float(row["general_study_id"])))
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid general_study_id at CSV row {row_number}: {row['general_study_id']!r}"
                ) from exc
    return studies


def prepare_release(
    paths: ReleasePaths | None = None,
    *,
    search_root: str | Path | None = None,
    chunksize: int = 2_048,
    resume: bool = True,
    delete_intermediate_waveforms: bool = False,
) -> dict[str, Any]:
    """Audit, prepare ECGs, build memmap, and shard tabular features natively."""
    resolved = paths or discover_release_inputs(search_root)
    audit = audit_release_csv(resolved.csv_path)
    output = resolved.output_dir
    output.mkdir(parents=True, exist_ok=True)
    output_csv = output / "mds_ed.csv"
    if resolved.csv_path.resolve() != output_csv.resolve():
        shutil.copy2(resolved.csv_path, output_csv)

    study_ids = read_release_study_ids(resolved.csv_path)
    records = discover_ecg_records(resolved.ecg_root, study_ids)
    discovered_ids = {record.study_id for record in records}
    missing = sorted(study_ids.difference(discovered_ids))
    if missing:
        raise ValueError(
            f"MIMIC-IV-ECG is missing {len(missing)} studies referenced by MDS-ED; examples: {missing[:10]}"
        )
    prepared = prepare_ecg_records(records, output, resume=resume)
    build_prepared_memmap(
        prepared,
        output,
        delete_waveforms=delete_intermediate_waveforms,
    )
    spec = fit_tabular_transform(output_csv, chunksize=chunksize, workspace=output)
    write_tabular_chunks(output_csv, output, spec, chunksize=chunksize)
    memmap_audit = audit_prepared_memmap(output, audit.rows)
    payload = {
        "format_version": 1,
        "release": audit.to_dict(),
        "memmap": memmap_audit.to_dict(),
        "tabular": {
            "continuous_dim": spec.continuous_dim,
            "categorical_dim": spec.categorical_dim,
            "category_sizes": list(spec.category_sizes),
        },
        "paths": {
            "release_csv": str(output_csv.resolve()),
            "ecg_root": str(resolved.ecg_root.resolve()),
            "output_dir": str(output.resolve()),
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload
