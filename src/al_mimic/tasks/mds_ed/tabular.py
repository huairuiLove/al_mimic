"""Low-memory fitting and transformation of MDS-ED tabular features."""

from __future__ import annotations

import json
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Literal

import numpy as np

from .audit import read_release_schema
from .constants import (
    DETERIORATION_LABEL_COUNT,
    DIAGNOSIS_LABEL_COUNT,
    RAW_TABULAR_FEATURE_COUNT,
    TRAIN_FOLDS,
)


@dataclass(frozen=True, slots=True)
class TabularSpec:
    raw_feature_columns: tuple[str, ...]
    input_columns: tuple[str, ...]
    continuous_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    mask_columns: tuple[str, ...]
    medians: tuple[float, ...]
    category_values: tuple[tuple[float, ...], ...]
    diagnosis_columns: tuple[str, ...]
    deterioration_columns: tuple[str, ...]

    @property
    def continuous_dim(self) -> int:
        return len(self.continuous_columns)

    @property
    def categorical_dim(self) -> int:
        return len(self.categorical_columns)

    @property
    def category_sizes(self) -> tuple[int, ...]:
        return tuple(len(values) for values in self.category_values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> TabularSpec:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        tuple_fields = (
            "raw_feature_columns",
            "input_columns",
            "continuous_columns",
            "categorical_columns",
            "mask_columns",
            "medians",
            "diagnosis_columns",
            "deterioration_columns",
        )
        for field in tuple_fields:
            payload[field] = tuple(payload[field])
        payload["category_values"] = tuple(tuple(values) for values in payload["category_values"])
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class TabularBatch:
    study_ids: np.ndarray
    subject_ids: np.ndarray
    folds: np.ndarray
    continuous: np.ndarray
    categorical: np.ndarray
    labels: np.ndarray

    @property
    def rows(self) -> int:
        return int(self.study_ids.size)


def _import_pandas():
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - dependency error is environment-specific
        raise RuntimeError("pandas is required for MDS-ED tabular transformation") from exc
    return pd


def _retained_rows(frame):
    return frame[
        (frame["general_strat_fold"] < min(TRAIN_FOLDS[-1] + 1, 18))
        | (frame["general_ecg_no_within_stay"] == 0)
    ].copy()


def _derive_features(frame, feature_columns: tuple[str, ...]):
    ethnicity = tuple(column for column in feature_columns if column.startswith("demographics_ethnicity_"))
    result = frame.loc[:, feature_columns].copy()
    if ethnicity:
        result["demographics_ethnicity"] = (
            result.loc[:, ethnicity].fillna(0.0).to_numpy(dtype=np.float32).argmax(axis=1)
        )
        result = result.drop(columns=list(ethnicity))
    if "vitals_acuity" in result:
        result["vitals_acuity"] = result["vitals_acuity"] - 1.0
    return result


def _input_columns(feature_columns: tuple[str, ...]) -> tuple[str, ...]:
    ethnicity = tuple(column for column in feature_columns if column.startswith("demographics_ethnicity_"))
    columns = [column for column in feature_columns if column not in ethnicity]
    if ethnicity:
        columns.append("demographics_ethnicity")
    return tuple(columns)


def _update_uniques(
    trackers: dict[str, set[float] | None],
    values,
    *,
    cardinality_limit: int = 10,
) -> None:
    for column in values.columns:
        tracker = trackers[column]
        if tracker is None:
            continue
        observed = np.asarray(values[column], dtype=np.float32)
        observed = np.unique(observed[np.isfinite(observed)])
        tracker.update(float(value) for value in observed)
        if len(tracker) >= cardinality_limit:
            trackers[column] = None


def fit_tabular_transform(
    csv_path: str | Path,
    *,
    chunksize: int = 2_048,
    introduce_missing_masks: bool = True,
    missing_masks_as_categorical: bool = False,
    expected_diagnosis_labels: int = DIAGNOSIS_LABEL_COUNT,
    expected_deterioration_labels: int = DETERIORATION_LABEL_COUNT,
    expected_raw_features: int = RAW_TABULAR_FEATURE_COUNT,
    workspace: str | Path | None = None,
) -> TabularSpec:
    """Fit exact train-fold medians while holding only one CSV chunk in RAM."""
    if chunksize < 1:
        raise ValueError("chunksize must be positive")
    pd = _import_pandas()
    schema = read_release_schema(
        csv_path,
        expected_diagnosis_labels=expected_diagnosis_labels,
        expected_deterioration_labels=expected_deterioration_labels,
        expected_raw_features=expected_raw_features,
    )
    input_columns = _input_columns(schema.raw_feature_columns)
    required = [
        "general_strat_fold",
        "general_ecg_no_within_stay",
        *schema.raw_feature_columns,
    ]
    dtype = {column: "float32" for column in schema.raw_feature_columns}
    dtype.update({"general_strat_fold": "int16", "general_ecg_no_within_stay": "int16"})
    missing_train = {column: False for column in input_columns}
    missing_all = {column: False for column in input_columns}
    unique_values: dict[str, set[float] | None] = {column: set() for column in input_columns}
    train_rows = 0

    workspace_path = None if workspace is None else Path(workspace)
    if workspace_path is not None:
        workspace_path.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=workspace_path, prefix="mdsed-tabular-") as temporary:
        raw_path = Path(temporary) / "train_features.float32"
        with raw_path.open("wb") as raw_handle:
            for chunk in pd.read_csv(
                csv_path,
                usecols=required,
                dtype=dtype,
                chunksize=chunksize,
                low_memory=False,
            ):
                retained = _retained_rows(chunk)
                features = _derive_features(retained, schema.raw_feature_columns)
                _update_uniques(unique_values, features)
                missing = features.isna().any(axis=0)
                for column in input_columns:
                    missing_all[column] |= bool(missing[column])
                train_features = features[retained["general_strat_fold"] < 18]
                train_missing = train_features.isna().any(axis=0)
                for column in input_columns:
                    missing_train[column] |= bool(train_missing[column])
                values = train_features.loc[:, input_columns].to_numpy(dtype=np.float32)
                values.tofile(raw_handle)
                train_rows += len(values)
        if train_rows == 0:
            raise ValueError("MDS-ED tabular fit has no rows in training folds 0..17")
        train_values = np.memmap(
            raw_path,
            dtype=np.float32,
            mode="r",
            shape=(train_rows, len(input_columns)),
        )
        medians = np.empty(len(input_columns), dtype=np.float32)
        for index, column in enumerate(input_columns):
            values = np.asarray(train_values[:, index])
            finite = values[np.isfinite(values)]
            medians[index] = np.median(finite) if finite.size else 0.0
            if missing_all[column] and unique_values[column] is not None:
                unique_values[column].add(float(medians[index]))
        del train_values

    categorical_columns: list[str] = []
    category_values: list[tuple[float, ...]] = []
    for column in input_columns:
        values = unique_values[column]
        if values is None or len(values) >= 10 or column.startswith("labvalues_"):
            continue
        categorical_columns.append(column)
        category_values.append(tuple(sorted(values)))
    mask_columns = tuple(
        f"{column}_nan" for column in input_columns if introduce_missing_masks and missing_train[column]
    )
    if missing_masks_as_categorical:
        categorical_columns.extend(mask_columns)
        category_values.extend((0.0, 1.0) for _ in mask_columns)
    continuous_columns = tuple(column for column in input_columns if column not in categorical_columns) + (
        () if missing_masks_as_categorical else mask_columns
    )
    return TabularSpec(
        raw_feature_columns=schema.raw_feature_columns,
        input_columns=input_columns,
        continuous_columns=continuous_columns,
        categorical_columns=tuple(categorical_columns),
        mask_columns=mask_columns,
        medians=tuple(float(value) for value in medians),
        category_values=tuple(category_values),
        diagnosis_columns=schema.diagnosis_columns,
        deterioration_columns=schema.deterioration_columns,
    )


def transform_tabular_chunks(
    csv_path: str | Path,
    spec: TabularSpec,
    *,
    task: Literal["diagnoses", "deterioration"] = "diagnoses",
    chunksize: int = 2_048,
) -> Iterator[TabularBatch]:
    """Yield transformed arrays one bounded chunk at a time."""
    if chunksize < 1:
        raise ValueError("chunksize must be positive")
    pd = _import_pandas()
    label_columns = spec.diagnosis_columns if task == "diagnoses" else spec.deterioration_columns
    required = list(
        dict.fromkeys(
            [
                "general_study_id",
                "general_subject_id",
                "general_strat_fold",
                "general_ecg_no_within_stay",
                *spec.raw_feature_columns,
                *label_columns,
            ]
        )
    )
    dtype = {column: "float32" for column in (*spec.raw_feature_columns, *label_columns)}
    dtype.update(
        {
            "general_study_id": "int64",
            "general_subject_id": "int64",
            "general_strat_fold": "int16",
            "general_ecg_no_within_stay": "int16",
        }
    )
    median_by_column = dict(zip(spec.input_columns, spec.medians))
    category_by_column = dict(zip(spec.categorical_columns, spec.category_values))
    for chunk in pd.read_csv(
        csv_path,
        usecols=required,
        dtype=dtype,
        chunksize=chunksize,
        low_memory=False,
    ):
        retained = _retained_rows(chunk)
        if retained.empty:
            continue
        features = _derive_features(retained, spec.raw_feature_columns)
        for mask_column in spec.mask_columns:
            source = mask_column.removesuffix("_nan")
            features[mask_column] = features[source].isna().astype(np.float32)
        for column in spec.input_columns:
            features[column] = features[column].fillna(median_by_column[column])

        categorical = np.empty((len(features), len(spec.categorical_columns)), dtype=np.int64)
        for index, column in enumerate(spec.categorical_columns):
            mapping = {value: code for code, value in enumerate(category_by_column[column])}
            encoded = features[column].map(mapping)
            if encoded.isna().any():
                examples = sorted(set(float(value) for value in features.loc[encoded.isna(), column]))[:5]
                raise ValueError(f"unseen categorical values in {column}: {examples}")
            categorical[:, index] = encoded.to_numpy(dtype=np.int64)
        continuous = features.loc[:, spec.continuous_columns].to_numpy(dtype=np.float32)
        labels = retained.loc[:, label_columns].to_numpy(dtype=np.float32)
        if task == "deterioration":
            labels[labels == -999.0] = np.nan
        yield TabularBatch(
            study_ids=retained["general_study_id"].to_numpy(dtype=np.int64),
            subject_ids=retained["general_subject_id"].to_numpy(dtype=np.int64),
            folds=retained["general_strat_fold"].to_numpy(dtype=np.int16),
            continuous=continuous,
            categorical=categorical,
            labels=labels,
        )


def write_tabular_chunks(
    csv_path: str | Path,
    output_dir: str | Path,
    spec: TabularSpec,
    *,
    task: Literal["diagnoses", "deterioration"] = "diagnoses",
    chunksize: int = 2_048,
) -> Path:
    """Write bounded NPZ shards and a manifest consumed by native trainers."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    spec_path = output / "tabular_spec.json"
    spec.save(spec_path)
    shard_entries: list[dict[str, int | str]] = []
    total_rows = 0
    for index, batch in enumerate(transform_tabular_chunks(csv_path, spec, task=task, chunksize=chunksize)):
        shard = output / f"tabular-{index:05d}.npz"
        np.savez(
            shard,
            study_id=batch.study_ids,
            subject_id=batch.subject_ids,
            fold=batch.folds,
            continuous=batch.continuous,
            categorical=batch.categorical,
            labels=batch.labels,
        )
        shard_entries.append({"file": shard.name, "rows": batch.rows})
        total_rows += batch.rows
    if not shard_entries:
        raise ValueError("MDS-ED tabular transform produced no rows")
    manifest = {
        "format_version": 1,
        "task": task,
        "rows": total_rows,
        "continuous_dim": spec.continuous_dim,
        "categorical_dim": spec.categorical_dim,
        "category_sizes": list(spec.category_sizes),
        "shards": shard_entries,
    }
    manifest_path = output / "tabular_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest_path
