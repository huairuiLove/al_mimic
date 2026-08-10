"""Configuration loading and validation for MIMIC-III experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _validate_scratch_multimodal(config: dict[str, Any]) -> None:
    model = config.get("model", {})
    if str(model.get("architecture", "")).lower() != "multimodal_transformer_scratch":
        return
    features = config.get("features", {})
    if str(features.get("encoder", "")).lower() != "multimodal_scratch":
        raise ValueError("scratch multimodal model requires features.encoder=multimodal_scratch")
    if str(model.get("initialization", "")).lower() != "random":
        raise ValueError("scratch multimodal model requires model.initialization=random")
    forbidden = ("pretrained", "checkpoint", "model_path", "weights_path", "resume_from")

    def visit(value: Any, prefix: str = "") -> None:
        if not isinstance(value, dict):
            return
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if any(token in str(key).lower() for token in forbidden) and child not in (None, False, ""):
                raise ValueError(f"pretrained/checkpoint input is forbidden for scratch training: {path}")
            visit(child, path)

    visit(config)


def _validate_active_learning(config: dict[str, Any]) -> None:
    strategy = str(config.get("active_learning", {}).get("strategy", "comal")).lower()
    allowed = {"comal", "mm_comal", "modis", "mosaic", "random"}
    if strategy not in allowed:
        raise ValueError(f"active_learning.strategy must be one of {sorted(allowed)}")
    architecture = str(config.get("model", {}).get("architecture", "")).lower()
    if strategy in {"mm_comal", "modis", "mosaic"} and architecture != "multimodal_transformer_scratch":
        raise ValueError(f"{strategy} requires model.architecture=multimodal_transformer_scratch")
    minimum_observed_bins = int(
        config.get("dataset", {}).get(
            "min_observed_bins", config.get("data", {}).get("min_observed_bins", 0)
        )
    )
    if minimum_observed_bins < 0:
        raise ValueError("dataset.min_observed_bins must be non-negative")
    modality_dropout = float(config.get("model", {}).get("modality_dropout", 0.0))
    if not 0.0 <= modality_dropout < 1.0:
        raise ValueError("model.modality_dropout must be in [0, 1)")
    if strategy == "mm_comal":
        mm = config.get("acquisition", {}).get("mm", {})
        if float(mm.get("alpha", 1.0)) < 0.0:
            raise ValueError("acquisition.mm.alpha must be non-negative")
        if str(mm.get("threshold_estimator", "shrunk")).lower() not in {"shrunk", "midpoint"}:
            raise ValueError("acquisition.mm.threshold_estimator must be shrunk or midpoint")
    if strategy == "mosaic":
        mosaic = config.get("mosaic", {})
        if not 0.0 <= float(mosaic.get("eta", 0.25)) <= 1.0:
            raise ValueError("mosaic.eta must be in [0, 1]")
        for key in ("partners", "workset_size", "synergy_workset_size"):
            if int(mosaic.get(key, 1)) < 1:
                raise ValueError(f"mosaic.{key} must be positive")
    if strategy == "modis":
        modis = config.get("modis", {})
        beta = modis.get("beta", [1.0, 1.0, 1.0])
        if not isinstance(beta, list) or len(beta) != 3:
            raise ValueError("modis.beta must contain three values")
        if any(not isinstance(value, (int, float)) for value in beta):
            raise ValueError("modis.beta values must be numeric")
        for key in ("grid_k", "workset_size", "fusion_batch_size", "probe_epochs"):
            if int(modis.get(key, 1)) < 1:
                raise ValueError(f"modis.{key} must be positive")
        if int(modis.get("bisect_steps", 0)) < 0:
            raise ValueError("modis.bisect_steps must be non-negative")
        if int(modis.get("oof_folds", 5)) < 2:
            raise ValueError("modis.oof_folds must be at least 2")
        if str(modis.get("prototype", "mean")).lower() not in {"mean", "medoid"}:
            raise ValueError("modis.prototype must be mean or medoid")


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a mapping")
    parent = config.pop("extends", None)
    if parent:
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = path.parent / parent_path
        config = _merge(load_config(parent_path), config)
    config["_config_path"] = str(path.resolve())
    _validate_scratch_multimodal(config)
    _validate_active_learning(config)
    return config


def require_paths(config: dict[str, Any]) -> dict[str, Path]:
    dataset = config.get("dataset", {})
    root = Path(dataset.get("root", "mimic-iii-clinical-database-1.4"))

    def table_path(config_key: str, table_name: str) -> Path:
        explicit = dataset.get(config_key)
        if explicit:
            return Path(explicit)
        compressed = root / f"{table_name}.csv.gz"
        return compressed if compressed.is_file() else root / f"{table_name}.csv"

    paths = {
        "root": root,
        "admissions": table_path("admissions", "ADMISSIONS"),
        "diagnoses": table_path("diagnoses", "DIAGNOSES_ICD"),
        "notes": table_path("notes", "NOTEEVENTS"),
    }
    missing = [name for name, value in paths.items() if name != "root" and not value.is_file()]
    if missing:
        raise FileNotFoundError("missing MIMIC-III files: " + ", ".join(missing))
    return paths


def require_multimodal_paths(config: dict[str, Any]) -> dict[str, Path]:
    """Resolve the additional raw tables required by the multimodal adapter."""
    paths = require_paths(config)
    dataset = config.get("dataset", {})
    root = paths["root"]

    def table_path(config_key: str, table_name: str) -> Path:
        explicit = dataset.get(config_key)
        if explicit:
            return Path(explicit)
        compressed = root / f"{table_name}.csv.gz"
        return compressed if compressed.is_file() else root / f"{table_name}.csv"

    paths.update(
        {
            "patients": table_path("patients", "PATIENTS"),
            "icustays": table_path("icustays", "ICUSTAYS"),
            "chartevents": table_path("chartevents", "CHARTEVENTS"),
        }
    )
    missing = [name for name in ("patients", "icustays", "chartevents") if not paths[name].is_file()]
    if missing:
        raise FileNotFoundError("missing multimodal MIMIC-III files: " + ", ".join(missing))
    return paths


def resolve_path(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value)
    config_path = Path(config.get("_config_path", ".")).resolve()
    if not path.is_absolute() and config_path.parent.exists() and str(path).startswith("./"):
        return (config_path.parent / path).resolve()
    return path
