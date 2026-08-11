"""Configuration loading for the formal Yang and Wu MIMIC-III diagnosis task."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


FORMAL_STRATEGIES = {"comal", "mm_comal", "modis", "mosaic", "random"}


def _require_equal(name: str, actual: Any, required: Any) -> None:
    if actual != required:
        raise ValueError(f"formal Yang-Wu protocol requires {name}={required!r}, got {actual!r}")


def _reject_shortcuts(value: Any, prefix: str = "") -> None:
    """Reject experiment-shortening knobs in every formal configuration."""
    if not isinstance(value, dict):
        return
    forbidden = {
        "dry_run",
        "dryrun",
        "smoke",
        "max_records",
        "max_rows",
        "max_batches",
        "max_steps",
        "limit_train_batches",
        "fast_dev_run",
        "resume_from",
        "round_checkpoint",
        "previous_checkpoint",
        "warm_start",
    }
    for key, child in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if str(key).lower() in forbidden:
            raise ValueError(f"formal experiments forbid shortcut setting: {path}")
        _reject_shortcuts(child, path)


def _validate_yang_wu_protocol(config: dict[str, Any]) -> None:
    dataset = config.get("dataset", {})
    preprocessing = config.get("preprocessing", {})
    model = config.get("model", {})
    training = config.get("training", {})
    active = config.get("active_learning", {})

    required_values = {
        "dataset.split_group": (dataset.get("split_group"), "with_notes"),
        "preprocessing.cohort": (preprocessing.get("cohort"), "metavision"),
        "preprocessing.task": (preprocessing.get("task"), "Diagnoses"),
        "preprocessing.split_protocol": (
            preprocessing.get("split_protocol"),
            "yang_wu_official_7_1.5_1.5",
        ),
        "preprocessing.label_format": (
            preprocessing.get("label_format"),
            "icd9_top3_multihot",
        ),
        "preprocessing.note_protocol": (
            preprocessing.get("note_protocol"),
            "latest_per_category_description_within_48h",
        ),
        "model.architecture": (model.get("architecture"), "yang_wu_bertencoder"),
        "model.initialization": (
            model.get("initialization"),
            "clinicalbert_pretrained_fresh_fusion_each_round",
        ),
        "model.output_activation": (model.get("output_activation"), "sigmoid"),
        "training.optimizer": (training.get("optimizer"), "adam"),
        "training.loss": (training.get("loss"), "binary_cross_entropy"),
        "training.precision": (training.get("precision"), "fp32"),
    }
    for name, (actual, required) in required_values.items():
        _require_equal(name, actual, required)

    integer_values = {
        "preprocessing.observation_hours": (preprocessing.get("observation_hours"), 48),
        "preprocessing.timestep_hours": (preprocessing.get("timestep_hours"), 1),
        "preprocessing.max_note_tokens": (preprocessing.get("max_note_tokens"), 512),
        # Local FIDDLE Diagnoses rebuild dims (paper: 10210 / 1042 / 7411).
        "preprocessing.expected_total_samples": (
            preprocessing.get("expected_total_samples"),
            10258,
        ),
        "preprocessing.expected_label_count": (
            preprocessing.get("expected_label_count"),
            915,
        ),
        "preprocessing.time_invariant_dim": (
            preprocessing.get("time_invariant_dim"),
            97,
        ),
        "preprocessing.time_series_dim": (
            preprocessing.get("time_series_dim"),
            7749,
        ),
        "model.text_hidden_dim": (model.get("text_hidden_dim"), 768),
        "model.time_invariant_hidden_dim": (
            model.get("time_invariant_hidden_dim"),
            64,
        ),
        "model.time_series_hidden_dim": (model.get("time_series_hidden_dim"), 1024),
        "model.time_series_layers": (model.get("time_series_layers"), 3),
        "model.time_series_heads": (model.get("time_series_heads"), 16),
        "model.output_size": (model.get("output_size"), 915),
        "training.epochs": (training.get("epochs"), 20),
        "active_learning.rounds": (active.get("rounds"), 6),
    }
    for name, (actual, required) in integer_values.items():
        try:
            normalized = int(actual)
        except (TypeError, ValueError):
            normalized = actual
        _require_equal(name, normalized, required)

    float_values = {
        "model.dropout": (model.get("dropout"), 0.1),
        "training.learning_rate": (training.get("learning_rate"), 1e-4),
        "training.weight_decay": (training.get("weight_decay"), 0.0),
        "training.gradient_clip": (training.get("gradient_clip"), 1.0),
        "training.warmup_proportion": (training.get("warmup_proportion"), 0.1),
        "active_learning.initial_fraction": (active.get("initial_fraction"), 0.10),
        "active_learning.query_fraction": (active.get("query_fraction"), 0.05),
    }
    for name, (actual, required) in float_values.items():
        try:
            normalized = float(actual)
        except (TypeError, ValueError):
            normalized = actual
        _require_equal(name, normalized, required)

    strategy = str(active.get("strategy", "")).lower()
    if strategy not in FORMAL_STRATEGIES:
        raise ValueError(f"active_learning.strategy must be one of {sorted(FORMAL_STRATEGIES)}")
    if bool(training.get("inherit_across_rounds", False)):
        raise ValueError("formal protocol forbids cross-round weight inheritance")
    if not str(dataset.get("clinicalbert_checkpoint", "")).strip():
        raise ValueError("dataset.clinicalbert_checkpoint must name the ClinicalBERT source checkpoint")
    _reject_shortcuts(config)


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
    _validate_yang_wu_protocol(config)
    return config


def resolve_path(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    config_path = Path(config.get("_config_path", ".")).resolve()
    return (config_path.parent / path).resolve() if str(path).startswith("./") else path.resolve()


def require_paths(config: dict[str, Any]) -> dict[str, Path]:
    dataset = config.get("dataset", {})
    paths = {
        "split_hdf5": resolve_path(config, dataset.get("split_hdf5", "")),
        "clinicalbert_checkpoint": resolve_path(
            config, dataset.get("clinicalbert_checkpoint", "")
        ),
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        details = ", ".join(f"{name}={paths[name]}" for name in missing)
        raise FileNotFoundError(f"missing formal Yang-Wu inputs: {details}")
    return paths
