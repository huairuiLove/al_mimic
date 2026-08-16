"""Strict configuration for the BRSET multimodal multi-label base."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

FORMAL_STRATEGIES = {"comal", "mm_comal", "modis", "mosaic", "random"}
POST_DIAGNOSTIC_FIELDS = {
    "optic_disc",
    "vessels",
    "macula",
    "DR_SDRG",
    "DR_ICDR",
    "focus",
    "illumination",
    "image_field",
    "artifacts",
    "quality",
}


def _require_equal(name: str, actual: Any, required: Any) -> None:
    if actual != required:
        raise ValueError(f"formal BRSET protocol requires {name}={required!r}, got {actual!r}")


def _reject_shortcuts(value: Any, prefix: str = "") -> None:
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


def _validate_protocol(config: dict[str, Any]) -> None:
    dataset = config.get("dataset", {})
    preprocessing = config.get("preprocessing", {})
    model = config.get("model", {})
    training = config.get("training", {})
    active = config.get("active_learning", {})
    metadata = preprocessing.get("metadata", {})

    required_values = {
        "dataset.version": (dataset.get("version"), "1.0.2"),
        "preprocessing.split_protocol": (
            preprocessing.get("split_protocol"),
            "patient_multilabel_stratified_60_20_20",
        ),
        "preprocessing.query_unit": (preprocessing.get("query_unit"), "patient"),
        "model.architecture": (model.get("architecture"), "brset_resnet50_metadata_fusion"),
        "model.image_weights": (model.get("image_weights"), "IMAGENET1K_V2"),
        "model.output_activation": (model.get("output_activation"), "sigmoid"),
        "model.initialization": (
            model.get("initialization"),
            "imagenet_resnet50_fresh_fusion_each_round",
        ),
        "training.optimizer": (training.get("optimizer"), "adam"),
        "training.loss": (training.get("loss"), "binary_cross_entropy"),
        "training.precision": (training.get("precision"), "fp32"),
        "active_learning.query_unit": (active.get("query_unit"), "patient"),
        "active_learning.patient_score_aggregation": (
            active.get("patient_score_aggregation"),
            "fused_patient_representation",
        ),
    }
    for name, (actual, required) in required_values.items():
        _require_equal(name, actual, required)

    integer_values = {
        "preprocessing.expected_images": (preprocessing.get("expected_images"), 16266),
        "preprocessing.expected_patients": (preprocessing.get("expected_patients"), 8524),
        "preprocessing.expected_labels": (preprocessing.get("expected_labels"), 13),
        "preprocessing.resize_size": (preprocessing.get("resize_size"), 256),
        "preprocessing.image_size": (preprocessing.get("image_size"), 224),
        "model.output_size": (model.get("output_size"), 13),
        "model.image_feature_dim": (model.get("image_feature_dim"), 2048),
        "model.metadata_hidden_dim": (model.get("metadata_hidden_dim"), 128),
        "model.fusion_dim": (model.get("fusion_dim"), 512),
        "training.batch_size": (training.get("batch_size"), 8),
        "training.epochs": (training.get("epochs"), 20),
        "training.comal_epochs": (training.get("comal_epochs"), 20),
        "active_learning.rounds": (active.get("rounds"), 6),
    }
    for name, (actual, required) in integer_values.items():
        try:
            normalized = int(actual)
        except (TypeError, ValueError):
            normalized = actual
        _require_equal(name, normalized, required)

    float_values = {
        "preprocessing.train_fraction": (preprocessing.get("train_fraction"), 0.60),
        "preprocessing.validation_fraction": (
            preprocessing.get("validation_fraction"),
            0.20,
        ),
        "preprocessing.test_fraction": (preprocessing.get("test_fraction"), 0.20),
        "training.learning_rate": (training.get("learning_rate"), 1e-4),
        "training.weight_decay": (training.get("weight_decay"), 0.0),
        "training.gradient_clip": (training.get("gradient_clip"), 1.0),
        "active_learning.initial_fraction": (active.get("initial_fraction"), 0.10),
        "active_learning.query_fraction": (active.get("query_fraction"), 0.05),
    }
    for name, (actual, required) in float_values.items():
        try:
            normalized = float(actual)
        except (TypeError, ValueError):
            normalized = actual
        _require_equal(name, normalized, required)

    expected_metadata = {
        "numeric": ["patient_age", "diabetes_time_y"],
        "categorical": ["camera", "insulin", "patient_sex", "exam_eye", "diabetes"],
        "comorbidity": "comorbidities",
    }
    for key, required in expected_metadata.items():
        _require_equal(f"preprocessing.metadata.{key}", metadata.get(key), required)
    configured_inputs = {
        str(value) for value in [*metadata.get("numeric", []), *metadata.get("categorical", [])]
    }
    configured_inputs.add(str(metadata.get("comorbidity", "")))
    leaked = sorted(configured_inputs & POST_DIAGNOSTIC_FIELDS)
    if leaked:
        raise ValueError(f"BRSET metadata inputs leak post-diagnostic annotations: {leaked}")

    strategy = str(active.get("strategy", "")).lower()
    if strategy not in FORMAL_STRATEGIES:
        raise ValueError(f"active_learning.strategy must be one of {sorted(FORMAL_STRATEGIES)}")
    if bool(training.get("inherit_across_rounds", False)):
        raise ValueError("formal protocol forbids cross-round weight inheritance")
    if not bool(preprocessing.get("include_inadequate_images", False)):
        raise ValueError("formal BRSET protocol uses every image, including inadequate images")
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
    path = Path(path).resolve()
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
    config["_config_path"] = str(path)
    _validate_protocol(config)
    return config


def resolve_path(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path(config["_config_path"]).parent / path).resolve()


def data_paths(config: dict[str, Any], *, require_prepared: bool = False) -> dict[str, Path]:
    dataset = config.get("dataset", {})
    root = resolve_path(config, dataset.get("root", ""))
    labels_value = Path(str(dataset.get("labels_csv", "label_brset.csv")))
    images_value = Path(str(dataset.get("images_dir", "fundus_photos")))
    prepared = resolve_path(config, dataset.get("prepared_dir", "../dataset/prepared/brset_v1_0_2"))
    paths = {
        "root": root,
        "labels_csv": labels_value if labels_value.is_absolute() else root / labels_value,
        "images_dir": images_value if images_value.is_absolute() else root / images_value,
        "prepared_dir": prepared,
        "split_manifest": prepared / "split_manifest.csv",
        "metadata_schema": prepared / "metadata_schema.json",
        "audit": prepared / "data_audit.json",
    }
    required = ["labels_csv", "images_dir"]
    if require_prepared:
        required.extend(("split_manifest", "metadata_schema", "audit"))
    missing = [name for name in required if not paths[name].exists()]
    if missing:
        details = ", ".join(f"{name}={paths[name]}" for name in missing)
        raise FileNotFoundError(f"missing BRSET inputs: {details}")
    return paths
