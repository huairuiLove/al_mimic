"""Configuration loading for the registered MIMIC-III multi-label tasks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .tasks import task_spec

FORMAL_STRATEGIES = {"comal", "modis", "modimix", "mosaic", "random"}
DEFAULT_PROTOCOL_PROFILE = "yang_wu_diagnoses_48h"
PROFILE_KEYS = (
    "observation_hours",
    "timestep_hours",
    "max_note_tokens",
    "expected_total_samples",
    "expected_label_count",
    "time_invariant_dim",
    "time_series_dim",
    "note_protocol",
)
SCENARIO_SECTIONS = {"name", "label_subset", "missing_notes", "missing_series"}
LABEL_SUBSET_KEYS = {"drop_top_k", "min_train_positives"}
MISSING_MODALITY_KEYS = {"rate", "bias", "strength", "seed", "splits", "careunit"}
COHORT_MODES = {"official", "full_cohort"}


def _require_equal(name: str, actual: Any, required: Any) -> None:
    if actual != required:
        raise ValueError(f"formal task protocol requires {name}={required!r}, got {actual!r}")


def _load_protocol_profile(name: str) -> dict[str, Any]:
    path = Path(__file__).with_name("protocol_profiles.yaml")
    with path.open(encoding="utf-8") as handle:
        profiles = yaml.safe_load(handle) or {}
    if name not in profiles:
        raise ValueError(
            f"unknown preprocessing.protocol_profile {name!r}; known profiles: {sorted(profiles)}"
        )
    profile = profiles[name] or {}
    unresolved = [key for key in PROFILE_KEYS if profile.get(key) is None]
    if unresolved:
        raise ValueError(
            f"protocol profile {name!r} has unresolved dimensions {unresolved}; "
            "run scripts/build_scenario.py to materialise the cohort and register them"
        )
    return profile


def _validate_scenario(config: dict[str, Any]) -> None:
    """Fail loudly on a mistyped scenario key rather than silently ignoring it."""
    scenario = config.get("scenario")
    if scenario is None:
        return
    if not isinstance(scenario, dict):
        raise ValueError("scenario must be a mapping")
    unknown = set(scenario) - SCENARIO_SECTIONS
    if unknown:
        raise ValueError(f"unknown scenario sections: {sorted(unknown)}")
    for section, allowed in (
        ("label_subset", LABEL_SUBSET_KEYS),
        ("missing_notes", MISSING_MODALITY_KEYS),
        ("missing_series", MISSING_MODALITY_KEYS),
    ):
        settings = scenario.get(section) or {}
        if not isinstance(settings, dict):
            raise ValueError(f"scenario.{section} must be a mapping")
        unknown = set(settings) - allowed
        if unknown:
            raise ValueError(f"unknown scenario.{section} keys: {sorted(unknown)}")


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
    cohort_mode = str(dataset.get("cohort_mode", "official")).lower()
    if cohort_mode not in COHORT_MODES:
        raise ValueError(f"dataset.cohort_mode must be one of {sorted(COHORT_MODES)}")

    split_protocol = str(preprocessing.get("split_protocol", ""))
    expected_split_protocol = (
        "yang_wu_official_7_1.5_1.5" if cohort_mode == "official" else "yang_wu_full_cohort_grouped"
    )
    _require_equal("preprocessing.split_protocol", split_protocol, expected_split_protocol)
    expected_cohort = "metavision" if cohort_mode == "official" else "mimic_iii_all_icu"
    _require_equal("preprocessing.cohort", preprocessing.get("cohort"), expected_cohort)
    profile_name = preprocessing.get("protocol_profile") or (
        DEFAULT_PROTOCOL_PROFILE if cohort_mode == "official" else None
    )
    profile = _load_protocol_profile(str(profile_name)) if profile_name is not None else None

    required_values = {
        "dataset.split_group": (dataset.get("split_group"), "with_notes"),
        "preprocessing.task": (preprocessing.get("task"), "Diagnoses"),
        "preprocessing.label_format": (
            preprocessing.get("label_format"),
            "icd9_top3_multihot",
        ),
        "preprocessing.note_protocol": (
            preprocessing.get("note_protocol"),
            profile["note_protocol"] if profile is not None else "latest_per_category_description_within_48h",
        ),
        "model.architecture": (model.get("architecture"), "yang_wu_bertencoder"),
        "model.initialization": (
            model.get("initialization"),
            "clinicalbert_pretrained_fresh_fusion_each_round",
        ),
        "model.output_activation": (model.get("output_activation"), "sigmoid"),
        "training.optimizer": (training.get("optimizer"), "adamw"),
        "training.loss": (training.get("loss"), "binary_cross_entropy"),
        "training.precision": (training.get("precision"), "fp32"),
    }
    for name, (actual, required) in required_values.items():
        _require_equal(name, actual, required)

    if profile is None:
        integer_values = {
            "preprocessing.observation_hours": (preprocessing.get("observation_hours"), 48),
            "preprocessing.timestep_hours": (preprocessing.get("timestep_hours"), 1),
            "preprocessing.max_note_tokens": (preprocessing.get("max_note_tokens"), 512),
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
        }
    else:
        integer_values = {
            f"preprocessing.{key}": (preprocessing.get(key), profile[key])
            for key in PROFILE_KEYS
            if key != "note_protocol"
        }
    integer_values |= {
        "model.output_size": (
            model.get("output_size"),
            profile["expected_label_count"] if profile is not None else 915,
        ),
        "model.text_hidden_dim": (model.get("text_hidden_dim"), 768),
        "model.time_invariant_hidden_dim": (
            model.get("time_invariant_hidden_dim"),
            64,
        ),
        "model.time_series_hidden_dim": (model.get("time_series_hidden_dim"), 1024),
        "model.time_series_layers": (model.get("time_series_layers"), 3),
        "model.time_series_heads": (model.get("time_series_heads"), 16),
        "training.epochs": (training.get("epochs"), 80),
        "training.optimizer_steps_per_round": (
            training.get("optimizer_steps_per_round"),
            1200,
        ),
        "training.early_stopping_patience": (
            training.get("early_stopping_patience"),
            5,
        ),
        "active_learning.rounds": (active.get("rounds"), 9),
    }
    for name, (actual, required) in integer_values.items():
        try:
            normalized = int(actual)
        except (TypeError, ValueError):
            normalized = actual
        _require_equal(name, normalized, required)

    expected_total_samples = preprocessing.get("expected_total_samples")
    if cohort_mode == "official":
        _require_equal(
            "preprocessing.expected_total_samples",
            expected_total_samples,
            profile["expected_total_samples"] if profile is not None else 10258,
        )
    elif expected_total_samples is not None:
        try:
            normalized_total = int(expected_total_samples)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "full_cohort preprocessing.expected_total_samples must be null or a positive integer"
            ) from exc
        if normalized_total < 1:
            raise ValueError("full_cohort preprocessing.expected_total_samples must be positive")

    float_values = {
        "model.dropout": (model.get("dropout"), 0.1),
        "training.learning_rate": (training.get("learning_rate"), 1e-4),
        "training.bert_learning_rate": (training.get("bert_learning_rate"), 2e-5),
        "training.bert_layerwise_lr_decay": (
            training.get("bert_layerwise_lr_decay"),
            0.95,
        ),
        "training.weight_decay": (training.get("weight_decay"), 0.01),
        "training.early_stopping_min_delta": (
            training.get("early_stopping_min_delta"),
            1e-4,
        ),
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
    _validate_scenario(config)
    _reject_shortcuts(config)


def _validate_common_task_protocol(config: dict[str, Any]) -> None:
    spec = task_spec(config)
    task = config.get("task", {})
    preprocessing = config.get("preprocessing", {})
    model = config.get("model", {})
    training = config.get("training", {})
    active = config.get("active_learning", {})
    evaluation = config.get("evaluation", {})

    required_values = {
        "task.native_multilabel": (task.get("native_multilabel"), True),
        "task.query_unit": (task.get("query_unit"), spec.query_unit),
        "preprocessing.label_format": (
            preprocessing.get("label_format"),
            spec.label_format,
        ),
        "model.output_activation": (model.get("output_activation"), "sigmoid"),
        "training.optimizer": (training.get("optimizer"), "adamw"),
        "training.loss": (training.get("loss"), "binary_cross_entropy"),
        "evaluation.primary_metric": (
            evaluation.get("primary_metric"),
            spec.primary_metric,
        ),
        "evaluation.metrics": (tuple(evaluation.get("metrics", ())), spec.metrics),
    }
    for name, (actual, required) in required_values.items():
        _require_equal(name, actual, required)

    for name, actual in {
        "preprocessing.expected_label_count": preprocessing.get("expected_label_count"),
        "model.output_size": model.get("output_size"),
    }.items():
        try:
            normalized = int(actual)
        except (TypeError, ValueError):
            normalized = actual
        _require_equal(name, normalized, spec.label_count)

    strategy = str(active.get("strategy", "")).lower()
    if strategy not in FORMAL_STRATEGIES:
        raise ValueError(f"active_learning.strategy must be one of {sorted(FORMAL_STRATEGIES)}")
    if strategy == "modimix":
        mixup = config.get("mixup") or {}
        if not isinstance(mixup, dict):
            raise ValueError("ModiMix requires mixup to be a mapping")
        if not bool(mixup.get("enabled", False)) or str(mixup.get("space", "")).lower() != "modalities":
            raise ValueError("ModiMix requires enabled modality-space mixup")
    if bool(training.get("inherit_across_rounds", False)):
        raise ValueError("formal protocol forbids cross-round weight inheritance")
    _reject_shortcuts(config)


def _validate_phenotyping_protocol(config: dict[str, Any]) -> None:
    spec = task_spec(config)
    dataset = config.get("dataset", {})
    preprocessing = config.get("preprocessing", {})
    model = config.get("model", {})

    required_values = {
        "dataset.split_group": (dataset.get("split_group"), "with_notes"),
        "preprocessing.task": (preprocessing.get("task"), "Phenotyping"),
        "preprocessing.note_protocol": (
            preprocessing.get("note_protocol"),
            "all_stay_notes_chronological_512",
        ),
        "model.architecture": (
            model.get("architecture"),
            "clinicalbert_measurement_fusion",
        ),
        "model.initialization": (
            model.get("initialization"),
            "clinicalbert_pretrained_fresh_fusion_each_round",
        ),
        "model.time_series_pooling": (model.get("time_series_pooling"), "masked_mean"),
    }
    expected_split = {
        "phenotyping_25": "mimic3_benchmark_subject_split",
        "phenotyping_ccs_239": "notes_benchmark_subject_split",
    }[spec.task_id]
    required_values["preprocessing.split_protocol"] = (
        preprocessing.get("split_protocol"),
        expected_split,
    )
    for name, (actual, required) in required_values.items():
        _require_equal(name, actual, required)

    integer_values = {
        "preprocessing.timestep_hours": (preprocessing.get("timestep_hours"), 1),
        "preprocessing.max_time_steps": (preprocessing.get("max_time_steps"), 256),
        "preprocessing.max_note_tokens": (preprocessing.get("max_note_tokens"), 512),
        "preprocessing.time_invariant_dim": (
            preprocessing.get("time_invariant_dim"),
            0,
        ),
        "preprocessing.time_series_dim": (preprocessing.get("time_series_dim"), 76),
        "model.text_hidden_dim": (model.get("text_hidden_dim"), 768),
        "model.time_invariant_hidden_dim": (
            model.get("time_invariant_hidden_dim"),
            0,
        ),
    }
    for name, (actual, required) in integer_values.items():
        try:
            normalized = int(actual)
        except (TypeError, ValueError):
            normalized = actual
        _require_equal(name, normalized, required)

    expected_total = preprocessing.get("expected_total_samples")
    if expected_total is not None and int(expected_total) < 1:
        raise ValueError("preprocessing.expected_total_samples must be null or positive")
    if not str(dataset.get("clinicalbert_checkpoint", "")).strip():
        raise ValueError("dataset.clinicalbert_checkpoint must name the ClinicalBERT checkpoint")


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
    # A list of parents lets a run combine an orthogonal pair of bases -- the
    # method's own tuning and a dataset scenario -- without either being
    # restated, and so without one of them silently going missing.
    parents = config.pop("extends", None) or []
    if isinstance(parents, str):
        parents = [parents]
    if parents:
        inherited: dict[str, Any] = {}
        for entry in parents:
            parent_path = Path(entry)
            if not parent_path.is_absolute():
                parent_path = path.parent / parent_path
            inherited = _merge(inherited, load_config(parent_path))
        inherited.pop("_config_path", None)
        config = _merge(inherited, config)
    config["_config_path"] = str(path.resolve())
    _validate_common_task_protocol(config)
    if task_spec(config).task_id == "icd9_diagnoses":
        _validate_yang_wu_protocol(config)
    else:
        _validate_phenotyping_protocol(config)
    return config


def resolve_path(config: dict[str, Any], value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    config_path = Path(config.get("_config_path", ".")).resolve()
    return (config_path.parent / path).resolve()


def require_paths(config: dict[str, Any]) -> dict[str, Path]:
    dataset = config.get("dataset", {})
    paths = {
        "split_hdf5": resolve_path(config, dataset.get("split_hdf5", "")),
        "clinicalbert_checkpoint": resolve_path(config, dataset.get("clinicalbert_checkpoint", "")),
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        details = ", ".join(f"{name}={paths[name]}" for name in missing)
        raise FileNotFoundError(f"missing formal MIMIC task inputs: {details}")
    return paths
