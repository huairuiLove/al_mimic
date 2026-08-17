"""Patient-query active-learning runner for the first-party BRSET task."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from al_mimic.core.methods import fit_method, prepare_method_context
from al_mimic.methods import get_method

from .config import data_paths, resolve_path
from .data import LABEL_COLUMNS, BrsetFeatureStore
from .metrics import fit_f1_thresholds, multilabel_metrics
from .training import TrainedBrsetRound, collect_outputs, collect_patient_outputs, train_round


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import orjson
    except ImportError:  # pragma: no cover - optional acceleration
        path.write_text(
            json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    else:
        path.write_bytes(orjson.dumps(value, option=orjson.OPT_INDENT_2) + b"\n")


def labeled_schedule(
    train_size: int,
    *,
    rounds: int = 6,
    initial_fraction: float = 0.10,
    query_fraction: float = 0.05,
) -> list[int]:
    """Return exact cumulative patient budgets with half-up rounding."""
    if train_size < 1 or rounds < 1:
        raise ValueError("train_size and rounds must be positive")
    if not 0 < initial_fraction <= 1 or query_fraction < 0:
        raise ValueError("active-learning fractions must be positive and bounded")
    targets = [
        int(np.floor(train_size * (initial_fraction + query_fraction * index) + 0.5))
        for index in range(rounds)
    ]
    if targets[0] < 1 or targets[-1] > train_size:
        raise ValueError("active-learning fractions exceed the patient train pool")
    if any(right <= left for left, right in zip(targets, targets[1:])):
        raise ValueError("active-learning fractions do not produce an increasing patient schedule")
    return targets


def _initial_patients(train_patients: Sequence[str], size: int, seed: int) -> list[str]:
    if size < 1 or size > len(train_patients):
        raise ValueError("initial BRSET patient budget is outside the train pool")
    generator = np.random.default_rng(seed)
    selected = generator.choice(np.asarray(train_patients, dtype=object), size=size, replace=False)
    return sorted(str(value) for value in selected)


def _host(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().float().cpu().numpy()


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(child) for child in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _score_summary(scores: Mapping[str, Any], selected_positions: np.ndarray) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for name, values in scores.items():
        if not isinstance(values, torch.Tensor) or values.ndim != 1:
            continue
        selected = values.index_select(
            0,
            torch.as_tensor(selected_positions, dtype=torch.long, device=values.device),
        )
        summary[str(name)] = {
            "pool_mean": float(values.float().mean()),
            "selected_mean": float(selected.float().mean()),
        }
    return summary


def _validate_patient_outputs(
    name: str,
    patient_ids: Sequence[str],
    outputs: Mapping[str, torch.Tensor],
) -> None:
    identifiers = tuple(str(patient_id) for patient_id in patient_ids)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"{name} patient IDs must be unique")
    required = {"labels", "probabilities", "features", "modality_tokens"}
    missing = sorted(required - outputs.keys())
    if missing:
        raise ValueError(f"{name} patient outputs are missing fields: {missing}")
    mismatched = [
        field
        for field in required
        if not isinstance(outputs[field], torch.Tensor)
        or outputs[field].ndim == 0
        or int(outputs[field].shape[0]) != len(identifiers)
    ]
    if mismatched:
        raise ValueError(f"{name} outputs are not patient-row-aligned: {sorted(mismatched)}")


def build_method_context(
    *,
    classifier: Any,
    candidate_ids: Sequence[str],
    labeled_ids: Sequence[str],
    reference_ids: Sequence[str],
    candidate_outputs: dict[str, torch.Tensor],
    labeled_outputs: dict[str, torch.Tensor],
    reference_outputs: dict[str, torch.Tensor],
    query_size: int,
    config: dict[str, Any],
    seed: int,
    round_index: int,
    initial_prevalence: torch.Tensor,
) -> dict[str, Any]:
    """Build the BRSET-to-method adapter with patient-level tensors only."""
    candidate_ids = tuple(str(patient_id) for patient_id in candidate_ids)
    labeled_ids = tuple(str(patient_id) for patient_id in labeled_ids)
    reference_ids = tuple(str(patient_id) for patient_id in reference_ids)
    _validate_patient_outputs("candidate", candidate_ids, candidate_outputs)
    _validate_patient_outputs("labeled", labeled_ids, labeled_outputs)
    _validate_patient_outputs("reference", reference_ids, reference_outputs)
    overlap = {
        "candidate_labeled": set(candidate_ids) & set(labeled_ids),
        "candidate_reference": set(candidate_ids) & set(reference_ids),
        "labeled_reference": set(labeled_ids) & set(reference_ids),
    }
    leaking = {name: sorted(values) for name, values in overlap.items() if values}
    if leaking:
        raise ValueError(f"BRSET method patient pools must be disjoint: {leaking}")
    if query_size < 0 or query_size > len(candidate_ids):
        raise ValueError("query_size must fit the patient candidate pool")

    context: dict[str, Any] = {
        "config": config,
        "classifier": classifier,
        "candidate_ids": candidate_ids,
        "candidate_patient_ids": candidate_ids,
        "labeled_patient_ids": labeled_ids,
        "reference_patient_ids": reference_ids,
        "query_size": int(query_size),
        "seed": int(seed),
        "round_index": int(round_index),
        "candidate_outputs": candidate_outputs,
        "labeled_outputs": labeled_outputs,
        "validation_outputs": reference_outputs,
        "reference_outputs": reference_outputs,
        "groups": labeled_ids,
        "initial_prevalence": initial_prevalence,
        "probabilities": candidate_outputs["probabilities"],
        "modality_tokens": candidate_outputs["modality_tokens"],
        "labeled_labels": labeled_outputs["labels"],
    }
    return context


def acquire_method(plugin: Any, context: Mapping[str, Any]) -> tuple[list[str], dict[str, Any]]:
    """Acquire and enforce the exact patient-ID contract for every method."""
    result = plugin.acquire(context)
    selected_ids = getattr(result, "selected_ids", None)
    selected_positions = getattr(result, "selected_positions", None)
    if selected_ids is None or selected_positions is None:
        raise TypeError("method acquire(context) must return selected_ids and selected_positions")
    candidates = tuple(str(patient_id) for patient_id in context["candidate_ids"])
    queries = [str(patient_id) for patient_id in selected_ids]
    positions = np.asarray(selected_positions, dtype=np.int64)
    expected = int(context["query_size"])
    if positions.ndim != 1 or len(queries) != expected or positions.size != expected:
        raise ValueError("method did not return the exact patient query budget")
    if len(set(queries)) != expected or len(np.unique(positions)) != expected:
        raise ValueError("method must return unique patient IDs and positions")
    if np.any(positions < 0) or np.any(positions >= len(candidates)):
        raise ValueError("method selected position is outside the patient candidate pool")
    positioned_ids = [candidates[int(position)] for position in positions]
    if queries != positioned_ids:
        raise ValueError("method selected patient IDs do not match selected positions")
    if set(queries) & set(context["labeled_patient_ids"]):
        raise ValueError("method reselected a labeled patient")
    scores = getattr(result, "scores", {})
    diagnostics = getattr(result, "diagnostics", {})
    return queries, {
        "method": str(getattr(result, "diagnostics", {}).get("method", "unknown")),
        "query_unit": "patient",
        "candidate_patients": len(candidates),
        "selected_patients": len(queries),
        "score_components": _score_summary(scores, positions) if isinstance(scores, Mapping) else {},
        "method_diagnostics": _jsonable(diagnostics),
    }


class BrsetActiveLearningExperiment:
    """Cold-start BRSET experiments with task-independent method plugins."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.training = config.get("training", {})
        self.active = config.get("active_learning", {})
        self.store = BrsetFeatureStore(config, validate=True)
        requested_device = str(self.training.get("device", "cuda"))
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("formal BRSET configuration requires CUDA, but CUDA is unavailable")
        self.device = torch.device(requested_device)
        if self.device.type == "cuda":
            allow_tf32 = bool(self.training.get("allow_tf32", False))
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
            torch.backends.cudnn.allow_tf32 = allow_tf32
            torch.backends.cudnn.benchmark = bool(self.training.get("cudnn_benchmark", False))
        experiment = config.get("experiment", {})
        output_root = resolve_path(config, experiment.get("output_root", "../experiments"))
        self.output_dir = output_root / str(experiment.get("name", "brset_v1_0_2_resnet50_metadata_comal"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = int(self.training.get("seed", 17))

    def _save_checkpoint(
        self,
        trained: TrainedBrsetRound,
        round_index: int,
        labeled_patients: Sequence[str],
        strategy: str,
    ) -> None:
        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        labeled_indices = self.store.indices_for_patients(labeled_patients)
        payload: dict[str, Any] = {
            "round": round_index,
            "labeled_patient_count": len(labeled_patients),
            "labeled_image_count": int(labeled_indices.size),
            "classifier": {
                key: value.detach().cpu() for key, value in trained.classifier.state_dict().items()
            },
            "strategy": strategy,
            "query_unit": "patient",
            "target_model": "ImageNet ResNet-50 + BRSET clinical metadata MLP",
            "initialization": "fresh ImageNet ResNet-50 and fresh fusion layers",
            "classifier_epochs": int(self.training["epochs"]),
        }
        round_path = checkpoint_dir / f"round_{round_index:03d}.pt"
        torch.save(payload, round_path)
        if round_index == int(self.active["rounds"]) - 1:
            torch.save(payload, checkpoint_dir / "final.pt")

    def _method_context(
        self,
        *,
        plugin: Any,
        trained: TrainedBrsetRound,
        labeled_patients: Sequence[str],
        candidate_patients: Sequence[str],
        reference_patients: Sequence[str],
        initial_patients: Sequence[str],
        query_size: int,
        round_index: int,
    ) -> dict[str, Any]:
        labeled_outputs = collect_patient_outputs(
            trained.classifier,
            self.store,
            labeled_patients,
            self.config,
            self.device,
        )
        candidate_outputs = collect_patient_outputs(
            trained.classifier,
            self.store,
            candidate_patients,
            self.config,
            self.device,
        )
        reference_outputs = collect_patient_outputs(
            trained.classifier,
            self.store,
            reference_patients,
            self.config,
            self.device,
        )
        context = build_method_context(
            classifier=trained.classifier,
            candidate_ids=candidate_patients,
            labeled_ids=labeled_patients,
            reference_ids=reference_patients,
            candidate_outputs=candidate_outputs,
            labeled_outputs=labeled_outputs,
            reference_outputs=reference_outputs,
            query_size=query_size,
            config=self.config,
            seed=self.seed + round_index,
            round_index=round_index,
            initial_prevalence=torch.as_tensor(
                self.store.patient_targets(initial_patients).mean(axis=0),
                dtype=torch.float32,
                device=self.device,
            ),
        )
        start = time.perf_counter()
        trained.method_state = fit_method(plugin, context)
        if trained.method_state is not None:
            trained.timings["method_auxiliary_training_sec"] = time.perf_counter() - start
            history = getattr(trained.method_state, "history", None)
            if history is not None:
                trained.history["method_auxiliary_loss"] = list(history)
        return prepare_method_context(plugin, context, trained.method_state)

    def run_full_data(self) -> dict[str, Any]:
        """Train once on all train patients and evaluate image-level predictions."""
        train_patients = self.store.patient_ids("train")
        trained = train_round(self.store, train_patients, self.config, self.device)
        validation = collect_outputs(
            trained.classifier,
            self.store,
            self.store.indices("val"),
            self.config,
            self.device,
            return_tokens=False,
        )
        test = collect_outputs(
            trained.classifier,
            self.store,
            self.store.indices("test"),
            self.config,
            self.device,
            return_tokens=False,
        )
        thresholds = fit_f1_thresholds(_host(validation["labels"]), _host(validation["probabilities"]))
        final = {
            "validation": multilabel_metrics(
                _host(validation["labels"]), _host(validation["probabilities"]), thresholds
            ),
            "test": multilabel_metrics(_host(test["labels"]), _host(test["probabilities"]), thresholds),
            "strategy": "full_data_upper_bound",
            "query_unit": "patient",
        }
        output_dir = self.output_dir / "full_data"
        output_dir.mkdir(parents=True, exist_ok=True)
        _write_json(output_dir / "final_metrics.json", final)
        return {"output_dir": str(output_dir), "final_metrics": final}

    def run(self) -> dict[str, Any]:
        random.seed(self.seed)
        np.random.seed(self.seed)
        train_patients = self.store.patient_ids("train")
        validation_patients = self.store.patient_ids("val")
        validation_indices = self.store.indices("val")
        test_indices = self.store.indices("test")
        rounds = int(self.active["rounds"])
        schedule = labeled_schedule(
            len(train_patients),
            rounds=rounds,
            initial_fraction=float(self.active["initial_fraction"]),
            query_fraction=float(self.active["query_fraction"]),
        )
        labeled = _initial_patients(train_patients, schedule[0], self.seed)
        initial = list(labeled)
        strategy = str(self.active["strategy"]).lower()
        plugin = get_method(strategy)
        records: list[dict[str, Any]] = []
        total_start = time.perf_counter()
        final_validation: dict[str, torch.Tensor] | None = None
        final_test: dict[str, torch.Tensor] | None = None
        final_thresholds: np.ndarray | None = None

        for round_index, target_count in enumerate(schedule):
            if len(labeled) != target_count:
                raise RuntimeError(
                    f"round {round_index} expected {target_count} patients, got {len(labeled)}"
                )
            round_start = time.perf_counter()
            trained = train_round(self.store, labeled, self.config, self.device)
            validation = collect_outputs(
                trained.classifier,
                self.store,
                validation_indices,
                self.config,
                self.device,
                return_tokens=False,
            )
            test = collect_outputs(
                trained.classifier,
                self.store,
                test_indices,
                self.config,
                self.device,
                return_tokens=False,
            )
            thresholds = fit_f1_thresholds(_host(validation["labels"]), _host(validation["probabilities"]))
            validation_metrics = multilabel_metrics(
                _host(validation["labels"]), _host(validation["probabilities"]), thresholds
            )
            test_metrics = multilabel_metrics(_host(test["labels"]), _host(test["probabilities"]), thresholds)
            final_validation, final_test, final_thresholds = validation, test, thresholds
            queries: list[str] = []
            acquisition: dict[str, Any] = {"method": "none (final training round)"}
            if round_index + 1 < rounds:
                labeled_set = set(labeled)
                candidates = [patient for patient in train_patients if patient not in labeled_set]
                query_size = schedule[round_index + 1] - len(labeled)
                context = self._method_context(
                    plugin=plugin,
                    trained=trained,
                    labeled_patients=labeled,
                    candidate_patients=candidates,
                    reference_patients=validation_patients,
                    initial_patients=initial,
                    query_size=query_size,
                    round_index=round_index,
                )
                queries, acquisition = acquire_method(plugin, context)

            labeled_indices = self.store.indices_for_patients(labeled)
            query_indices = self.store.indices_for_patients(queries)
            self._save_checkpoint(trained, round_index, labeled, strategy)
            records.append(
                {
                    "round_index": round_index,
                    "labeled_patient_count": len(labeled),
                    "labeled_image_count": int(labeled_indices.size),
                    "labeled_fraction_of_train_patients": len(labeled) / len(train_patients),
                    "query_patient_count": len(queries),
                    "query_image_count": int(query_indices.size),
                    "query_patient_ids": queries,
                    "query_image_ids": [str(self.store.image_ids[index]) for index in query_indices],
                    "validation_thresholds": {
                        name: float(thresholds[index]) for index, name in enumerate(LABEL_COLUMNS)
                    },
                    "validation_metrics": validation_metrics,
                    "test_metrics": test_metrics,
                    "training_history": trained.history,
                    "timing": trained.timings | {"round_total_sec": time.perf_counter() - round_start},
                    "acquisition": acquisition,
                }
            )
            if queries:
                labeled = sorted([*labeled, *queries])

        assert final_validation is not None and final_test is not None and final_thresholds is not None
        train_indices = self.store.indices("train")
        initial_indices = self.store.indices_for_patients(initial)
        final_indices = self.store.indices_for_patients(labeled)
        data_usage = {
            "dataset_version": "BRSET v1.0.2",
            "full_dataset_images": self.store.audit.images,
            "full_dataset_patients": self.store.audit.patients,
            "train_pool_images": int(train_indices.size),
            "train_pool_patients": len(train_patients),
            "validation_images": int(validation_indices.size),
            "validation_patients": len(validation_patients),
            "test_images": int(test_indices.size),
            "test_patients": len(self.store.patient_ids("test")),
            "initial_labeled_patients": len(initial),
            "initial_labeled_images": int(initial_indices.size),
            "newly_queried_patients": len(labeled) - len(initial),
            "final_labeled_patients": len(labeled),
            "final_labeled_images": int(final_indices.size),
            "final_fraction_of_train_patients": len(labeled) / len(train_patients),
            "cumulative_patient_targets": schedule,
        }
        state = {
            "format_version": 2,
            "protocol": "BRSET v1.0.2 multimodal 13-label patient-level active learning",
            "strategy": strategy,
            "seed": self.seed,
            "labeled_patient_schedule": schedule,
            "initial_patient_ids": initial,
            "final_labeled_patient_ids": labeled,
            "records": records,
            "data_usage": data_usage,
            "total_wall_sec": time.perf_counter() - total_start,
        }
        np.savez(
            self.output_dir / "final_predictions.npz",
            validation_labels=_host(final_validation["labels"]),
            validation_probabilities=_host(final_validation["probabilities"]),
            test_labels=_host(final_test["labels"]),
            test_probabilities=_host(final_test["probabilities"]),
            validation_f1_thresholds=final_thresholds,
        )
        _write_json(self.output_dir / "active_state.json", state)
        final = {
            "validation": records[-1]["validation_metrics"],
            "test": records[-1]["test_metrics"],
            "data_usage": data_usage,
            "strategy": strategy,
            "query_unit": "patient",
        }
        _write_json(self.output_dir / "final_metrics.json", final)
        _write_json(self.output_dir / "resolved_config.json", self.config)
        _write_json(
            self.output_dir / "source_audit.json",
            json.loads(data_paths(self.config, require_prepared=True)["audit"].read_text()),
        )
        return {"output_dir": str(self.output_dir), "final_metrics": final, "rounds": rounds}


__all__ = [
    "BrsetActiveLearningExperiment",
    "acquire_method",
    "build_method_context",
    "fit_method",
    "labeled_schedule",
]
