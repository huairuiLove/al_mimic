"""Runnable active-learning runner for the native MIMIC-III task plugin."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from al_mimic.core.methods import fit_method, prepare_method_context
from al_mimic.methods import get_method

from .config import resolve_path
from .data import YangWuFeatureStore
from .metrics import task_multilabel_metrics
from .tasks import task_manifest, task_spec
from .training import (
    TrainedMultimodalRound,
    collect_classifier_outputs,
    train_multimodal_round,
)


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
    """Return cumulative, half-up rounded label targets for each round."""
    if train_size < 1 or rounds < 1:
        raise ValueError("train_size and rounds must be positive")
    if not 0 < initial_fraction <= 1 or query_fraction < 0:
        raise ValueError("active-learning fractions must be positive and bounded")
    targets = [
        int(np.floor(train_size * (initial_fraction + query_fraction * index) + 0.5))
        for index in range(rounds)
    ]
    if targets[0] < 1 or targets[-1] > train_size:
        raise ValueError("active-learning fractions exceed the train pool")
    if any(right <= left for left, right in zip(targets, targets[1:])):
        raise ValueError("active-learning fractions do not produce an increasing schedule")
    return targets


def _initial_indices(train_indices: np.ndarray, size: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    return sorted(int(value) for value in rng.choice(train_indices, size=size, replace=False))


def _host(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().float().cpu().numpy()


def _numeric_score_summary(
    scores: Mapping[str, Any], selected_positions: np.ndarray
) -> dict[str, dict[str, float]]:
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


def _jsonable(value: Any) -> Any:
    """Convert plugin diagnostics to JSON without knowing method internals."""
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


def _method_context(
    *,
    experiment: "ActiveLearningExperiment",
    trained: TrainedMultimodalRound,
    labeled_indices: np.ndarray,
    candidates: np.ndarray,
    query_size: int,
    round_index: int,
    initial_indices: np.ndarray,
    candidate_outputs: dict[str, torch.Tensor] | None = None,
    labeled_outputs: dict[str, torch.Tensor] | None = None,
    validation_outputs: dict[str, torch.Tensor] | None = None,
) -> dict[str, Any]:
    """Build the stable task-to-method hook contract."""
    context = {
        "experiment": experiment,
        "config": experiment.config,
        "task": experiment.task,
        "store": experiment.store,
        "trained": trained,
        "classifier": trained.classifier,
        "method_state": trained.method_state,
        "labeled_indices": labeled_indices,
        "initial_indices": initial_indices,
        "candidate_ids": tuple(int(value) for value in candidates),
        "candidate_indices": candidates,
        "candidate_outputs": candidate_outputs,
        "labeled_outputs": labeled_outputs,
        "validation_outputs": validation_outputs,
        "reference_outputs": validation_outputs,
        "query_size": int(query_size),
        "round_index": int(round_index),
        "seed": int(experiment.seed),
        "groups": None if labeled_outputs is None else labeled_outputs.get("subject_ids"),
        "candidate_metadata": None,
        "initial_prevalence": torch.as_tensor(
            experiment.store.labels[np.asarray(initial_indices, dtype=np.int64)].mean(axis=0),
            dtype=torch.float32,
            device=experiment.device,
        ),
    }
    if candidate_outputs is not None:
        context["probabilities"] = candidate_outputs.get("probabilities")
        context["modality_tokens"] = candidate_outputs.get("modality_tokens")
        context["own_similarity"] = candidate_outputs.get("own_similarity")
        context["view_own_similarity"] = candidate_outputs.get("view_own_similarity")
    if labeled_outputs is not None:
        context["labeled_labels"] = labeled_outputs.get("labels")
        context["labeled_own_similarity"] = labeled_outputs.get("own_similarity")
        context["labeled_view_own_similarity"] = labeled_outputs.get("view_own_similarity")
    return context


def _acquire_method(plugin: Any, context: dict[str, Any]) -> dict[str, Any]:
    required = tuple(getattr(plugin, "required_context_fields", ()))
    missing = [name for name in required if context.get(name) is None]
    if missing:
        method_id = str(getattr(plugin, "method_id", "unknown"))
        raise RuntimeError(f"method {method_id!r} requires unresolved core hook fields: {missing}")
    result = plugin.acquire(context)
    selected_ids = getattr(result, "selected_ids", None)
    selected_positions = getattr(result, "selected_positions", None)
    if selected_ids is None and isinstance(result, Mapping):
        selected_ids = result.get("selected_ids")
        selected_positions = result.get("selected_positions")
    if selected_ids is None or selected_positions is None:
        raise TypeError("method acquire(context) must return selected_ids and selected_positions")
    candidates = np.asarray(context["candidate_indices"], dtype=np.int64)
    positions = np.asarray(selected_positions, dtype=np.int64)
    if positions.ndim != 1 or np.any(positions < 0) or np.any(positions >= candidates.size):
        raise ValueError("method selected_positions must index the candidate pool")
    queries = [int(value) for value in np.asarray(selected_ids, dtype=np.int64).tolist()]
    expected = int(context["query_size"])
    if len(queries) != expected or len(np.unique(queries)) != expected:
        raise ValueError("method did not return the exact unique query budget")
    if set(queries) != set(candidates[positions].tolist()):
        raise ValueError("method selected_ids do not match selected_positions")
    scores = getattr(result, "scores", {})
    diagnostics = getattr(result, "diagnostics", {})
    if isinstance(result, Mapping):
        scores = result.get("scores", scores)
        diagnostics = result.get("diagnostics", diagnostics)
    return {
        "queries": sorted(queries),
        "method": str(getattr(plugin, "method_id", context["config"]["active_learning"]["strategy"])),
        "candidate_count": int(candidates.size),
        "selected_count": len(queries),
        "score_components": _numeric_score_summary(scores, positions) if isinstance(scores, Mapping) else {},
        "method_diagnostics": _jsonable(diagnostics),
    }


class ActiveLearningExperiment:
    """Task-owned experiment orchestration with method plugins at the edge."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.task = task_spec(config)
        self.training = config.get("training", {})
        self.active = config.get("active_learning", {})
        self.store = YangWuFeatureStore(config, validate=True)
        requested_device = str(self.training.get("device", "cuda"))
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("formal configuration requires CUDA, but no CUDA device is available")
        self.device = torch.device(requested_device)
        experiment = config.get("experiment", {})
        output_root = resolve_path(config, experiment.get("output_root", "../../../experiments"))
        self.output_dir = output_root / str(experiment.get("name", "mimic_iii_yang_wu"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = int(self.training.get("seed", 17))

    def _save_checkpoint(self, trained: TrainedMultimodalRound, round_index: int, labeled_count: int) -> None:
        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        payload: dict[str, Any] = {
            "round": round_index,
            "labeled_count": labeled_count,
            "classifier": {
                key: value.detach().cpu() for key, value in trained.classifier.state_dict().items()
            },
            "target_model": str(self.config.get("model", {}).get("architecture")),
            "task": task_manifest(self.config),
            "initialization": "fresh from the same ClinicalBERT source checkpoint",
            "classifier_epochs": int(trained.training_summary["epochs_ran"]),
            "training_summary": trained.training_summary,
        }
        round_path = checkpoint_dir / f"round_{round_index:03d}.pt"
        torch.save(payload, round_path)
        if round_index == int(self.active["rounds"]) - 1:
            torch.save(payload, checkpoint_dir / "final.pt")

    def _outputs_for_method(
        self,
        trained: TrainedMultimodalRound,
        labeled_indices: np.ndarray,
        candidates: np.ndarray,
        validation_indices: np.ndarray,
        *,
        return_tokens: bool,
    ) -> tuple[
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
    ]:
        return (
            collect_classifier_outputs(
                trained.classifier,
                self.store,
                labeled_indices,
                self.config,
                self.device,
                return_tokens=return_tokens,
            ),
            collect_classifier_outputs(
                trained.classifier,
                self.store,
                candidates,
                self.config,
                self.device,
                return_tokens=return_tokens,
            ),
            collect_classifier_outputs(
                trained.classifier,
                self.store,
                validation_indices,
                self.config,
                self.device,
                return_tokens=return_tokens,
            ),
        )

    def run_full_data(self) -> dict[str, Any]:
        """Train once on every row in the configured train split."""
        train_indices = self.store.indices("train")
        validation_indices = self.store.indices("val")
        test_indices = self.store.indices("test")
        start = time.perf_counter()
        trained = train_multimodal_round(
            self.store,
            train_indices.tolist(),
            self.config,
            self.device,
            validation_indices=validation_indices,
        )
        validation = collect_classifier_outputs(
            trained.classifier,
            self.store,
            validation_indices,
            self.config,
            self.device,
            return_tokens=False,
        )
        test = collect_classifier_outputs(
            trained.classifier,
            self.store,
            test_indices,
            self.config,
            self.device,
            return_tokens=False,
        )
        validation_metrics = task_multilabel_metrics(
            self.config, _host(validation["labels"]), _host(validation["probabilities"])
        )
        test_metrics = task_multilabel_metrics(
            self.config, _host(test["labels"]), _host(test["probabilities"])
        )
        output_dir = self.output_dir / "full_data"
        output_dir.mkdir(parents=True, exist_ok=True)
        data_usage = {
            "task_id": self.task.task_id,
            "cohort": str(self.config.get("preprocessing", {}).get("cohort", "")),
            "total_multimodal_visits": self.store.audit.total_samples,
            "train_pool": int(train_indices.size),
            "validation": int(validation_indices.size),
            "test": int(test_indices.size),
            "labeled_train_fraction": 1.0,
        }
        np.savez(
            output_dir / "final_predictions.npz",
            validation_labels=_host(validation["labels"]),
            validation_probabilities=_host(validation["probabilities"]),
            test_labels=_host(test["labels"]),
            test_probabilities=_host(test["probabilities"]),
        )
        final = {
            "validation": validation_metrics,
            "test": test_metrics,
            "data_usage": data_usage,
            "strategy": "full_data_upper_bound",
        }
        _write_json(
            output_dir / "full_data_state.json",
            {
                "format_version": 1,
                "protocol": task_manifest(self.config),
                "training_summary": trained.training_summary,
                "timing": trained.timings | {"total_sec": time.perf_counter() - start},
                "data_usage": data_usage,
                "metrics": final,
            },
        )
        _write_json(output_dir / "final_metrics.json", final)
        _write_json(output_dir / "resolved_config.json", self.config)
        return {"output_dir": str(output_dir), "final_metrics": final}

    def run(self) -> dict[str, Any]:
        random.seed(self.seed)
        np.random.seed(self.seed)
        train_indices = self.store.indices("train")
        validation_indices = self.store.indices("val")
        test_indices = self.store.indices("test")
        rounds = int(self.active["rounds"])
        schedule = labeled_schedule(
            int(train_indices.size),
            rounds=rounds,
            initial_fraction=float(self.active["initial_fraction"]),
            query_fraction=float(self.active["query_fraction"]),
        )
        labeled = _initial_indices(train_indices, schedule[0], self.seed)
        initial = np.asarray(labeled, dtype=np.int64)
        labeled_mask = np.zeros(self.store.audit.total_samples, dtype=bool)
        labeled_mask[labeled] = True
        strategy = str(self.active["strategy"]).lower()
        plugin = get_method(strategy)
        round_records: list[dict[str, Any]] = []
        total_start = time.perf_counter()
        final_validation: dict[str, torch.Tensor] | None = None
        final_test: dict[str, torch.Tensor] | None = None

        for round_index, target_count in enumerate(schedule):
            if len(labeled) != target_count:
                raise RuntimeError(f"round {round_index} expected {target_count} labels, got {len(labeled)}")
            round_start = time.perf_counter()
            trained = train_multimodal_round(
                self.store,
                labeled,
                self.config,
                self.device,
                validation_indices=validation_indices,
            )
            validation = collect_classifier_outputs(
                trained.classifier,
                self.store,
                validation_indices,
                self.config,
                self.device,
                return_tokens=False,
            )
            test = collect_classifier_outputs(
                trained.classifier,
                self.store,
                test_indices,
                self.config,
                self.device,
                return_tokens=False,
            )
            final_validation, final_test = validation, test
            validation_metrics = task_multilabel_metrics(
                self.config, _host(validation["labels"]), _host(validation["probabilities"])
            )
            test_metrics = task_multilabel_metrics(
                self.config, _host(test["labels"]), _host(test["probabilities"])
            )
            queries: list[int] = []
            acquisition: dict[str, Any] = {"method": "none (final training round)"}
            if round_index + 1 < rounds:
                candidates = train_indices[~labeled_mask[train_indices]]
                query_size = schedule[round_index + 1] - len(labeled)
                labeled_outputs, candidate_outputs, validation_outputs = self._outputs_for_method(
                    trained,
                    np.asarray(labeled, dtype=np.int64),
                    candidates,
                    validation_indices,
                    return_tokens=("modality_tokens" in tuple(getattr(plugin, "required_capabilities", ()))),
                )
                context = _method_context(
                    experiment=self,
                    trained=trained,
                    labeled_indices=np.asarray(labeled, dtype=np.int64),
                    candidates=candidates,
                    query_size=query_size,
                    round_index=round_index,
                    initial_indices=initial,
                    candidate_outputs=candidate_outputs,
                    labeled_outputs=labeled_outputs,
                    validation_outputs=validation_outputs,
                )
                method_fit_start = time.perf_counter()
                trained.method_state = fit_method(plugin, context)
                method_fit_elapsed = time.perf_counter() - method_fit_start
                if trained.method_state is not None:
                    trained.timings["method_auxiliary_training_sec"] = method_fit_elapsed
                    history = getattr(trained.method_state, "history", None)
                    if history is not None:
                        trained.history["method_auxiliary_loss"] = list(history)
                prepared_context = prepare_method_context(plugin, context, trained.method_state)
                acquisition = _acquire_method(plugin, prepared_context)
                queries = acquisition.pop("queries")
            self._save_checkpoint(trained, round_index, target_count)
            round_records.append(
                {
                    "round_index": round_index,
                    "labeled_count": target_count,
                    "labeled_fraction_of_train": target_count / train_indices.size,
                    "query_count": len(queries),
                    "query_indices": queries,
                    "validation_metrics": validation_metrics,
                    "test_metrics": test_metrics,
                    "training_history": trained.history,
                    "training_plan": {
                        "target_model": str(self.config.get("model", {}).get("architecture")),
                        "initialization": "fresh from ClinicalBERT source checkpoint",
                        "classifier_max_epochs": int(self.training["epochs"]),
                        "optimizer_step_budget": int(self.training["optimizer_steps_per_round"]),
                        "optimizer": "AdamW",
                        "weight_decay": float(self.training["weight_decay"]),
                        "bert_layerwise_learning_rate": True,
                        "early_stopping": "validation_bce",
                    },
                    "training_summary": trained.training_summary,
                    "timing": trained.timings | {"round_total_sec": time.perf_counter() - round_start},
                    "acquisition": acquisition,
                }
            )
            if queries:
                labeled_mask[np.asarray(queries, dtype=np.int64)] = True
                labeled = sorted([*labeled, *queries])

        assert final_validation is not None and final_test is not None
        data_usage = {
            "task_id": self.task.task_id,
            "cohort": str(self.config.get("preprocessing", {}).get("cohort", "")),
            "total_multimodal_visits": self.store.audit.total_samples,
            "train_pool": int(train_indices.size),
            "validation_model_selection": int(validation_indices.size),
            "test_evaluation_only": int(test_indices.size),
            "initial_labeled": len(initial),
            "newly_queried": len(labeled) - len(initial),
            "final_labeled": len(labeled),
            "final_fraction_of_train": len(labeled) / train_indices.size,
            "cumulative_fraction_targets": [count / float(train_indices.size) for count in schedule],
        }
        state = {
            "format_version": 5,
            "protocol": task_manifest(self.config),
            "strategy": strategy,
            "seed": self.seed,
            "labeled_schedule": schedule,
            "initial_indices": initial.tolist(),
            "final_labeled_indices": labeled,
            "records": round_records,
            "data_usage": data_usage,
            "total_wall_sec": time.perf_counter() - total_start,
        }
        np.savez(
            self.output_dir / "final_predictions.npz",
            validation_labels=_host(final_validation["labels"]),
            validation_probabilities=_host(final_validation["probabilities"]),
            test_labels=_host(final_test["labels"]),
            test_probabilities=_host(final_test["probabilities"]),
        )
        _write_json(self.output_dir / "active_state.json", state)
        final = {
            "validation": round_records[-1]["validation_metrics"],
            "test": round_records[-1]["test_metrics"],
            "data_usage": data_usage,
            "strategy": strategy,
        }
        _write_json(self.output_dir / "final_metrics.json", final)
        _write_json(self.output_dir / "resolved_config.json", self.config)
        return {"output_dir": str(self.output_dir), "final_metrics": final, "rounds": rounds}


__all__ = ["ActiveLearningExperiment", "labeled_schedule"]
