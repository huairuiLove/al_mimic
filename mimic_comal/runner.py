"""Six-round active learning on the Yang-Wu MIMIC-III diagnosis baseline."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .integrity import assert_original_unchanged
from .metrics import multilabel_metrics
from .model import (
    estimate_mm_comal_statistics,
    mm_comal_acquisition_scores,
    paper_comal_acquisition_scores,
    positive_similarity_thresholds,
)
from .multimodal_data import YangWuFeatureStore
from .multimodal_training import (
    TrainedMultimodalRound,
    attach_comal_outputs,
    collect_classifier_outputs,
    train_multimodal_round,
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import orjson

        path.write_bytes(orjson.dumps(value, option=orjson.OPT_INDENT_2) + b"\n")
    except ImportError:  # pragma: no cover
        path.write_text(
            json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )


def labeled_schedule(
    train_size: int,
    *,
    rounds: int = 6,
    initial_fraction: float = 0.10,
    query_fraction: float = 0.05,
) -> list[int]:
    """Half-up rounded cumulative targets based on the actual official train split."""
    if train_size < 1 or rounds < 1:
        raise ValueError("train_size and rounds must be positive")
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


def _score_summary(
    components: dict[str, torch.Tensor], selected_positions: torch.Tensor
) -> dict[str, dict[str, float]]:
    return {
        name: {
            "pool_mean": float(values.float().mean()),
            "selected_mean": float(values.index_select(0, selected_positions).float().mean()),
        }
        for name, values in components.items()
    }


class ActiveLearningExperiment:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.training = config.get("training", {})
        self.active = config.get("active_learning", {})
        self.store = YangWuFeatureStore(config, validate=True)
        requested_device = str(self.training.get("device", "cuda"))
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("formal configuration requires CUDA, but no CUDA device is available")
        self.device = torch.device(requested_device)
        experiment = config.get("experiment", {})
        self.output_dir = Path(experiment.get("output_root", "experiments")) / str(
            experiment.get("name", "mimic_iii_yang_wu_comal")
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = int(self.training.get("seed", 17))

    def _save_checkpoint(
        self, trained: TrainedMultimodalRound, round_index: int, labeled_count: int
    ) -> None:
        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        payload: dict[str, Any] = {
            "round": round_index,
            "labeled_count": labeled_count,
            "classifier": {
                key: value.detach().cpu()
                for key, value in trained.classifier.state_dict().items()
            },
            "target_model": "Yang-Wu BertEncoder (EMNLP 2021)",
            "task": "MIMIC-III Diagnoses 48h, 1,042 ICD-9 groups",
            "initialization": "fresh from the same ClinicalBERT source checkpoint",
            "classifier_epochs": int(self.training["epochs"]),
        }
        if trained.comal is not None:
            payload["comal"] = {
                key: value.detach().cpu() for key, value in trained.comal.state_dict().items()
            }
        round_path = checkpoint_dir / f"round_{round_index:03d}.pt"
        torch.save(payload, round_path)
        if round_index == int(self.active["rounds"]) - 1:
            torch.save(payload, checkpoint_dir / "final.pt")
        if trained.modis_state is not None:
            torch.save(
                {
                    "round": round_index,
                    "probes": {
                        key: value.detach().cpu()
                        for key, value in trained.modis_state.probes.state_dict().items()
                    },
                    "prototypes": trained.modis_state.prototypes.detach().cpu(),
                    "diagnostics": trained.modis_state.diagnostics,
                },
                checkpoint_dir / f"round_{round_index:03d}_modis.pt",
            )

    def _acquire_random(
        self, candidates: np.ndarray, query_size: int, round_index: int
    ) -> tuple[list[int], dict[str, Any]]:
        """Uniform sampling control. Shares the schedule, seed and training path with
        the informed strategies, so any gap is attributable to acquisition alone."""
        rng = np.random.default_rng(self.seed + round_index)
        chosen = rng.choice(candidates, size=query_size, replace=False)
        queries = sorted(int(value) for value in chosen)
        return queries, {
            "method": "uniform random sampling control",
            "candidate_count": int(candidates.size),
            "selected_count": len(queries),
            "score_components": {},
        }

    def _acquire_comal(
        self,
        trained: TrainedMultimodalRound,
        candidates: np.ndarray,
        query_size: int,
    ) -> tuple[list[int], dict[str, Any]]:
        if trained.comal is None or trained.labeled_outputs is None:
            raise RuntimeError("CoMAL state is missing")
        candidate_outputs = collect_classifier_outputs(
            trained.classifier,
            self.store,
            candidates,
            self.config,
            self.device,
            return_tokens=False,
        )
        candidate_outputs = attach_comal_outputs(
            trained.comal,
            candidate_outputs,
            batch_size=int(self.training["eval_batch_size"]),
        )
        labeled_labels = trained.labeled_outputs["labels"]
        if trained.labeled_own_similarity is None:
            raise RuntimeError("labeled CoMAL similarities are missing")
        thresholds = positive_similarity_thresholds(
            None,
            labeled_labels,
            trained.comal.prototypes,
            own_similarity=trained.labeled_own_similarity,
        )
        parts = paper_comal_acquisition_scores(
            candidate_outputs["probabilities"],
            None,
            trained.comal.prototypes,
            thresholds,
            expected_cardinality=labeled_labels.sum(dim=1).mean(),
            own_similarity=candidate_outputs["prototype_similarities"][..., 0],
        )
        order = torch.argsort(parts.combined, descending=True, stable=True)[:query_size]
        queries = candidates[_host(order).astype(np.int64)].astype(np.int64).tolist()
        components = {
            "inverse_positive_evidence": parts.inverse_positive_evidence,
            "cardinality_mismatch": parts.cardinality_mismatch,
            "prototype_positive_count": parts.prototype_positive_count,
            "combined": parts.combined,
        }
        return queries, {
            "method": "CoMAL paper score on Yang-Wu fused features",
            "candidate_count": int(candidates.size),
            "selected_count": len(queries),
            "score_components": _score_summary(components, order),
        }

    def _acquire_mm_comal(
        self,
        trained: TrainedMultimodalRound,
        candidates: np.ndarray,
        query_size: int,
    ) -> tuple[list[int], dict[str, Any]]:
        if (
            trained.comal is None
            or trained.labeled_outputs is None
            or trained.labeled_view_own_similarity is None
        ):
            raise RuntimeError("MM-CoMAL state is missing")
        candidate_outputs = collect_classifier_outputs(
            trained.classifier,
            self.store,
            candidates,
            self.config,
            self.device,
            return_tokens=True,
        )
        candidate_outputs = attach_comal_outputs(
            trained.comal,
            candidate_outputs,
            batch_size=int(self.training["eval_batch_size"]),
        )
        cfg = self.config.get("acquisition", {}).get("mm", {})
        statistics = estimate_mm_comal_statistics(
            trained.labeled_view_own_similarity,
            trained.labeled_outputs["labels"],
            reliability_shrinkage=float(cfg.get("reliability_shrinkage", 10.0)),
            threshold_shrinkage=float(cfg.get("threshold_shrinkage", 10.0)),
            threshold_estimator=str(cfg.get("threshold_estimator", "shrunk")),
            include_fused_in_weights=bool(cfg.get("include_fused_in_weights", False)),
            equal_weights=bool(cfg.get("equal_weights", False)),
        )
        parts = mm_comal_acquisition_scores(
            candidate_outputs["probabilities"],
            candidate_outputs["view_own_similarity"],
            statistics,
            expected_cardinality=trained.labeled_outputs["labels"].sum(dim=1).mean(),
            alpha=float(cfg.get("alpha", 1.0)),
            dispersion=str(cfg.get("dispersion", "weighted_mad")),
        )
        order = torch.argsort(parts.combined, descending=True, stable=True)[:query_size]
        queries = candidates[_host(order).astype(np.int64)].astype(np.int64).tolist()
        components = {
            "inverse_positive_evidence": parts.inverse_positive_evidence,
            "cardinality_mismatch": parts.cardinality_mismatch,
            "prototype_positive_count": parts.prototype_positive_count,
            "dispersion": parts.dispersion,
            "base_score": parts.base_score,
            "combined": parts.combined,
        }
        return queries, {
            "method": "MM-CoMAL four-view prototype evidence",
            "candidate_count": int(candidates.size),
            "selected_count": len(queries),
            "view_names": [*trained.classifier.modality_names, "fused"],
            "mean_reliability_by_view": [
                float(value) for value in statistics.reliability.mean(dim=1)
            ],
            "mean_weight_by_view": [float(value) for value in statistics.weights.mean(dim=1)],
            "score_components": _score_summary(components, order),
        }

    def _acquire_modis(
        self,
        trained: TrainedMultimodalRound,
        candidates: np.ndarray,
        query_size: int,
        initial: list[int],
    ) -> tuple[list[int], dict[str, Any]]:
        if trained.modis_state is None:
            raise RuntimeError("MoDIS probe state is missing")
        candidate_outputs = collect_classifier_outputs(
            trained.classifier,
            self.store,
            candidates,
            self.config,
            self.device,
            return_tokens=True,
        )
        from modis.acquire import acquire_modis

        result = acquire_modis(
            trained.classifier,
            trained.modis_state,
            candidate_outputs,
            query_size=query_size,
            config=self.config,
            initial_prevalence=torch.as_tensor(
                self.store.labels[np.asarray(initial)].mean(axis=0),
                dtype=torch.float32,
                device=self.device,
            ),
        )
        positions = result.selected_positions
        queries = candidates[_host(positions).astype(np.int64)].astype(np.int64).tolist()
        components = {
            "disagreement": result.disagreement,
            "instability": result.instability,
            "dominance": result.dominance,
            "sufficiency_penalty": result.sufficiency_penalty,
            "combined": result.combined,
        }
        return queries, {
            "method": "MoDIS modality disagreement and intervention instability",
            "candidate_count": int(candidates.size),
            "selected_count": len(queries),
            "score_components": _score_summary(components, positions),
            "method_diagnostics": result.diagnostics,
        }

    def _acquire_mosaic(
        self,
        trained: TrainedMultimodalRound,
        candidates: np.ndarray,
        query_size: int,
        round_index: int,
    ) -> tuple[list[int], dict[str, Any]]:
        if trained.labeled_outputs is None:
            raise RuntimeError("MoSAIC labeled outputs are missing")
        candidate_outputs = collect_classifier_outputs(
            trained.classifier,
            self.store,
            candidates,
            self.config,
            self.device,
            return_tokens=True,
        )
        # Fisher reference labels come only from the currently labeled train set.
        reference_outputs = trained.labeled_outputs
        from mosaic.acquire import acquire_mosaic

        result = acquire_mosaic(
            trained.classifier,
            trained.labeled_outputs,
            reference_outputs,
            candidate_outputs,
            query_size=query_size,
            config=self.config,
            seed=self.seed + round_index,
        )
        positions = result.selected_positions
        queries = candidates[_host(positions).astype(np.int64)].astype(np.int64).tolist()
        components = {
            "additive": result.additive,
            "synergy": result.synergy,
            "total_gain": result.total_gain,
            "combined": result.combined,
        }
        return queries, {
            "method": "MoSAIC Fisher design with modality-lattice synergy",
            "candidate_count": int(candidates.size),
            "selected_count": len(queries),
            "score_components": _score_summary(components, positions),
            "method_diagnostics": result.diagnostics,
        }

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
        initial = list(labeled)
        labeled_mask = np.zeros(self.store.audit.total_samples, dtype=bool)
        labeled_mask[labeled] = True
        strategy = str(self.active["strategy"]).lower()
        round_records: list[dict[str, Any]] = []
        total_start = time.perf_counter()
        final_validation: dict[str, torch.Tensor] | None = None
        final_test: dict[str, torch.Tensor] | None = None

        for round_index, target_count in enumerate(schedule):
            if len(labeled) != target_count:
                raise RuntimeError(
                    f"round {round_index} expected {target_count} labels, got {len(labeled)}"
                )
            round_start = time.perf_counter()
            trained = train_multimodal_round(
                self.store, labeled, self.config, self.device
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
            validation_metrics = multilabel_metrics(
                _host(validation["labels"]), _host(validation["probabilities"])
            )
            test_metrics = multilabel_metrics(
                _host(test["labels"]), _host(test["probabilities"])
            )
            queries: list[int] = []
            acquisition: dict[str, Any] = {"method": "none (final training round)"}
            if round_index + 1 < rounds:
                candidates = train_indices[~labeled_mask[train_indices]]
                query_size = schedule[round_index + 1] - len(labeled)
                if strategy == "random":
                    queries, acquisition = self._acquire_random(
                        candidates, query_size, round_index
                    )
                elif strategy == "comal":
                    queries, acquisition = self._acquire_comal(
                        trained, candidates, query_size
                    )
                elif strategy == "mm_comal":
                    queries, acquisition = self._acquire_mm_comal(
                        trained, candidates, query_size
                    )
                elif strategy == "modis":
                    queries, acquisition = self._acquire_modis(
                        trained, candidates, query_size, initial
                    )
                elif strategy == "mosaic":
                    queries, acquisition = self._acquire_mosaic(
                        trained,
                        candidates,
                        query_size,
                        round_index,
                    )
                else:  # protected by config validation
                    raise AssertionError(strategy)
                if len(queries) != query_size or len(set(queries)) != query_size:
                    raise RuntimeError("acquisition did not return the exact unique query budget")
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
                        "target_model": "Yang-Wu BertEncoder",
                        "initialization": "fresh from ClinicalBERT source checkpoint",
                        "classifier_epochs": int(self.training["epochs"]),
                        "comal_epochs": (
                            int(self.training["comal_epochs"])
                            if strategy in {"comal", "mm_comal"}
                            else 0
                        ),
                    },
                    "timing": trained.timings
                    | {"round_total_sec": time.perf_counter() - round_start},
                    "acquisition": acquisition,
                }
            )
            if queries:
                labeled_mask[np.asarray(queries, dtype=np.int64)] = True
                labeled = sorted([*labeled, *queries])

        assert final_validation is not None and final_test is not None
        data_usage = {
            "paper_total_multimodal_visits": self.store.audit.total_samples,
            "official_train_pool": int(train_indices.size),
            "official_validation_evaluation_only": int(validation_indices.size),
            "official_test_evaluation_only": int(test_indices.size),
            "initial_labeled": len(initial),
            "newly_queried": len(labeled) - len(initial),
            "final_labeled": len(labeled),
            "final_fraction_of_train": len(labeled) / train_indices.size,
            "cumulative_fraction_targets": [0.10, 0.15, 0.20, 0.25, 0.30, 0.35],
        }
        state = {
            "format_version": 3,
            "protocol": "Yang and Wu 2021 BertEncoder, MIMIC-III Diagnoses 48h",
            "strategy": strategy,
            "seed": self.seed,
            "labeled_schedule": schedule,
            "initial_indices": initial,
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
        _write_json(
            self.output_dir / "source_integrity.json",
            assert_original_unchanged(Path.cwd()),
        )
        return {"output_dir": str(self.output_dir), "final_metrics": final, "rounds": rounds}
