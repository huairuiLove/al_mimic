"""Six cold-start active-learning rounds on full BRSET v1.0.2."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from mimic_comal.model import (
    estimate_mm_comal_statistics,
    mm_comal_acquisition_scores,
    paper_comal_acquisition_scores,
    positive_similarity_thresholds,
)
from mimic_comal.multimodal_training import attach_comal_outputs
from mimic_comal.runner import labeled_schedule

from .config import data_paths, resolve_path
from .data import BrsetFeatureStore, LABEL_COLUMNS
from .metrics import fit_f1_thresholds, multilabel_metrics
from .training import (
    TrainedBrsetRound,
    aggregate_patient_outputs,
    collect_outputs,
    train_round,
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


def _host(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().float().cpu().numpy()


def _initial_patients(train_patients: Sequence[str], size: int, seed: int) -> list[str]:
    if size < 1 or size > len(train_patients):
        raise ValueError("initial BRSET patient budget is outside the train pool")
    rng = np.random.default_rng(seed)
    selected = rng.choice(np.asarray(train_patients, dtype=object), size=size, replace=False)
    return sorted(str(value) for value in selected)


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


class BrsetActiveLearningExperiment:
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
            "target_model": "ImageNet ResNet-50 + BRSET clinical metadata MLP",
            "task": "BRSET v1.0.2, 13 retinal disease labels",
            "initialization": "fresh ImageNet ResNet-50 and fresh fusion layers",
            "classifier_epochs": int(self.training["epochs"]),
        }
        if trained.comal is not None:
            payload["comal"] = {
                key: value.detach().cpu() for key, value in trained.comal.state_dict().items()
            }
        torch.save(payload, checkpoint_dir / f"round_{round_index:03d}.pt")
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

    def _patient_candidates(
        self,
        classifier,
        candidate_patients: Sequence[str],
    ) -> dict[str, torch.Tensor]:
        image_indices = self.store.indices_for_patients(candidate_patients)
        image_outputs = collect_outputs(
            classifier,
            self.store,
            image_indices,
            self.config,
            self.device,
            return_tokens=True,
        )
        grouped, ordered_patients = aggregate_patient_outputs(classifier, self.store, image_outputs)
        if ordered_patients != sorted(candidate_patients):
            raise RuntimeError("candidate patient aggregation changed patient ordering")
        return grouped

    def _acquire_comal(
        self,
        trained: TrainedBrsetRound,
        candidates: Sequence[str],
        query_size: int,
    ) -> tuple[list[str], dict[str, Any]]:
        if trained.comal is None or trained.labeled_outputs is None:
            raise RuntimeError("CoMAL state is missing")
        candidate_outputs = self._patient_candidates(trained.classifier, candidates)
        candidate_outputs = attach_comal_outputs(
            trained.comal,
            candidate_outputs,
            batch_size=int(self.training["eval_batch_size"]),
        )
        labels = trained.labeled_outputs["labels"]
        if trained.labeled_own_similarity is None:
            raise RuntimeError("labeled CoMAL similarities are missing")
        thresholds = positive_similarity_thresholds(
            None,
            labels,
            trained.comal.prototypes,
            own_similarity=trained.labeled_own_similarity,
        )
        parts = paper_comal_acquisition_scores(
            candidate_outputs["probabilities"],
            None,
            trained.comal.prototypes,
            thresholds,
            expected_cardinality=labels.sum(dim=1).mean(),
            own_similarity=candidate_outputs["prototype_similarities"][..., 0],
        )
        order = torch.argsort(parts.combined, descending=True, stable=True)[:query_size]
        queries = [sorted(candidates)[int(position)] for position in order.detach().cpu()]
        components = {
            "inverse_positive_evidence": parts.inverse_positive_evidence,
            "cardinality_mismatch": parts.cardinality_mismatch,
            "prototype_positive_count": parts.prototype_positive_count,
            "combined": parts.combined,
        }
        return queries, {
            "method": "CoMAL paper score on fused patient representations",
            "candidate_patients": len(candidates),
            "selected_patients": len(queries),
            "score_components": _score_summary(components, order),
        }

    def _acquire_mm_comal(
        self,
        trained: TrainedBrsetRound,
        candidates: Sequence[str],
        query_size: int,
    ) -> tuple[list[str], dict[str, Any]]:
        if (
            trained.comal is None
            or trained.labeled_outputs is None
            or trained.labeled_view_own_similarity is None
        ):
            raise RuntimeError("MM-CoMAL state is missing")
        candidate_outputs = self._patient_candidates(trained.classifier, candidates)
        candidate_outputs = attach_comal_outputs(
            trained.comal,
            candidate_outputs,
            batch_size=int(self.training["eval_batch_size"]),
        )
        cfg = self.config.get("acquisition", {}).get("mm", {})
        labels = trained.labeled_outputs["labels"]
        statistics = estimate_mm_comal_statistics(
            trained.labeled_view_own_similarity,
            labels,
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
            expected_cardinality=labels.sum(dim=1).mean(),
            alpha=float(cfg.get("alpha", 1.0)),
            dispersion=str(cfg.get("dispersion", "weighted_mad")),
        )
        order = torch.argsort(parts.combined, descending=True, stable=True)[:query_size]
        queries = [sorted(candidates)[int(position)] for position in order.detach().cpu()]
        components = {
            "inverse_positive_evidence": parts.inverse_positive_evidence,
            "cardinality_mismatch": parts.cardinality_mismatch,
            "prototype_positive_count": parts.prototype_positive_count,
            "dispersion": parts.dispersion,
            "combined": parts.combined,
        }
        return queries, {
            "method": "MM-CoMAL image/metadata/fusion prototype evidence by patient",
            "candidate_patients": len(candidates),
            "selected_patients": len(queries),
            "view_names": [*trained.classifier.modality_names, "fused"],
            "score_components": _score_summary(components, order),
        }

    def _acquire_modis(
        self,
        trained: TrainedBrsetRound,
        candidates: Sequence[str],
        query_size: int,
        initial: Sequence[str],
    ) -> tuple[list[str], dict[str, Any]]:
        if trained.modis_state is None:
            raise RuntimeError("MoDIS probe state is missing")
        candidate_outputs = self._patient_candidates(trained.classifier, candidates)
        from modis.acquire import acquire_modis

        result = acquire_modis(
            trained.classifier,
            trained.modis_state,
            candidate_outputs,
            query_size=query_size,
            config=self.config,
            initial_prevalence=torch.as_tensor(
                self.store.patient_targets(initial).mean(axis=0),
                dtype=torch.float32,
                device=self.device,
            ),
        )
        positions = result.selected_positions
        queries = [sorted(candidates)[int(position)] for position in positions.detach().cpu()]
        components = {
            "disagreement": result.disagreement,
            "instability": result.instability,
            "dominance": result.dominance,
            "sufficiency_penalty": result.sufficiency_penalty,
            "combined": result.combined,
        }
        return queries, {
            "method": "MoDIS image/metadata disagreement by patient",
            "candidate_patients": len(candidates),
            "selected_patients": len(queries),
            "score_components": _score_summary(components, positions),
            "method_diagnostics": result.diagnostics,
        }

    def _acquire_mosaic(
        self,
        trained: TrainedBrsetRound,
        candidates: Sequence[str],
        query_size: int,
        round_index: int,
    ) -> tuple[list[str], dict[str, Any]]:
        if trained.labeled_outputs is None:
            raise RuntimeError("MoSAIC labeled outputs are missing")
        candidate_outputs = self._patient_candidates(trained.classifier, candidates)
        from mosaic.acquire import acquire_mosaic

        result = acquire_mosaic(
            trained.classifier,
            trained.labeled_outputs,
            trained.labeled_outputs,
            candidate_outputs,
            query_size=query_size,
            config=self.config,
            seed=self.seed + round_index,
        )
        positions = result.selected_positions
        queries = [sorted(candidates)[int(position)] for position in positions.detach().cpu()]
        components = {
            "additive": result.additive,
            "synergy": result.synergy,
            "total_gain": result.total_gain,
            "combined": result.combined,
        }
        return queries, {
            "method": "MoSAIC image/metadata Fisher synergy by patient",
            "candidate_patients": len(candidates),
            "selected_patients": len(queries),
            "score_components": _score_summary(components, positions),
            "method_diagnostics": result.diagnostics,
        }

    def run(self) -> dict[str, Any]:
        random.seed(self.seed)
        np.random.seed(self.seed)
        train_patients = self.store.patient_ids("train")
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
                _host(validation["labels"]),
                _host(validation["probabilities"]),
                thresholds,
            )
            test_metrics = multilabel_metrics(_host(test["labels"]), _host(test["probabilities"]), thresholds)
            final_validation, final_test, final_thresholds = validation, test, thresholds
            queries: list[str] = []
            acquisition: dict[str, Any] = {"method": "none (final training round)"}
            if round_index + 1 < rounds:
                labeled_set = set(labeled)
                candidates = [patient for patient in train_patients if patient not in labeled_set]
                query_size = schedule[round_index + 1] - len(labeled)
                if strategy == "comal":
                    queries, acquisition = self._acquire_comal(trained, candidates, query_size)
                elif strategy == "mm_comal":
                    queries, acquisition = self._acquire_mm_comal(trained, candidates, query_size)
                elif strategy == "modis":
                    queries, acquisition = self._acquire_modis(trained, candidates, query_size, initial)
                elif strategy == "mosaic":
                    queries, acquisition = self._acquire_mosaic(trained, candidates, query_size, round_index)
                else:  # protected by configuration validation
                    raise AssertionError(strategy)
                if len(queries) != query_size or len(set(queries)) != query_size:
                    raise RuntimeError("patient acquisition did not return the exact unique budget")

            labeled_indices = self.store.indices_for_patients(labeled)
            query_indices = self.store.indices_for_patients(queries)
            self._save_checkpoint(trained, round_index, labeled)
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
                    "training_plan": {
                        "target_model": "ImageNet ResNet-50 + metadata MLP",
                        "initialization": "fresh pretrained ResNet-50 and fresh fusion layers",
                        "classifier_epochs": int(self.training["epochs"]),
                        "comal_epochs": (
                            int(self.training["comal_epochs"]) if strategy in {"comal", "mm_comal"} else 0
                        ),
                    },
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
            "validation_patients": len(self.store.patient_ids("val")),
            "test_images": int(test_indices.size),
            "test_patients": len(self.store.patient_ids("test")),
            "initial_labeled_patients": len(initial),
            "initial_labeled_images": int(initial_indices.size),
            "newly_queried_patients": len(labeled) - len(initial),
            "final_labeled_patients": len(labeled),
            "final_labeled_images": int(final_indices.size),
            "final_fraction_of_train_patients": len(labeled) / len(train_patients),
            "cumulative_fraction_targets": [0.10, 0.15, 0.20, 0.25, 0.30, 0.35],
        }
        state = {
            "format_version": 1,
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
        }
        _write_json(self.output_dir / "final_metrics.json", final)
        _write_json(self.output_dir / "resolved_config.json", self.config)
        _write_json(
            self.output_dir / "source_audit.json",
            json.loads(data_paths(self.config, require_prepared=True)["audit"].read_text()),
        )
        return {"output_dir": str(self.output_dir), "final_metrics": final, "rounds": rounds}
