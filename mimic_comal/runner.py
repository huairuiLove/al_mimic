"""End-to-end active-learning runner for MIMIC-III CoMAL."""

from __future__ import annotations

import json
import random
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .data import load_records
from .diagnostics import acquisition_summary, build_round_diagnostics
from .features import load_features
from .integrity import assert_original_unchanged
from .model import (
    comal_acquisition_scores,
    paper_comal_acquisition_scores,
    positive_similarity_thresholds,
)
from .metrics import multilabel_metrics
from .training import (
    TrainedRound,
    _prototype_similarity_metrics,
    label_matrix,
    predict_tensors,
    train_round,
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _initial_indices(train_indices: np.ndarray, labels: np.ndarray, size: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    selected: set[int] = set()
    train_labels = labels[train_indices]
    # Match the original well-init option by seeding every supported label.
    for label in range(labels.shape[1]):
        hits = train_indices[train_labels[:, label] >= 0.5]
        if hits.size:
            selected.add(int(hits[rng.randrange(int(hits.size))]))
    remaining = [int(index) for index in train_indices if int(index) not in selected]
    rng.shuffle(remaining)
    selected.update(remaining[: max(0, size - len(selected))])
    if len(selected) > size:
        # Greedily retain broad label coverage when the budget is below label count.
        candidates = sorted(selected, key=lambda index: (-labels[index].sum(), index))
        selected = set(candidates[:size])
    return sorted(selected)


class ActiveLearningExperiment:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.dataset_cfg = config.get("dataset", {})
        self.training_cfg = config.get("training", {})
        self.active_cfg = config.get("active_learning", {})
        self.prepared_dir = Path(self.dataset_cfg.get("prepared_dir", "prepared/mimic_iii"))
        self.feature_dir = Path(self.dataset_cfg.get("feature_dir", self.prepared_dir / "features"))
        self.records = load_records(self.prepared_dir)
        labels_payload = json.loads((self.prepared_dir / "labels.json").read_text(encoding="utf-8"))
        self.label_names = tuple(labels_payload["labels"])
        self.labels = label_matrix(self.records, self.label_names)
        self.features = load_features(
            self.feature_dir,
            len(self.records),
            str(config.get("features", {}).get("encoder", "tfidf")),
        )
        requested_device = str(
            self.training_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            requested_device = "cpu"
        self.device = torch.device(requested_device)
        experiment = config.get("experiment", {})
        self.output_dir = Path(experiment.get("output_root", "experiments")) / str(
            experiment.get("name", "mimic_comal")
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = int(self.training_cfg.get("seed", 17))

    def _save_checkpoint(self, trained: TrainedRound, round_index: int) -> None:
        checkpoint_dir = self.output_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        torch.save(
            {
                "round": round_index,
                "classifier": {
                    key: value.detach().cpu() for key, value in trained.classifier.state_dict().items()
                },
                "comal": {key: value.detach().cpu() for key, value in trained.comal.state_dict().items()},
                "label_names": self.label_names,
                "feature_dim": int(self.features.shape[1]),
            },
            checkpoint_dir / "final.pt",
        )

    def run(self) -> dict[str, Any]:
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
            # Avoid allocator sync storms across many short AL rounds.
            torch.cuda.empty_cache()
        train_indices = np.asarray(
            [index for index, record in enumerate(self.records) if record.split == "train"], dtype=np.int64
        )
        validation_indices = np.asarray(
            [index for index, record in enumerate(self.records) if record.split == "validation"],
            dtype=np.int64,
        )
        test_indices = np.asarray(
            [index for index, record in enumerate(self.records) if record.split == "test"], dtype=np.int64
        )
        if min(train_indices.size, validation_indices.size, test_indices.size) == 0:
            raise RuntimeError("train, validation, and test splits must all be non-empty")
        initial_size = min(int(self.active_cfg.get("initial_labeled", 1000)), int(train_indices.size))
        labeled = _initial_indices(train_indices, self.labels, initial_size, self.seed)
        initial = list(labeled)
        records: list[dict[str, Any]] = []
        previous: TrainedRound | None = None
        total_start = time.perf_counter()
        rounds = max(1, int(self.active_cfg.get("rounds", 5)))
        query_size = int(self.active_cfg.get("query_size", 1000))
        strategy = str(self.active_cfg.get("strategy", "comal")).lower()
        labeled_mask = np.zeros(len(self.records), dtype=bool)
        labeled_mask[np.asarray(labeled, dtype=np.int64)] = True
        for round_index in range(rounds):
            round_start = time.perf_counter()
            round_config = deepcopy(self.config)
            round_training = round_config.setdefault("training", {})
            inherit = bool(self.training_cfg.get("inherit_across_rounds", False))
            if inherit and round_index > 0:
                round_training["epochs"] = int(
                    self.training_cfg.get("incremental_epochs", round_training.get("epochs", 20))
                )
                round_training["comal_epochs"] = int(
                    self.training_cfg.get("incremental_comal_epochs", round_training.get("comal_epochs", 10))
                )
            trained = train_round(
                self.features,
                self.labels,
                labeled,
                round_config,
                self.device,
                previous=previous,
            )
            # One fused forward over val+test cuts a full resident scan per round.
            eval_indices = np.concatenate([validation_indices, test_indices])
            eval_tensors = predict_tensors(
                trained, self.features, self.labels, eval_indices, self.config, self.device
            )
            eval_np = {name: value.detach().cpu().numpy() for name, value in eval_tensors.items()}
            split = int(validation_indices.size)
            validation_prediction = {name: value[:split] for name, value in eval_np.items()}
            test_prediction = {name: value[split:] for name, value in eval_np.items()}
            threshold = float(self.training_cfg.get("threshold", 0.5))
            validation_metrics = multilabel_metrics(
                validation_prediction["labels"], validation_prediction["probabilities"], threshold
            )
            test_metrics = multilabel_metrics(
                test_prediction["labels"], test_prediction["probabilities"], threshold
            )
            if "prototype_similarities" in validation_prediction:
                validation_metrics.update(
                    _prototype_similarity_metrics(
                        validation_prediction["labels"], validation_prediction["prototype_similarities"]
                    )
                )
            if "prototype_similarities" in test_prediction:
                test_metrics.update(
                    _prototype_similarity_metrics(
                        test_prediction["labels"], test_prediction["prototype_similarities"]
                    )
                )
            unlabeled = train_indices[~labeled_mask[train_indices]]
            queries: list[int] = []
            acquisition: dict[str, Any] = {}
            diagnostics_prediction = validation_prediction
            if unlabeled.size and round_index < rounds - 1:
                pool_size = min(
                    int(self.active_cfg.get("candidate_size", unlabeled.size)), int(unlabeled.size)
                )
                rng = np.random.default_rng(self.seed + round_index)
                candidates = (
                    np.sort(rng.choice(unlabeled, size=pool_size, replace=False))
                    if pool_size < unlabeled.size
                    else unlabeled
                )
                score_tensor: torch.Tensor | None = None
                if strategy == "random":
                    pool_tensors = predict_tensors(
                        trained, self.features, self.labels, candidates, self.config, self.device
                    )
                    scores = rng.random(len(candidates))
                    components = {"combined": scores}
                    pool = {name: value.detach().cpu().numpy() for name, value in pool_tensors.items()}
                elif strategy == "comal":
                    expected_cardinality = float(self.labels[np.asarray(labeled)].sum(axis=1).mean())
                    cfg = self.config.get("acquisition", {})
                    formula = str(cfg.get("formula", "paper")).lower()
                    if formula == "paper":
                        # Fuse labeled+candidate encode into one resident scan, then split.
                        labeled_array = np.asarray(labeled, dtype=np.int64)
                        fused_indices = np.concatenate([labeled_array, candidates])
                        fused = predict_tensors(
                            trained,
                            self.features,
                            self.labels,
                            fused_indices,
                            self.config,
                            self.device,
                        )
                        labeled_count = int(labeled_array.size)
                        labeled_tensors = {name: value[:labeled_count] for name, value in fused.items()}
                        pool_tensors = {name: value[labeled_count:] for name, value in fused.items()}
                        thresholds = positive_similarity_thresholds(
                            labeled_tensors["latents"],
                            labeled_tensors["labels"],
                            trained.comal.prototypes.detach(),
                        )
                        parts = paper_comal_acquisition_scores(
                            pool_tensors["probabilities"],
                            pool_tensors["latents"],
                            trained.comal.prototypes.detach(),
                            thresholds,
                            expected_cardinality=expected_cardinality,
                        )
                        component_names = (
                            "inverse_positive_evidence",
                            "cardinality_mismatch",
                            "prototype_positive_count",
                            "combined",
                        )
                    elif formula == "weighted":
                        pool_tensors = predict_tensors(
                            trained, self.features, self.labels, candidates, self.config, self.device
                        )
                        parts = comal_acquisition_scores(
                            pool_tensors["probabilities"],
                            pool_tensors["latents"],
                            trained.comal.prototypes.detach(),
                            expected_cardinality=expected_cardinality,
                            uncertainty_weight=float(cfg.get("uncertainty_weight", 0.5)),
                            prototype_weight=float(cfg.get("prototype_weight", 0.35)),
                            cardinality_weight=float(cfg.get("cardinality_weight", 0.15)),
                        )
                        component_names = (
                            "uncertainty",
                            "prototype_novelty",
                            "cardinality_mismatch",
                            "combined",
                        )
                    else:
                        raise ValueError("acquisition.formula must be paper or weighted")
                    score_tensor = parts.combined.detach()
                    components = {
                        name: getattr(parts, name).detach().cpu().numpy() for name in component_names
                    }
                    scores = components["combined"]
                    pool = {name: value.detach().cpu().numpy() for name, value in pool_tensors.items()}
                else:
                    raise ValueError("active_learning.strategy must be comal or random")
                count = min(query_size, len(candidates))
                if score_tensor is not None and count < int(score_tensor.numel()):
                    # Device top-k then stable reorder of the shortlist only.
                    top_values, top_positions = torch.topk(score_tensor, k=count, largest=True, sorted=False)
                    order = torch.argsort(top_values, descending=True, stable=True)
                    positions = top_positions[order].detach().cpu().numpy()
                else:
                    score_array = np.asarray(scores)
                    if count < score_array.size:
                        part = np.argpartition(-score_array, count - 1)[:count]
                        positions = part[np.argsort(-score_array[part], kind="stable")]
                    else:
                        positions = np.argsort(-score_array, kind="stable")
                queries = [int(value) for value in pool["indices"][positions]]
                acquisition = {
                    "candidate_count": int(len(candidates)),
                    "query_count": count,
                    "components": acquisition_summary(components, positions),
                    "combined_distribution": _summary(np.asarray(scores)),
                }
                diagnostics_prediction = pool
                diagnostic_scores = np.asarray(scores)
            else:
                diagnostic_scores = None
            diagnostics = build_round_diagnostics(
                diagnostics_prediction["labels"],
                diagnostics_prediction["probabilities"],
                diagnostics_prediction["latents"],
                trained.comal.prototypes,
                self.label_names,
                acquisition_scores=diagnostic_scores,
                prototype_similarities=diagnostics_prediction.get("prototype_similarities"),
            )
            diagnostics["oracle_audit"] = {
                "uses_withheld_labels": True,
                "used_for_selection": False,
                "purpose": "retrospective score reliability and rare-label diagnostics only",
            }
            _write_json(self.output_dir / "diagnostics" / f"round_{round_index:03d}.json", diagnostics)
            record = {
                "round_index": round_index,
                "labeled_before_query": len(labeled),
                "query_indices": queries,
                "validation_metrics": validation_metrics,
                "test_metrics": test_metrics,
                "training_history": trained.history,
                "training_plan": {
                    "initialization": ("inherit_previous" if inherit and round_index > 0 else "cold_start"),
                    "classifier_epochs": int(round_training.get("epochs", 20)),
                    "comal_epochs": int(round_training.get("comal_epochs", 10)),
                    "optimizer_inherited": False,
                },
                "timing": trained.timings | {"round_total_sec": time.perf_counter() - round_start},
                "acquisition": acquisition,
            }
            records.append(record)
            if queries:
                labeled_mask[np.asarray(queries, dtype=np.int64)] = True
            labeled.extend(queries)
            labeled = sorted(set(labeled))
            previous = trained if bool(self.training_cfg.get("inherit_across_rounds", False)) else None
        self._save_checkpoint(trained, rounds - 1)
        np.savez_compressed(
            self.output_dir / "final_predictions.npz",
            validation_labels=validation_prediction["labels"],
            validation_probabilities=validation_prediction["probabilities"],
            validation_prototype_similarities=validation_prediction.get("prototype_similarities"),
            test_labels=test_prediction["labels"],
            test_probabilities=test_prediction["probabilities"],
            test_prototype_similarities=test_prediction.get("prototype_similarities"),
        )
        state = {
            "format_version": 1,
            "strategy": strategy,
            "seed": self.seed,
            "initial_indices": initial,
            "final_labeled_indices": labeled,
            "label_names": self.label_names,
            "records": records,
            "total_wall_sec": time.perf_counter() - total_start,
        }
        _write_json(self.output_dir / "active_state.json", state)
        final = {
            "validation": records[-1]["validation_metrics"],
            "test": records[-1]["test_metrics"],
            "label_cost": {
                "initial": len(initial),
                "queries": len(labeled) - len(initial),
                "total": len(labeled),
            },
            "strategy": strategy,
        }
        _write_json(self.output_dir / "final_metrics.json", final)
        _write_json(self.output_dir / "resolved_config.json", self.config)
        _write_json(
            self.output_dir / "source_integrity.json",
            assert_original_unchanged(Path.cwd()),
        )
        return {"output_dir": str(self.output_dir), "final_metrics": final, "rounds": rounds}
