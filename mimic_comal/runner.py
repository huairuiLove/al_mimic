"""End-to-end active-learning runner for MIMIC-III CoMAL."""

from __future__ import annotations

import json
import random
import time
from copy import copy
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
    label_matrix,
    predict_tensors,
    prototype_similarity_metrics_torch,
    train_round,
    warm_resident_matrices,
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import orjson

        path.write_bytes(orjson.dumps(value, option=orjson.OPT_INDENT_2) + b"\n")
    except ImportError:  # pragma: no cover
        path.write_text(
            json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
        )


def _async_to_host(
    tensor: torch.Tensor, copy_stream: torch.cuda.Stream | None
) -> torch.Tensor:
    """D2H into pinned memory; optional side stream enables compute overlap."""
    if tensor.device.type != "cuda":
        return tensor.detach().cpu()
    host = torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=True)
    if copy_stream is None:
        host.copy_(tensor.detach(), non_blocking=True)
        return host
    copy_stream.wait_stream(torch.cuda.current_stream(tensor.device))
    with torch.cuda.stream(copy_stream):
        host.copy_(tensor.detach(), non_blocking=True)
    return host


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
    if len(selected) < size:
        mask = np.ones(train_indices.size, dtype=bool)
        if selected:
            # Vectorized membership vs. Python set scans over the full train pool.
            selected_arr = np.fromiter(selected, dtype=np.int64, count=len(selected))
            mask &= ~np.isin(train_indices, selected_arr, assume_unique=False)
        remaining = train_indices[mask].tolist()
        rng.shuffle(remaining)
        selected.update(int(index) for index in remaining[: size - len(selected)])
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
        splits = np.fromiter((record.split for record in self.records), dtype=object, count=len(self.records))
        train_indices = np.flatnonzero(splits == "train").astype(np.int64, copy=False)
        validation_indices = np.flatnonzero(splits == "validation").astype(np.int64, copy=False)
        test_indices = np.flatnonzero(splits == "test").astype(np.int64, copy=False)
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
        # Fixed each round; concatenate once instead of rebuilding val+test indices.
        eval_indices = np.concatenate([validation_indices, test_indices])
        warm_resident_matrices(self.features, self.labels, self.device, self.training_cfg)
        for round_index in range(rounds):
            round_start = time.perf_counter()
            # Shallow-copy only the training dict when overriding incremental epochs.
            inherit = bool(self.training_cfg.get("inherit_across_rounds", False))
            round_training = self.training_cfg
            if inherit and round_index > 0:
                round_config = copy(self.config)
                round_training = dict(self.training_cfg)
                round_training["epochs"] = int(
                    self.training_cfg.get("incremental_epochs", round_training.get("epochs", 20))
                )
                round_training["comal_epochs"] = int(
                    self.training_cfg.get("incremental_comal_epochs", round_training.get("comal_epochs", 10))
                )
                round_config["training"] = round_training
            else:
                round_config = self.config
            trained = train_round(
                self.features,
                self.labels,
                labeled,
                round_config,
                self.device,
                previous=previous,
            )
            unlabeled = train_indices[~labeled_mask[train_indices]]
            will_query = bool(unlabeled.size and round_index < rounds - 1)
            cfg = self.config.get("acquisition", {})
            formula = str(cfg.get("formula", "paper")).lower()
            # Fuse eval with acquisition pools into one resident scan when possible.
            fuse_mode = "none"
            if will_query and strategy == "comal" and formula == "paper":
                fuse_mode = "paper"
            elif will_query and strategy == "comal" and formula == "weighted":
                fuse_mode = "weighted"  # val+test+candidates
            elif will_query and strategy == "random":
                fuse_mode = "random"  # val+test+candidates (no latents)
            candidates = np.empty(0, dtype=np.int64)
            rng = np.random.default_rng(self.seed + round_index)
            if will_query:
                pool_size = min(
                    int(self.active_cfg.get("candidate_size", unlabeled.size)), int(unlabeled.size)
                )
                candidates = (
                    np.sort(rng.choice(unlabeled, size=pool_size, replace=False))
                    if pool_size < unlabeled.size
                    else unlabeled
                )
            split = int(validation_indices.size)
            labeled_array = np.asarray(labeled, dtype=np.int64)
            labeled_tensors: dict[str, torch.Tensor] | None = None
            pool_tensors: dict[str, torch.Tensor] | None = None
            # Eval always keeps full [N,L,L+1] similarities for metrics / final npz.
            eval_tensors = predict_tensors(
                trained,
                self.features,
                self.labels,
                eval_indices,
                self.config,
                self.device,
                return_latents=False,
                similarity_mode="full",
            )
            # Kick off pinned D2H on a side stream before acquisition predict.
            host_keys = ["indices", "labels", "probabilities"]
            if not will_query:
                host_keys.append("prototype_similarities")
            copy_stream = torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None
            validation_host = {
                name: _async_to_host(eval_tensors[name][:split], copy_stream) for name in host_keys
            }
            test_host = {
                name: _async_to_host(eval_tensors[name][split:], copy_stream) for name in host_keys
            }
            if fuse_mode == "paper":
                # Acquisition only needs own/background sims — skip the L×(L+1) GEMM.
                acq_indices = np.concatenate([labeled_array, candidates])
                acq = predict_tensors(
                    trained,
                    self.features,
                    self.labels,
                    acq_indices,
                    self.config,
                    self.device,
                    return_latents=False,
                    similarity_mode="own_bg",
                )
                labeled_count = int(labeled_array.size)
                labeled_tensors = {name: value[:labeled_count] for name, value in acq.items()}
                pool_tensors = {name: value[labeled_count:] for name, value in acq.items()}
            elif fuse_mode in {"weighted", "random"}:
                pool_tensors = predict_tensors(
                    trained,
                    self.features,
                    self.labels,
                    candidates,
                    self.config,
                    self.device,
                    return_latents=False,
                    similarity_mode="own_bg",
                )
            # Prototype similarity metrics stay on-device; only D2H sims on the final round for npz.
            validation_metrics = {}
            test_metrics = {}
            validation_metrics.update(
                prototype_similarity_metrics_torch(
                    eval_tensors["labels"][:split], eval_tensors["prototype_similarities"][:split]
                )
            )
            test_metrics.update(
                prototype_similarity_metrics_torch(
                    eval_tensors["labels"][split:], eval_tensors["prototype_similarities"][split:]
                )
            )
            if copy_stream is not None:
                copy_stream.synchronize()
            validation_prediction = {name: value.numpy() for name, value in validation_host.items()}
            test_prediction = {name: value.numpy() for name, value in test_host.items()}
            threshold = float(self.training_cfg.get("threshold", 0.5))
            validation_metrics.update(
                multilabel_metrics(
                    validation_prediction["labels"], validation_prediction["probabilities"], threshold
                )
            )
            test_metrics.update(
                multilabel_metrics(
                    test_prediction["labels"], test_prediction["probabilities"], threshold
                )
            )
            queries: list[int] = []
            acquisition: dict[str, Any] = {}
            diagnostics_prediction = validation_prediction
            if will_query:
                assert pool_tensors is not None
                score_tensor: torch.Tensor | None = None
                if strategy == "random":
                    scores = rng.random(len(candidates))
                    components = {"combined": scores}
                elif strategy == "comal":
                    if formula == "paper":
                        assert labeled_tensors is not None
                        expected_cardinality = labeled_tensors["labels"].sum(dim=1).mean()
                        prototypes = trained.comal.prototypes.detach()
                        labeled_sims = labeled_tensors["prototype_similarities"]
                        pool_sims = pool_tensors["prototype_similarities"]
                        if labeled_sims.shape[-1] == 2:
                            labeled_own = labeled_sims[..., 0]
                            pool_own = pool_sims[..., 0]
                        else:
                            num_labels = int(labeled_tensors["labels"].shape[1])
                            label_index = torch.arange(num_labels, device=self.device)
                            labeled_own = labeled_sims[:, label_index, label_index]
                            pool_own = pool_sims[:, label_index, label_index]
                        thresholds = positive_similarity_thresholds(
                            None,
                            labeled_tensors["labels"],
                            prototypes,
                            own_similarity=labeled_own,
                        )
                        parts = paper_comal_acquisition_scores(
                            pool_tensors["probabilities"],
                            None,
                            prototypes,
                            thresholds,
                            expected_cardinality=expected_cardinality,
                            own_similarity=pool_own,
                        )
                        component_names = (
                            "inverse_positive_evidence",
                            "cardinality_mismatch",
                            "prototype_positive_count",
                            "combined",
                        )
                    elif formula == "weighted":
                        expected_cardinality = float(
                            self.labels[np.asarray(labeled)].sum(axis=1).mean()
                        )
                        pool_sims = pool_tensors["prototype_similarities"]
                        pool_own = pool_sims[..., 0] if pool_sims.shape[-1] == 2 else None
                        parts = comal_acquisition_scores(
                            pool_tensors["probabilities"],
                            None,
                            trained.comal.prototypes.detach(),
                            expected_cardinality=expected_cardinality,
                            uncertainty_weight=float(cfg.get("uncertainty_weight", 0.5)),
                            prototype_weight=float(cfg.get("prototype_weight", 0.35)),
                            cardinality_weight=float(cfg.get("cardinality_weight", 0.15)),
                            own_similarity=pool_own,
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
                    component_host = {
                        name: _async_to_host(getattr(parts, name).detach(), copy_stream)
                        for name in component_names
                    }
                else:
                    raise ValueError("active_learning.strategy must be comal or random")
                # Pool diagnostics only need own/background sims, not the full L×(L+1) cube.
                pool_sims = pool_tensors["prototype_similarities"]
                if pool_sims.shape[-1] == 2:
                    compact_sims = pool_sims
                else:
                    num_labels = int(pool_tensors["probabilities"].shape[1])
                    label_index = torch.arange(num_labels, device=self.device)
                    compact_sims = torch.stack(
                        (pool_sims[:, label_index, label_index], pool_sims[:, :, -1]),
                        dim=-1,
                    )
                host = {
                    "indices": _async_to_host(pool_tensors["indices"], copy_stream),
                    "labels": _async_to_host(pool_tensors["labels"], copy_stream),
                    "probabilities": _async_to_host(pool_tensors["probabilities"], copy_stream),
                    "prototype_similarities": _async_to_host(compact_sims, copy_stream),
                }
                if copy_stream is not None:
                    copy_stream.synchronize()
                elif self.device.type == "cuda":
                    torch.cuda.current_stream(self.device).synchronize()
                pool = {name: value.numpy() for name, value in host.items()}
                if strategy == "comal":
                    components = {name: value.numpy() for name, value in component_host.items()}
                    scores = components["combined"]
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
                # Final / no-query rounds diagnose on validation; similarities already suffice.
                diagnostics_prediction = validation_prediction
            diagnostics = build_round_diagnostics(
                diagnostics_prediction["labels"],
                diagnostics_prediction["probabilities"],
                diagnostics_prediction.get("latents"),
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
                query_array = np.asarray(queries, dtype=np.int64)
                labeled_mask[query_array] = True
                labeled = np.unique(
                    np.concatenate([np.asarray(labeled, dtype=np.int64), query_array])
                ).tolist()
            previous = trained if bool(self.training_cfg.get("inherit_across_rounds", False)) else None
        self._save_checkpoint(trained, rounds - 1)
        prediction_payload = {
            "validation_labels": validation_prediction["labels"],
            "validation_probabilities": validation_prediction["probabilities"],
            "test_labels": test_prediction["labels"],
            "test_probabilities": test_prediction["probabilities"],
        }
        if validation_prediction.get("prototype_similarities") is not None:
            prediction_payload["validation_prototype_similarities"] = validation_prediction[
                "prototype_similarities"
            ]
        if test_prediction.get("prototype_similarities") is not None:
            prediction_payload["test_prototype_similarities"] = test_prediction["prototype_similarities"]
        # Uncompressed npz is much faster to write; size is modest for MIMIC splits.
        np.savez(self.output_dir / "final_predictions.npz", **prediction_payload)
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
