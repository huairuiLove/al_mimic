from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import pytest

from mimic_comal.data import MIMICRecord
from mimic_comal.runner import ActiveLearningExperiment


@pytest.mark.parametrize("strategy", ["comal", "mm_comal", "mosaic"])
def test_synthetic_active_learning_run(tmp_path, strategy: str) -> None:
    prepared = tmp_path / "prepared"
    features_dir = prepared / "features"
    output = tmp_path / "experiments"
    features_dir.mkdir(parents=True)
    labels = ("A", "B", "C", "D")
    records = []
    for index in range(36):
        split = "train" if index < 24 else "validation" if index < 30 else "test"
        positive = tuple(
            label for position, label in enumerate(labels) if (index + position) % (position + 2) == 0
        )
        records.append(MIMICRecord(index, str(index), str(index), split, positive or ("A",), "text"))
    with (prepared / "records.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record)) + "\n")
    (prepared / "labels.json").write_text(json.dumps({"labels": labels}))
    rng = np.random.default_rng(3)
    np.save(features_dir / "features.npy", rng.normal(size=(len(records), 30)).astype(np.float16))
    (features_dir / "metadata.json").write_text(
        json.dumps(
            {
                "encoder": "multimodal_scratch",
                "initialization": "random",
                "pretrained_weights": False,
                "modalities": [
                    {"name": "clinical_note", "start": 0, "stop": 6, "shape": [6]},
                    {"name": "icu_measurements", "start": 6, "stop": 22, "shape": [4, 4]},
                    {"name": "demographics", "start": 22, "stop": 30, "shape": [8]},
                ],
            }
        )
    )
    config = {
        "dataset": {"prepared_dir": str(prepared), "feature_dir": str(features_dir)},
        "experiment": {"name": strategy, "output_root": str(output)},
        "features": {"encoder": "multimodal_scratch"},
        "model": {
            "architecture": "multimodal_transformer_scratch",
            "initialization": "random",
            "fusion_dim": 16,
            "num_heads": 4,
            "measurement_layers": 1,
            "fusion_layers": 1,
            "dropout": 0.0,
            "modality_dropout": 0.1 if strategy != "comal" else 0.0,
        },
        "comal": {
            "label_dim": 4,
            "prototype_dim": 4,
            "anchor_chunk_size": 16,
            "cross_modal_weight": 0.1,
        },
        "training": {
            "device": "cpu",
            "precision": "fp32",
            "batch_size": 8,
            "comal_batch_size": 4,
            "eval_batch_size": 16,
            "epochs": 1,
            "comal_epochs": 1,
            "num_workers": 0,
            "pin_memory": False,
            "seed": 2,
        },
        "active_learning": {
            "strategy": strategy,
            "initial_labeled": 8,
            "query_size": 3,
            "candidate_size": 12,
            "rounds": 2,
        },
        "acquisition": {
            "formula": "paper_mm" if strategy == "mm_comal" else "paper",
            "mm": {
                "alpha": 1.0,
                "reliability_shrinkage": 2.0,
                "threshold_shrinkage": 2.0,
                "threshold_estimator": "shrunk",
            },
        },
        "mosaic": {
            "eta": 0.25,
            "partners": 1,
            "mixup_closure_samples": 4,
            "workset_size": 6,
            "synergy_workset_size": 4,
            "damping": 0.1,
            "fusion_batch_size": 32,
            "value_batch_size": 32,
            "deflation_steps": 1,
        },
    }
    result = ActiveLearningExperiment(config).run()
    experiment = output / strategy
    assert result["rounds"] == 2
    assert (experiment / "active_state.json").is_file()
    assert (experiment / "final_metrics.json").is_file()
    assert (experiment / "checkpoints" / "final.pt").is_file()
    state = json.loads((experiment / "active_state.json").read_text())
    assert len(state["records"]) == 2
    assert len(state["records"][0]["query_indices"]) == 3
