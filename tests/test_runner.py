from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np

from mimic_comal.data import MIMICRecord
from mimic_comal.runner import ActiveLearningExperiment


def test_synthetic_active_learning_run(tmp_path) -> None:
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
    np.save(features_dir / "features.npy", rng.normal(size=(len(records), 12)).astype(np.float16))
    config = {
        "dataset": {"prepared_dir": str(prepared), "feature_dir": str(features_dir)},
        "experiment": {"name": "test", "output_root": str(output)},
        "model": {"hidden_dims": [16, 8], "dropout": 0.0},
        "comal": {"label_dim": 4, "prototype_dim": 4, "anchor_chunk_size": 16},
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
            "strategy": "comal",
            "initial_labeled": 8,
            "query_size": 3,
            "candidate_size": 12,
            "rounds": 2,
        },
    }
    result = ActiveLearningExperiment(config).run()
    experiment = output / "test"
    assert result["rounds"] == 2
    assert (experiment / "active_state.json").is_file()
    assert (experiment / "final_metrics.json").is_file()
    assert (experiment / "checkpoints" / "final.pt").is_file()
    state = json.loads((experiment / "active_state.json").read_text())
    assert len(state["records"]) == 2
    assert len(state["records"][0]["query_indices"]) == 3
