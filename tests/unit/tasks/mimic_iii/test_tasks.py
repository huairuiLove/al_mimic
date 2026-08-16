from __future__ import annotations

import csv
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
import yaml

from al_mimic.tasks.mimic_iii.config import load_config
from al_mimic.tasks.mimic_iii.data import audit_split_hdf5
from al_mimic.tasks.mimic_iii.metrics import ranking_metrics, task_multilabel_metrics
from al_mimic.tasks.mimic_iii.model import YangWuBertEncoderClassifier
from al_mimic.tasks.mimic_iii.preprocessing.build_ccs_172_labels import build_ccs_labels
from al_mimic.tasks.mimic_iii.tasks import TASKS, task_spec

ROOT = Path(__file__).parents[4]


class _TinyBert(torch.nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(32, hidden_dim)

    def forward(self, input_ids, token_type_ids, attention_mask):
        del token_type_ids
        values = self.embedding(input_ids)
        weights = attention_mask.unsqueeze(-1).float()
        pooled = (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        return type("Output", (), {"pooler_output": pooled})()


@pytest.mark.parametrize(
    ("path", "task_id", "labels", "metric"),
    [
        ("configs/experiments/mimic_iii/comal.yaml", "icd9_diagnoses", 915, "recall_at_30"),
        (
            "configs/experiments/mimic_iii/phenotyping_25_comal.yaml",
            "phenotyping_25",
            25,
            "macro_auprc",
        ),
        (
            "configs/experiments/mimic_iii/phenotyping_ccs_172_comal.yaml",
            "phenotyping_ccs_172",
            172,
            "macro_auprc",
        ),
    ],
)
def test_registered_task_configs(path, task_id, labels, metric) -> None:
    config = load_config(ROOT / path)
    spec = task_spec(config)
    assert spec.task_id == task_id
    assert config["task"]["native_multilabel"] is True
    assert config["task"]["query_unit"] == "icu_stay"
    assert config["model"]["output_size"] == labels
    assert config["evaluation"]["primary_metric"] == metric


def test_all_three_tasks_are_registered() -> None:
    assert set(TASKS) == {"icd9_diagnoses", "phenotyping_25", "phenotyping_ccs_172"}


def test_phenotyping_metrics_are_ranking_metrics_not_top_k() -> None:
    labels = np.asarray([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=np.int8)
    probabilities = np.asarray([[0.9, 0.1], [0.2, 0.8], [0.7, 0.6], [0.1, 0.2]], dtype=np.float32)
    direct = ranking_metrics(labels, probabilities)
    config = load_config(ROOT / "configs/experiments/mimic_iii/phenotyping_25_comal.yaml")
    result = task_multilabel_metrics(config, labels, probabilities)
    assert result["macro_auprc"] == pytest.approx(direct["macro_auprc"])
    assert result["macro_auroc"] == pytest.approx(1.0)
    assert "recall_at_30" not in result


def test_two_modality_model_exposes_no_static_token() -> None:
    torch.manual_seed(4)
    model = YangWuBertEncoderClassifier(
        None,
        num_labels=25,
        time_invariant_dim=0,
        time_invariant_hidden_dim=0,
        time_series_dim=7,
        time_series_hidden_dim=16,
        time_series_layers=1,
        time_series_heads=4,
        text_hidden_dim=8,
        dropout=0.0,
        time_series_pooling="masked_mean",
        text_encoder=_TinyBert(8),
    ).eval()
    batch = {
        "time_series": torch.randn(3, 5, 7),
        "time_series_mask": torch.tensor(
            [[1, 1, 1, 0, 0], [1, 1, 0, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool
        ),
        "time_invariant": torch.empty(3, 0),
        "input_ids": torch.randint(0, 32, (3, 6)),
        "token_type_ids": torch.zeros(3, 6, dtype=torch.long),
        "attention_mask": torch.ones(3, 6, dtype=torch.long),
    }
    output = model(batch, return_tokens=True)
    assert model.modality_names == ("clinical_notes", "time_series")
    assert output["modality_tokens"].shape == (3, 2, 8)
    assert torch.allclose(model.fuse_from_tokens(output["modality_tokens"]), output["features"], atol=1e-6)


def _write_phenotyping_fixture(path: Path, *, task_id: str, labels: int) -> None:
    with h5py.File(path, "w") as handle:
        root = handle.create_group("with_notes")
        root.attrs["task_id"] = task_id
        root.create_dataset(
            "label_names",
            data=np.asarray([f"label-{index}" for index in range(labels)], dtype=object),
            dtype=h5py.string_dtype("utf-8"),
        )
        for split_index, split in enumerate(("train", "val", "test")):
            group = root.create_group(split)
            group.create_dataset("X", data=np.zeros((2, 256, 76), dtype=np.float32))
            group.create_dataset("time_series_mask", data=np.ones((2, 256), dtype=np.bool_))
            group.create_dataset("s", data=np.empty((2, 0), dtype=np.float32))
            group.create_dataset("input_ids", data=np.zeros((2, 512), dtype=np.int32))
            group.create_dataset("token_type_ids", data=np.zeros((2, 512), dtype=np.int8))
            group.create_dataset("attention_mask", data=np.ones((2, 512), dtype=np.int8))
            values = np.zeros((2, labels), dtype=np.int8)
            values[1, split_index % labels] = 1
            group.create_dataset("label", data=values)
            group.create_dataset("subject_id", data=np.asarray([10 * split_index + 1, 10 * split_index + 2]))


def test_phenotyping_hdf5_accepts_native_all_zero_rows(tmp_path: Path) -> None:
    artifact = tmp_path / "splits.hdf5"
    checkpoint = tmp_path / "pytorch_model.bin"
    checkpoint.touch()
    _write_phenotyping_fixture(artifact, task_id="phenotyping_25", labels=25)
    config = load_config(ROOT / "configs/experiments/mimic_iii/phenotyping_25_comal.yaml")
    config["dataset"]["split_hdf5"] = str(artifact)
    config["dataset"]["clinicalbert_checkpoint"] = str(checkpoint)
    audit = audit_split_hdf5(config)
    assert audit.label_count == 25
    assert audit.time_invariant_dim == 0
    assert len(audit.label_names) == 25


def test_ccs_builder_selects_labels_by_episode_count(tmp_path: Path) -> None:
    stays = tmp_path / "all_stays.csv"
    diagnoses = tmp_path / "all_diagnoses.csv"
    definitions = tmp_path / "definitions.yaml"
    with stays.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["SUBJECT_ID", "ICUSTAY_ID"])
        writer.writerows([(1, 11), (2, 22), (3, 33)])
    with diagnoses.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ICUSTAY_ID", "ICD9_CODE"])
        writer.writerows([(11, "001"), (22, "001"), (22, "002"), (33, "002")])
    definitions.write_text(
        yaml.safe_dump(
            {
                "First": {"id": 2, "codes": ["001"]},
                "Second": {"id": 1, "codes": ["002"]},
                "Rare": {"id": 3, "codes": ["003"]},
            }
        ),
        encoding="utf-8",
    )
    _subjects, selected, _by_stay, counts = build_ccs_labels(
        stays, diagnoses, definitions, minimum_episodes=2, expected_labels=2
    )
    assert selected == ("Second", "First")
    assert counts == {"First": 2, "Second": 2}
