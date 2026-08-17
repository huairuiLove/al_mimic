from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import al_mimic.tasks.mimic_iii.training as training_module
from al_mimic.tasks.mimic_iii.config import load_config
from al_mimic.tasks.mimic_iii.training import (
    _classifier_parameter_groups,
    train_multimodal_round,
)


def test_formal_config_enables_the_round_training_controls() -> None:
    config = load_config(Path(__file__).parents[4] / "configs/experiments/mimic_iii/comal.yaml")
    training = config["training"]

    assert training["epochs"] == 80
    assert training["optimizer_steps_per_round"] == 1200
    assert training["optimizer"] == "adamw"
    assert training["weight_decay"] == pytest.approx(0.01)
    assert training["bert_learning_rate"] == pytest.approx(2e-5)
    assert training["bert_layerwise_lr_decay"] == pytest.approx(0.95)
    assert training["early_stopping_patience"] == 5


def test_full_cohort_config_remains_the_yang_wu_task() -> None:
    config = load_config(Path(__file__).parents[4] / "configs/experiments/mimic_iii/full_cohort_random.yaml")

    assert config["dataset"]["cohort_mode"] == "full_cohort"
    assert config["preprocessing"]["cohort"] == "mimic_iii_all_icu"
    assert config["preprocessing"]["task"] == "Diagnoses"
    assert config["preprocessing"]["label_format"] == "icd9_top3_multihot"
    assert config["preprocessing"]["expected_label_count"] == 915
    assert config["model"]["architecture"] == "yang_wu_bertencoder"
    assert config["model"]["output_size"] == 915


class _EncoderStack(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.ModuleList([torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)])


class _LayeredBert(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embeddings = torch.nn.Embedding(16, 4)
        self.encoder = _EncoderStack()
        self.pooler = torch.nn.Linear(4, 4)


class _LayeredClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.text_encoder = _LayeredBert()
        self.multimodal_head = torch.nn.Linear(4, 2)


def _training_config(**overrides) -> dict:
    training = {
        "epochs": 10,
        "optimizer_steps_per_round": 20,
        "early_stopping_patience": 100,
        "early_stopping_min_delta": 0.0,
        "learning_rate": 1e-3,
        "bert_learning_rate": 2e-4,
        "bert_layerwise_lr_decay": 0.95,
        "weight_decay": 0.01,
        "warmup_proportion": 0.0,
        "gradient_clip": 1.0,
        "precision": "fp32",
        "batch_size": 2,
        "eval_batch_size": 2,
        "num_workers": 0,
        "pin_memory": False,
    }
    training.update(overrides)
    return {
        "model": {"seed": 11},
        "training": training,
        "active_learning": {"strategy": "random"},
    }


def test_bert_parameter_groups_use_layerwise_lr_and_skip_decay_for_biases() -> None:
    classifier = _LayeredClassifier()
    training = _training_config()["training"]
    groups = _classifier_parameter_groups(classifier, training)
    group_by_parameter = {id(parameter): group for group in groups for parameter in group["params"]}

    embedding_group = group_by_parameter[id(classifier.text_encoder.embeddings.weight)]
    bottom_group = group_by_parameter[id(classifier.text_encoder.encoder.layer[0].weight)]
    top_group = group_by_parameter[id(classifier.text_encoder.encoder.layer[1].weight)]
    pooler_group = group_by_parameter[id(classifier.text_encoder.pooler.weight)]
    multimodal_group = group_by_parameter[id(classifier.multimodal_head.weight)]
    bias_group = group_by_parameter[id(classifier.multimodal_head.bias)]

    assert embedding_group["lr"] == pytest.approx(2e-4 * 0.95**2)
    assert bottom_group["lr"] == pytest.approx(2e-4 * 0.95)
    assert top_group["lr"] == pytest.approx(2e-4)
    assert pooler_group["lr"] == pytest.approx(2e-4)
    assert multimodal_group["lr"] == pytest.approx(1e-3)
    assert multimodal_group["weight_decay"] == pytest.approx(0.01)
    assert bias_group["weight_decay"] == 0.0
    assert sum(len(group["params"]) for group in groups) == sum(
        parameter.requires_grad for parameter in classifier.parameters()
    )


class _TinyRoundClassifier(torch.nn.Module):
    modality_names = ("clinical_notes", "time_series", "time_invariant")

    def __init__(self) -> None:
        super().__init__()
        self.text_encoder = torch.nn.Linear(3, 4)
        self.series_encoder = torch.nn.Linear(3, 4)
        self.static_encoder = torch.nn.Linear(3, 4)
        self.fusion = torch.nn.Linear(12, 4)
        self.classifier = torch.nn.Linear(4, 2)
        self.feature_dim = 4
        self.num_labels = 2

    def encode_modalities(self, batch):
        values = batch["features"]
        return (
            torch.tanh(self.text_encoder(values)),
            torch.tanh(self.series_encoder(values)),
            torch.tanh(self.static_encoder(values)),
        )

    def forward_from_modalities(self, text, series, static):
        features = torch.tanh(self.fusion(torch.cat((text, series, static), dim=1)))
        logits = self.classifier(features)
        return {
            "logits": logits,
            "probabilities": torch.sigmoid(logits),
            "features": features,
        }

    def forward(self, batch, *, return_tokens=False):
        del return_tokens
        return self.forward_from_modalities(*self.encode_modalities(batch))


class _TinyStore:
    def __init__(self) -> None:
        generator = torch.Generator().manual_seed(3)
        self.features = torch.randn(8, 3, generator=generator)
        self.labels = (torch.randn(8, 2, generator=generator) > 0).float()
        self.audit = SimpleNamespace(label_count=2)

    def make_loader(
        self,
        indices,
        *,
        batch_size,
        shuffle,
        num_workers,
        pin_memory,
    ):
        del shuffle, num_workers, pin_memory
        selected = np.asarray(list(indices), dtype=np.int64)
        return [
            {
                "features": self.features[selected[start : start + batch_size]],
                "labels": self.labels[selected[start : start + batch_size]],
            }
            for start in range(0, len(selected), batch_size)
        ]


def _train_tiny_round(monkeypatch, training_overrides):
    monkeypatch.setattr(
        "al_mimic.tasks.mimic_iii.training.build_classifier",
        lambda store, config, device: _TinyRoundClassifier().to(device),
    )
    return train_multimodal_round(
        _TinyStore(),
        [0, 1, 2, 3],
        _training_config(**training_overrides),
        torch.device("cpu"),
        validation_indices=[4, 5],
    )


def test_modality_mixup_generates_virtual_samples_without_changing_step_budget(monkeypatch) -> None:
    config = _training_config(epochs=3, optimizer_steps_per_round=3)
    config["mixup"] = {
        "enabled": True,
        "space": "modalities",
        "alpha": 1.0,
        "weight": 1.0,
        "pairing": "random",
        "anchor_quantile": 1.0,
        "keep_anchor": False,
    }
    monkeypatch.setattr(
        "al_mimic.tasks.mimic_iii.training.build_classifier",
        lambda store, config, device: _TinyRoundClassifier().to(device),
    )

    trained = train_multimodal_round(
        _TinyStore(),
        [0, 1, 2, 3],
        config,
        torch.device("cpu"),
        validation_indices=[4, 5],
    )

    assert trained.training_summary["optimizer_steps"] == 3
    assert trained.training_summary["mixup_space"] == "modalities"
    assert trained.training_summary["mixup_virtual_samples"] > 0
    assert sum(trained.history["mixup_virtual_samples"]) > 0
    assert all(value > 0.0 for value in trained.history["mixup_loss"])


def test_round_stops_at_the_fixed_optimizer_step_budget(monkeypatch) -> None:
    trained = _train_tiny_round(
        monkeypatch,
        {
            "epochs": 3,
            "optimizer_steps_per_round": 3,
        },
    )

    assert trained.training_summary["optimizer_steps"] == 3
    assert trained.training_summary["epochs_ran"] == 2
    assert trained.training_summary["stop_reason"] == "optimizer_step_budget"
    assert len(trained.history["validation_loss"]) == 2


def test_round_early_stops_after_consecutive_stale_validation_epochs(monkeypatch) -> None:
    best_states = []
    clone_module_state = training_module._clone_module_state

    def capture_best_state(module):
        state = clone_module_state(module)
        best_states.append(state)
        return state

    monkeypatch.setattr(training_module, "_clone_module_state", capture_best_state)
    trained = _train_tiny_round(
        monkeypatch,
        {
            "early_stopping_patience": 2,
            "early_stopping_min_delta": 1e9,
        },
    )

    assert trained.training_summary["stop_reason"] == "early_stopping"
    assert trained.training_summary["epochs_ran"] == 3
    assert trained.training_summary["optimizer_steps"] == 6
    assert trained.training_summary["best_epoch"] == 1
    assert trained.training_summary["best_optimizer_step"] == 2
    assert len(best_states) == 1
    for name, value in trained.classifier.state_dict().items():
        assert torch.equal(value, best_states[0][name])
