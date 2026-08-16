"""Lazy supervised training entry point for the native MDS-ED adapter."""

from __future__ import annotations

import importlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .constants import DETERIORATION_LABEL_COUNT, DIAGNOSIS_LABEL_COUNT
from .tabular import TabularSpec


class TrainingDependencyError(RuntimeError):
    """Raised when optional supervised-training dependencies are unavailable."""


@dataclass(frozen=True, slots=True)
class SupervisedTrainingConfig:
    task: str = "diagnoses"
    epochs: int = 20
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-3
    device: str = "cuda"
    num_workers: int = 0
    model_dim: int = 512
    temporal_layers: int = 4
    tabular_dim: int = 128
    dropout: float = 0.1

    def validate(self) -> None:
        if self.task not in {"diagnoses", "deterioration"}:
            raise ValueError(f"unsupported MDS-ED task: {self.task}")
        if self.epochs < 1 or self.batch_size < 1 or self.num_workers < 0:
            raise ValueError("epochs/batch_size must be positive and num_workers non-negative")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")


def _load_training_dependencies():
    try:
        torch = importlib.import_module("torch")
        model_module = importlib.import_module("al_mimic.tasks.mds_ed.s4_backend.model")
        data_module = importlib.import_module("al_mimic.tasks.mds_ed.data")
    except (ImportError, OSError) as exc:
        raise TrainingDependencyError(
            "native MDS-ED supervised training requires a working PyTorch installation. "
            "Install the project training dependencies; no external MDS-ED source checkout "
            "is loaded or executed."
        ) from exc
    return torch, model_module, data_module


def _masked_bce(torch, logits, labels):
    valid = torch.isfinite(labels)
    if not bool(valid.any()):
        raise ValueError("supervised MDS-ED batch contains no finite labels")
    targets = torch.where(valid, labels, torch.zeros_like(labels))
    losses = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    return losses.masked_select(valid).mean()


def _move_batch(torch, batch: dict[str, Any], device):
    return {name: value.to(device, non_blocking=device.type == "cuda") for name, value in batch.items()}


def train_supervised(
    prepared_dir: str | Path,
    output_dir: str | Path,
    config: SupervisedTrainingConfig | None = None,
) -> dict[str, Any]:
    """Train the package-local model; heavy dependencies load only in this call."""
    settings = config or SupervisedTrainingConfig()
    settings.validate()
    torch, model_module, data_module = _load_training_dependencies()
    if settings.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"MDS-ED training requested device={settings.device!r}, but CUDA is unavailable; "
            "set device='cpu' explicitly for a CPU run"
        )
    device = torch.device(settings.device)
    root = Path(prepared_dir)
    manifest = json.loads((root / "tabular_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("task") != settings.task:
        raise ValueError(
            f"tabular artifacts contain task={manifest.get('task')!r}, "
            f"but training requested {settings.task!r}"
        )
    spec = TabularSpec.load(root / "tabular_spec.json")
    output_size = DIAGNOSIS_LABEL_COUNT if settings.task == "diagnoses" else DETERIORATION_LABEL_COUNT
    actual_labels = (
        len(spec.diagnosis_columns) if settings.task == "diagnoses" else len(spec.deterioration_columns)
    )
    if actual_labels != output_size:
        raise ValueError(f"MDS-ED {settings.task} requires {output_size} labels, found {actual_labels}")
    loaders = data_module.make_dataloaders(
        root,
        batch_size=settings.batch_size,
        num_workers=settings.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = model_module.NativeMdsEdTemporalAdapter(
        output_size=output_size,
        continuous_dim=spec.continuous_dim,
        category_sizes=spec.category_sizes,
        model_dim=settings.model_dim,
        temporal_layers=settings.temporal_layers,
        tabular_dim=settings.tabular_dim,
        dropout=settings.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    for _epoch in range(settings.epochs):
        model.train()
        train_total = train_rows = 0
        for raw_batch in loaders["train"]:
            batch = _move_batch(torch, raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["ecg"], batch["continuous"], batch["categorical"])
            loss = _masked_bce(torch, logits, batch["labels"])
            loss.backward()
            optimizer.step()
            batch_rows = int(batch["labels"].shape[0])
            train_total += float(loss.detach()) * batch_rows
            train_rows += batch_rows
        model.eval()
        val_total = val_rows = 0
        with torch.inference_mode():
            for raw_batch in loaders["val"]:
                batch = _move_batch(torch, raw_batch, device)
                logits = model(batch["ecg"], batch["continuous"], batch["categorical"])
                loss = _masked_bce(torch, logits, batch["labels"])
                batch_rows = int(batch["labels"].shape[0])
                val_total += float(loss) * batch_rows
                val_rows += batch_rows
        if train_rows == 0 or val_rows == 0:
            raise ValueError("MDS-ED train and validation splits must both be non-empty")
        history["train_loss"].append(train_total / train_rows)
        history["val_loss"].append(val_total / val_rows)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "native_temporal_adapter.pt"
    torch.save(
        {
            "backend": model.backend_name,
            "model": model.state_dict(),
            "training": asdict(settings),
            "tabular_spec": spec.to_dict(),
            "history": history,
        },
        checkpoint,
    )
    result = {
        "backend": model.backend_name,
        "checkpoint": str(checkpoint),
        "epochs": settings.epochs,
        "history": history,
    }
    (output / "training_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result
