"""Non-interactive diagnostics and publication-ready experiment plots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .data import audit_records, load_records
from .training import label_matrix


COLORS = ("#287271", "#d9895b", "#5b6f9c", "#b35c44")
# Diagnostics plots prioritize throughput over print DPI.
_SAVE_DPI = 140


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )


def explore_dataset(config: dict[str, Any], output_dir: str | Path | None = None) -> dict[str, Any]:
    dataset = config.get("dataset", {})
    prepared = Path(dataset.get("prepared_dir", "prepared/mimic_iii"))
    output = Path(output_dir or prepared / "exploration")
    output.mkdir(parents=True, exist_ok=True)
    records = load_records(prepared)
    label_names = tuple(json.loads((prepared / "labels.json").read_text(encoding="utf-8"))["labels"])
    labels = label_matrix(records, label_names)
    files: dict[str, str] = {}

    positives = labels.sum(axis=0)
    order = np.argsort(positives)
    fig, axis = plt.subplots(figsize=(9, max(5, len(label_names) * 0.22)))
    axis.barh(np.arange(len(order)), positives[order], color=COLORS[0])
    axis.set_yticks(np.arange(len(order)), [label_names[index] for index in order])
    axis.set_xlabel("Admissions with positive ICD-9 group")
    axis.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    files["label_prevalence"] = "label_prevalence.png"
    fig.savefig(output / files["label_prevalence"], dpi=_SAVE_DPI)
    plt.close(fig)

    cardinality = labels.sum(axis=1).astype(int)
    bins = np.arange(cardinality.max() + 2) - 0.5
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.hist(cardinality, bins=bins, color=COLORS[1], edgecolor="white")
    axis.set(xlabel="Positive ICD-9 groups per admission", ylabel="Admissions")
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    files["cardinality"] = "label_cardinality.png"
    fig.savefig(output / files["cardinality"], dpi=_SAVE_DPI)
    plt.close(fig)

    top = np.argsort(-positives)[: min(25, len(label_names))]
    cooccurrence = labels[:, top].T @ labels[:, top]
    denominator = np.sqrt(np.outer(np.diag(cooccurrence), np.diag(cooccurrence))).clip(min=1)
    normalised = cooccurrence / denominator
    fig, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(normalised, cmap="viridis", vmin=0, vmax=1)
    names = [label_names[index] for index in top]
    axis.set_xticks(np.arange(len(names)), names, rotation=60, ha="right")
    axis.set_yticks(np.arange(len(names)), names)
    axis.set_title("ICD-9 co-occurrence cosine (top labels)")
    fig.colorbar(image, ax=axis, label="cosine")
    fig.tight_layout()
    files["cooccurrence"] = "label_cooccurrence.png"
    fig.savefig(output / files["cooccurrence"], dpi=_SAVE_DPI)
    plt.close(fig)

    batches = np.asarray([4, 8, 16, 32, 64, 128])
    tokens = batches * len(label_names)
    similarity_mib = tokens.astype(float) ** 2 * 4 / 1024**2
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(batches, similarity_mib, marker="o", color=COLORS[2])
    configured = int(config.get("training", {}).get("comal_batch_size", 32))
    axis.axvline(configured, color=COLORS[3], linestyle="--", label=f"configured={configured}")
    axis.set(xlabel="CoMAL batch size", ylabel="One FP32 similarity matrix (MiB)", yscale="log")
    axis.grid(alpha=0.2)
    axis.legend()
    fig.tight_layout()
    files["method_scaling"] = "comal_memory_scaling.png"
    fig.savefig(output / files["method_scaling"], dpi=_SAVE_DPI)
    plt.close(fig)

    audit = audit_records(records, label_names)
    split_positive = {
        split: labels[[record.split == split for record in records]].sum(axis=0)
        for split in ("train", "validation", "test")
    }
    audit["split_label_coverage"] = {
        split: int((values > 0).sum()) for split, values in split_positive.items()
    }
    audit["cooccurrence_density"] = float(((labels.T @ labels) > 0).mean())
    audit["feature_feasibility"] = {
        "coMAL_batch_tokens": configured * len(label_names),
        "coMAL_similarity_matrix_mib": float((configured * len(label_names)) ** 2 * 4 / 1024**2),
        "recommendation": "keep comal_batch_size bounded; scale cached-feature classifier batches independently",
    }
    _write_json(output / "dataset_audit.json", audit)
    report = {"output_dir": str(output), "files": files, "audit": str(output / "dataset_audit.json")}
    _write_json(output / "manifest.json", report)
    return report


def visualize_experiment(experiment_dir: str | Path, output_dir: str | Path | None = None) -> dict[str, Any]:
    experiment = Path(experiment_dir)
    output = Path(output_dir or experiment / "figures")
    output.mkdir(parents=True, exist_ok=True)
    state = json.loads((experiment / "active_state.json").read_text(encoding="utf-8"))
    prediction = np.load(experiment / "final_predictions.npz")
    records = state["records"]
    labels = state["label_names"]
    files: dict[str, str] = {}

    rounds = [record["round_index"] for record in records]
    labeled = [record["labeled_before_query"] for record in records]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for index, split in enumerate(("validation", "test")):
        for metric, color in (
            ("auprc_macro", COLORS[0]),
            ("auprc_micro", COLORS[1]),
            ("f1_micro", COLORS[2]),
        ):
            axes[index].plot(
                labeled,
                [record[f"{split}_metrics"][metric] for record in records],
                marker="o",
                label=metric,
                color=color,
            )
        axes[index].set(xlabel="Labeled admissions", ylabel="Metric", title=split.title())
        axes[index].grid(alpha=0.2)
        axes[index].legend()
    fig.tight_layout()
    files["learning_curve"] = "learning_curves.png"
    fig.savefig(output / files["learning_curve"], dpi=_SAVE_DPI)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for round_index, record in enumerate(records):
        axes[0].plot(record["training_history"]["classifier_loss"], alpha=0.75, label=f"r{round_index}")
        axes[1].plot(record["training_history"]["comal_loss"], alpha=0.75, label=f"r{round_index}")
    axes[0].set(title="Classifier training", xlabel="Epoch", ylabel="Loss")
    axes[1].set(title="CoMAL training", xlabel="Epoch", ylabel="Loss")
    for axis in axes:
        axis.grid(alpha=0.2)
    if len(records) <= 10:
        axes[1].legend(ncol=2)
    fig.tight_layout()
    files["losses"] = "training_losses.png"
    fig.savefig(output / files["losses"], dpi=_SAVE_DPI)
    plt.close(fig)

    classifier_time = [record["timing"]["classifier_training_sec"] for record in records]
    comal_time = [record["timing"]["comal_training_sec"] for record in records]
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(rounds, classifier_time, label="classifier", color=COLORS[0])
    axis.bar(rounds, comal_time, bottom=classifier_time, label="CoMAL", color=COLORS[1])
    axis.set(xlabel="Round", ylabel="Wall seconds", title="Training time by round")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    files["timing"] = "round_timing.png"
    fig.savefig(output / files["timing"], dpi=_SAVE_DPI)
    plt.close(fig)

    test_labels = prediction["test_labels"]
    test_probs = prediction["test_probabilities"]
    from sklearn.metrics import average_precision_score

    per_label = []
    for index in range(len(labels)):
        value = (
            average_precision_score(test_labels[:, index], test_probs[:, index])
            if np.unique(test_labels[:, index]).size == 2
            else np.nan
        )
        per_label.append(value)
    valid = np.flatnonzero(np.isfinite(per_label))
    order = valid[np.argsort(np.asarray(per_label)[valid])]
    fig, axis = plt.subplots(figsize=(9, max(5, len(order) * 0.22)))
    axis.barh(np.arange(len(order)), np.asarray(per_label)[order], color=COLORS[2])
    axis.set_yticks(np.arange(len(order)), [labels[index] for index in order])
    axis.set(xlabel="Test average precision", xlim=(0, 1))
    axis.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    files["per_label"] = "test_per_label_auprc.png"
    fig.savefig(output / files["per_label"], dpi=_SAVE_DPI)
    plt.close(fig)

    report = {"experiment": str(experiment), "output_dir": str(output), "files": files}
    _write_json(output / "manifest.json", report)
    return report


def compare_experiments(
    experiment_dirs: list[str | Path], names: list[str], output: str | Path
) -> dict[str, Any]:
    if len(experiment_dirs) != len(names):
        raise ValueError("one display name is required for each experiment")
    curves: dict[str, Any] = {}
    fig, axis = plt.subplots(figsize=(8, 5))
    for position, (directory, name) in enumerate(zip(experiment_dirs, names)):
        state = json.loads((Path(directory) / "active_state.json").read_text(encoding="utf-8"))
        x = [record["labeled_before_query"] for record in state["records"]]
        y = [record["test_metrics"]["auprc_macro"] for record in state["records"]]
        area = float(np.trapezoid(y, x) / max(x[-1] - x[0], 1)) if len(x) > 1 else y[0]
        curves[name] = {"labeled": x, "test_auprc_macro": y, "normalised_area": area}
        axis.plot(x, y, marker="o", label=name, color=COLORS[position % len(COLORS)])
    axis.set(xlabel="Labeled admissions", ylabel="Test macro AUPRC", title="Active-learning comparison")
    axis.grid(alpha=0.2)
    axis.legend()
    fig.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=_SAVE_DPI)
    plt.close(fig)
    report = {"figure": str(output), "curves": curves}
    _write_json(output.with_suffix(".json"), report)
    return report
