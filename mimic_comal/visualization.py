"""Non-interactive diagnostics and publication-ready experiment plots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .multimodal_data import YangWuFeatureStore


COLORS = ("#287271", "#d9895b", "#5b6f9c", "#b35c44")
# Diagnostics plots prioritize throughput over print DPI.
_SAVE_DPI = 140


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )


def explore_dataset(config: dict[str, Any], output_dir: str | Path | None = None) -> dict[str, Any]:
    dataset = config.get("dataset", {})
    prepared = Path(dataset.get("prepared_dir", "prepared/yang_wu_diagnoses_48h"))
    output = Path(output_dir or prepared / "exploration")
    output.mkdir(parents=True, exist_ok=True)
    store = YangWuFeatureStore(config, validate=True)
    labels = store.labels
    label_names = tuple(f"ICD9-group-{index:04d}" for index in range(labels.shape[1]))
    files: dict[str, str] = {}

    positives = labels.sum(axis=0)
    order = np.argsort(positives)[-min(50, len(label_names)) :]
    fig, axis = plt.subplots(figsize=(9, 12))
    axis.barh(np.arange(len(order)), positives[order], color=COLORS[0])
    axis.set_yticks(np.arange(len(order)), [label_names[index] for index in order])
    axis.set_xlabel("ICU visits with positive three-digit ICD-9 group (top 50)")
    axis.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    files["label_prevalence"] = "label_prevalence.png"
    fig.savefig(output / files["label_prevalence"], dpi=_SAVE_DPI)
    plt.close(fig)

    cardinality = labels.sum(axis=1).astype(int)
    bins = np.arange(cardinality.max() + 2) - 0.5
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.hist(cardinality, bins=bins, color=COLORS[1], edgecolor="white")
    axis.set(xlabel="Positive ICD-9 groups per ICU visit", ylabel="ICU visits")
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
    configured = int(config.get("training", {}).get("batch_size", 32))
    axis.axvline(configured, color=COLORS[3], linestyle="--", label=f"configured={configured}")
    axis.set(xlabel="CoMAL batch size", ylabel="One FP32 similarity matrix (MiB)", yscale="log")
    axis.grid(alpha=0.2)
    axis.legend()
    fig.tight_layout()
    files["method_scaling"] = "comal_memory_scaling.png"
    fig.savefig(output / files["method_scaling"], dpi=_SAVE_DPI)
    plt.close(fig)

    audit: dict[str, Any] = {
        "protocol": "Yang-Wu Diagnoses 48h",
        "records": store.audit.total_samples,
        "labels": store.audit.label_count,
        "split_counts": store.audit.split_counts,
        "time_steps": store.audit.time_steps,
        "time_series_dim": store.audit.time_series_dim,
        "time_invariant_dim": store.audit.time_invariant_dim,
        "note_tokens": store.audit.note_tokens,
    }
    split_positive = {
        split: labels[store.indices(split)].sum(axis=0)
        for split in ("train", "val", "test")
    }
    audit["split_label_coverage"] = {
        split: int((values > 0).sum()) for split, values in split_positive.items()
    }
    audit["cooccurrence_density"] = float(((labels.T @ labels) > 0).mean())
    audit["feature_feasibility"] = {
        "coMAL_batch_tokens": configured * len(label_names),
        "coMAL_similarity_matrix_mib": float((configured * len(label_names)) ** 2 * 4 / 1024**2),
        "recommendation": "keep the joint classifier/CoMAL batch bounded; scale evaluation batches independently",
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
    files: dict[str, str] = {}

    rounds = [record["round_index"] for record in records]
    labeled = [record["labeled_count"] for record in records]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for index, split in enumerate(("validation", "test")):
        for metric, color in (
            ("recall_at_10", COLORS[0]),
            ("recall_at_20", COLORS[1]),
            ("recall_at_30", COLORS[2]),
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
    has_comal = any(record["training_history"].get("comal_loss") for record in records)
    for round_index, record in enumerate(records):
        axes[0].plot(record["training_history"]["classifier_loss"], alpha=0.75, label=f"r{round_index}")
        auxiliary = record["training_history"].get("comal_loss") or record["training_history"].get(
            "probe_loss", []
        )
        if auxiliary:
            axes[1].plot(auxiliary, alpha=0.75, label=f"r{round_index}")
    axes[0].set(title="Classifier training", xlabel="Epoch", ylabel="Loss")
    axes[1].set(
        title="CoMAL training" if has_comal else "Acquisition auxiliary training",
        xlabel="Epoch",
        ylabel="Loss",
    )
    for axis in axes:
        axis.grid(alpha=0.2)
    if len(records) <= 10:
        axes[1].legend(ncol=2)
    fig.tight_layout()
    files["losses"] = "training_losses.png"
    fig.savefig(output / files["losses"], dpi=_SAVE_DPI)
    plt.close(fig)

    classifier_time = [record["timing"]["classifier_training_sec"] for record in records]
    auxiliary_time = [
        record["timing"].get("comal_training_sec", record["timing"].get("modis_probe_training_sec", 0.0))
        for record in records
    ]
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(rounds, classifier_time, label="classifier", color=COLORS[0])
    axis.bar(rounds, auxiliary_time, bottom=classifier_time, label="acquisition auxiliary", color=COLORS[1])
    axis.set(xlabel="Round", ylabel="Wall seconds", title="Training time by round")
    axis.legend()
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    files["timing"] = "round_timing.png"
    fig.savefig(output / files["timing"], dpi=_SAVE_DPI)
    plt.close(fig)

    test_labels = prediction["test_labels"]
    test_probs = prediction["test_probabilities"]
    top_30 = np.argsort(-test_probs, axis=1, kind="stable")[:, :30]
    hits = np.take_along_axis(test_labels, top_30, axis=1).sum(axis=1)
    visit_recall = hits / test_labels.sum(axis=1)
    fig, axis = plt.subplots(figsize=(9, 5))
    axis.hist(visit_recall, bins=40, range=(0, 1), color=COLORS[2], edgecolor="white")
    axis.set(xlabel="Per-visit test Recall@30", ylabel="ICU visits", xlim=(0, 1))
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    files["per_visit"] = "test_per_visit_recall_at_30.png"
    fig.savefig(output / files["per_visit"], dpi=_SAVE_DPI)
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
        x = [record["labeled_count"] for record in state["records"]]
        y = [record["test_metrics"]["recall_at_30"] for record in state["records"]]
        area = float(np.trapezoid(y, x) / max(x[-1] - x[0], 1)) if len(x) > 1 else y[0]
        curves[name] = {"labeled": x, "test_recall_at_30": y, "normalised_area": area}
        axis.plot(x, y, marker="o", label=name, color=COLORS[position % len(COLORS)])
    axis.set(xlabel="Labeled ICU visits", ylabel="Test Recall@30", title="Active-learning comparison")
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
