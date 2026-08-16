"""Full BRSET v1.0.2 audit, patient split, metadata, and image loading."""

from __future__ import annotations

import csv
import hashlib
import json
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .config import data_paths

LABEL_COLUMNS = (
    "diabetic_retinopathy",
    "macular_edema",
    "scar",
    "nevus",
    "amd",
    "vascular_occlusion",
    "hypertensive_retinopathy",
    "drusen",
    "hemorrhage",
    "retinal_detachment",
    "myopic_fundus",
    "increased_cup_disc",
    "other",
)
NUMERIC_FIELDS = ("patient_age", "diabetes_time_y")
CATEGORICAL_FIELDS = ("camera", "insulin", "patient_sex", "exam_eye", "diabetes")
MISSING_TOKEN = "<missing>"
UNKNOWN_TOKEN = "<unknown>"
SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True, slots=True)
class BrsetDataAudit:
    images: int
    patients: int
    labels: int
    positive_labels: dict[str, int]
    images_per_patient: dict[str, int]
    zero_label_images: int
    multilabel_images: int
    csv_sha256: str
    split_image_counts: dict[str, int] | None = None
    split_patient_counts: dict[str, int] | None = None
    metadata_dim: int | None = None


@dataclass(frozen=True, slots=True)
class MetadataSchema:
    numeric_means: dict[str, float]
    numeric_stds: dict[str, float]
    categorical_vocabularies: dict[str, list[str]]
    comorbidity_vocabulary: list[str]
    dimension: int

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> MetadataSchema:
        return cls(
            numeric_means={str(k): float(v) for k, v in value["numeric_means"].items()},
            numeric_stds={str(k): float(v) for k, v in value["numeric_stds"].items()},
            categorical_vocabularies={
                str(k): [str(item) for item in values]
                for k, values in value["categorical_vocabularies"].items()
            },
            comorbidity_vocabulary=[str(item) for item in value["comorbidity_vocabulary"]],
            dimension=int(value["dimension"]),
        )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("BRSET label file has no header")
        missing = [
            name for name in ("image_id", "patient_id", *LABEL_COLUMNS) if name not in reader.fieldnames
        ]
        if missing:
            raise ValueError(f"BRSET label file is missing columns: {missing}")
        return [dict(row) for row in reader]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_source(config: dict[str, Any]) -> BrsetDataAudit:
    paths = data_paths(config)
    rows = _read_rows(paths["labels_csv"])
    expected = config.get("preprocessing", {})
    image_ids = [row["image_id"].strip() for row in rows]
    patient_ids = [row["patient_id"].strip() for row in rows]
    if any(not value for value in image_ids) or any(not value for value in patient_ids):
        raise ValueError("BRSET image_id and patient_id must be non-empty")
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("BRSET image_id values must be unique")

    labels = np.asarray([[int(row[name]) for name in LABEL_COLUMNS] for row in rows], dtype=np.int8)
    if labels.shape != (len(rows), len(LABEL_COLUMNS)) or not np.isin(labels, (0, 1)).all():
        raise ValueError("BRSET disease labels must form a binary [N,13] matrix")

    disk_images = {path.stem for path in paths["images_dir"].glob("*.jpg")}
    csv_images = set(image_ids)
    missing_images = sorted(csv_images - disk_images)
    extra_images = sorted(disk_images - csv_images)
    if missing_images or extra_images:
        raise ValueError(f"BRSET CSV/image mismatch: missing={missing_images[:5]}, extra={extra_images[:5]}")

    actual = {
        "images": len(rows),
        "patients": len(set(patient_ids)),
        "labels": len(LABEL_COLUMNS),
    }
    required = {
        "images": int(expected.get("expected_images", 16266)),
        "patients": int(expected.get("expected_patients", 8524)),
        "labels": int(expected.get("expected_labels", 13)),
    }
    mismatches = [
        f"{name}: expected {required[name]}, got {actual[name]}"
        for name in required
        if actual[name] != required[name]
    ]
    if mismatches:
        raise ValueError("formal BRSET data mismatch: " + "; ".join(mismatches))

    image_counts = Counter(patient_ids)
    return BrsetDataAudit(
        images=len(rows),
        patients=len(image_counts),
        labels=len(LABEL_COLUMNS),
        positive_labels={name: int(labels[:, index].sum()) for index, name in enumerate(LABEL_COLUMNS)},
        images_per_patient={
            str(count): int(total) for count, total in sorted(Counter(image_counts.values()).items())
        },
        zero_label_images=int((labels.sum(axis=1) == 0).sum()),
        multilabel_images=int((labels.sum(axis=1) > 1).sum()),
        csv_sha256=_sha256(paths["labels_csv"]),
    )


def _patient_targets(rows: Sequence[dict[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    grouped: dict[str, np.ndarray] = {}
    for row in rows:
        patient = row["patient_id"].strip()
        labels = np.asarray([int(row[name]) for name in LABEL_COLUMNS], dtype=np.int8)
        grouped[patient] = np.maximum(
            grouped.get(patient, np.zeros(len(LABEL_COLUMNS), dtype=np.int8)), labels
        )
    patients = np.asarray(sorted(grouped), dtype=object)
    targets = np.stack([grouped[str(patient)] for patient in patients])
    return patients, targets


def patient_multilabel_split(rows: Sequence[dict[str, str]], *, seed: int) -> dict[str, str]:
    """Return a deterministic 60/20/20 split over patients, stratified by 13 labels."""
    try:
        from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
    except ImportError as exc:  # pragma: no cover - dependency error is environment-specific
        raise RuntimeError("iterative-stratification is required for the BRSET patient split") from exc

    patients, targets = _patient_targets(rows)
    first = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.40, random_state=seed)
    train_positions, holdout_positions = next(first.split(np.zeros((len(patients), 1)), targets))
    holdout_targets = targets[holdout_positions]
    second = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=seed + 1)
    validation_relative, test_relative = next(
        second.split(np.zeros((len(holdout_positions), 1)), holdout_targets)
    )
    validation_positions = holdout_positions[validation_relative]
    test_positions = holdout_positions[test_relative]
    mapping = {str(patients[index]): "train" for index in train_positions}
    mapping.update({str(patients[index]): "val" for index in validation_positions})
    mapping.update({str(patients[index]): "test" for index in test_positions})
    if len(mapping) != len(patients):
        raise RuntimeError("patient split did not assign every BRSET patient exactly once")
    return mapping


def _normalize_category(field: str, value: str) -> str:
    normalized = value.strip()
    return normalized.lower() if field in {"insulin", "diabetes"} else normalized


def _parse_numeric(value: str) -> float | None:
    normalized = unicodedata.normalize("NFKD", value.strip()).encode("ascii", "ignore").decode()
    if not normalized or normalized.lower() == "nao":
        return None
    if normalized.upper() == "1O":
        normalized = "10"
    try:
        return float(normalized.replace(",", "."))
    except ValueError as exc:
        raise ValueError(f"unsupported BRSET numeric value: {value!r}") from exc


def _comorbidity_tokens(value: str) -> list[str]:
    normalized = value.strip().lower()
    if not normalized or normalized == "0":
        return []
    return sorted(
        {token.strip() for token in normalized.split(",") if token.strip() and token.strip() != "0"}
    )


def fit_metadata_schema(rows: Sequence[dict[str, str]], *, minimum_comorbidity_count: int) -> MetadataSchema:
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for field in NUMERIC_FIELDS:
        observed = np.asarray(
            [number for row in rows if (number := _parse_numeric(row[field])) is not None],
            dtype=np.float64,
        )
        if not observed.size:
            raise ValueError(f"BRSET training split has no observed values for {field}")
        means[field] = float(observed.mean())
        stds[field] = float(observed.std()) or 1.0

    categorical: dict[str, list[str]] = {}
    for field in CATEGORICAL_FIELDS:
        observed = sorted({_normalize_category(field, row[field]) for row in rows if row[field].strip()})
        categorical[field] = [MISSING_TOKEN, UNKNOWN_TOKEN, *observed]

    counts: Counter[str] = Counter()
    for row in rows:
        counts.update(_comorbidity_tokens(row["comorbidities"]))
    comorbidities = sorted(
        token for token, count in counts.items() if count >= int(minimum_comorbidity_count)
    )
    dimension = 2 * len(NUMERIC_FIELDS)
    dimension += sum(len(values) for values in categorical.values())
    dimension += 1 + len(comorbidities)
    return MetadataSchema(means, stds, categorical, comorbidities, dimension)


def vectorize_metadata(row: dict[str, str], schema: MetadataSchema) -> np.ndarray:
    values: list[float] = []
    for field in NUMERIC_FIELDS:
        number = _parse_numeric(row[field])
        missing = number is None
        if number is None:
            number = schema.numeric_means[field]
        values.extend(
            (
                (number - schema.numeric_means[field]) / schema.numeric_stds[field],
                float(missing),
            )
        )
    for field in CATEGORICAL_FIELDS:
        vocabulary = schema.categorical_vocabularies[field]
        raw = row[field].strip()
        category = _normalize_category(field, raw) if raw else MISSING_TOKEN
        if category not in vocabulary:
            category = UNKNOWN_TOKEN
        values.extend(float(category == item) for item in vocabulary)
    tokens = set(_comorbidity_tokens(row["comorbidities"]))
    values.append(float(not tokens))
    values.extend(float(item in tokens) for item in schema.comorbidity_vocabulary)
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (schema.dimension,):
        raise RuntimeError(f"metadata vector has shape {result.shape}, expected {(schema.dimension,)}")
    return result


def prepare_data(config: dict[str, Any]) -> dict[str, Any]:
    paths = data_paths(config)
    source_audit = audit_source(config)
    rows = _read_rows(paths["labels_csv"])
    preprocessing = config.get("preprocessing", {})
    seed = int(training_seed(config))
    patient_splits = patient_multilabel_split(rows, seed=seed)
    split_rows = {name: [] for name in SPLIT_NAMES}
    for row in rows:
        split_rows[patient_splits[row["patient_id"].strip()]].append(row)

    split_label_counts = {
        split: np.asarray([[int(row[name]) for name in LABEL_COLUMNS] for row in values], dtype=np.int64).sum(
            axis=0
        )
        for split, values in split_rows.items()
    }
    absent = {
        split: [LABEL_COLUMNS[index] for index, count in enumerate(counts) if count == 0]
        for split, counts in split_label_counts.items()
        if np.any(counts == 0)
    }
    if absent:
        raise ValueError(f"patient split leaves disease labels absent from evaluation: {absent}")

    schema = fit_metadata_schema(
        split_rows["train"],
        minimum_comorbidity_count=int(preprocessing.get("minimum_comorbidity_count", 10)),
    )
    paths["prepared_dir"].mkdir(parents=True, exist_ok=True)
    temporary = paths["split_manifest"].with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("image_id", "patient_id", "split"))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "image_id": row["image_id"].strip(),
                    "patient_id": row["patient_id"].strip(),
                    "split": patient_splits[row["patient_id"].strip()],
                }
            )
    temporary.replace(paths["split_manifest"])
    _write_json(paths["metadata_schema"], asdict(schema))

    split_image_counts = {name: len(values) for name, values in split_rows.items()}
    split_patient_counts = {
        name: len({row["patient_id"].strip() for row in values}) for name, values in split_rows.items()
    }
    audit = BrsetDataAudit(
        **{
            key: value
            for key, value in asdict(source_audit).items()
            if key not in {"split_image_counts", "split_patient_counts", "metadata_dim"}
        },
        split_image_counts=split_image_counts,
        split_patient_counts=split_patient_counts,
        metadata_dim=schema.dimension,
    )
    audit_value = asdict(audit) | {
        "split_positive_labels": {
            split: {label: int(split_label_counts[split][index]) for index, label in enumerate(LABEL_COLUMNS)}
            for split in SPLIT_NAMES
        },
        "protocol": "BRSET v1.0.2 patient-level multilabel-stratified 60/20/20",
        "seed": seed,
    }
    _write_json(paths["audit"], audit_value)
    return audit_value


def training_seed(config: dict[str, Any]) -> int:
    return int(config.get("training", {}).get("seed", 17))


def audit_prepared(config: dict[str, Any]) -> BrsetDataAudit:
    paths = data_paths(config, require_prepared=True)
    source = audit_source(config)
    audit_value = json.loads(paths["audit"].read_text(encoding="utf-8"))
    if audit_value.get("csv_sha256") != source.csv_sha256:
        raise ValueError("BRSET source CSV changed after preparation; rerun prepare")
    schema = MetadataSchema.from_dict(json.loads(paths["metadata_schema"].read_text(encoding="utf-8")))
    with paths["split_manifest"].open(newline="", encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle))
    if len(manifest) != source.images:
        raise ValueError("BRSET split manifest does not contain every image")
    patient_splits: dict[str, set[str]] = defaultdict(set)
    split_images = Counter()
    for row in manifest:
        split = row.get("split", "")
        if split not in SPLIT_NAMES:
            raise ValueError(f"invalid BRSET split name: {split!r}")
        patient_splits[row["patient_id"]].add(split)
        split_images[split] += 1
    leaking = [patient for patient, splits in patient_splits.items() if len(splits) != 1]
    if leaking:
        raise ValueError(f"BRSET patients cross data splits: {leaking[:5]}")
    split_patients = Counter(next(iter(splits)) for splits in patient_splits.values())
    return BrsetDataAudit(
        **{
            key: value
            for key, value in asdict(source).items()
            if key not in {"split_image_counts", "split_patient_counts", "metadata_dim"}
        },
        split_image_counts={name: int(split_images[name]) for name in SPLIT_NAMES},
        split_patient_counts={name: int(split_patients[name]) for name in SPLIT_NAMES},
        metadata_dim=schema.dimension,
    )


class BrsetDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, store: BrsetFeatureStore, indices: Sequence[int], *, train: bool) -> None:
        self.store = store
        self.indices = np.asarray(indices, dtype=np.int64)
        self.transform = store.image_transform(train=train)

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, position: int) -> dict[str, torch.Tensor]:
        index = int(self.indices[position])
        with Image.open(self.store.image_paths[index]) as image:
            pixels = self.transform(image.convert("RGB"))
        return {
            "index": torch.tensor(index, dtype=torch.long),
            "image": pixels,
            "metadata": torch.from_numpy(self.store.metadata[index]),
            "labels": torch.from_numpy(self.store.labels[index]),
        }


class BrsetFeatureStore:
    def __init__(self, config: dict[str, Any], *, validate: bool = True) -> None:
        self.config = config
        self.paths = data_paths(config, require_prepared=True)
        self.audit = audit_prepared(config) if validate else audit_source(config)
        rows = _read_rows(self.paths["labels_csv"])
        with self.paths["split_manifest"].open(newline="", encoding="utf-8") as handle:
            manifest = {row["image_id"]: row for row in csv.DictReader(handle)}
        self.schema = MetadataSchema.from_dict(
            json.loads(self.paths["metadata_schema"].read_text(encoding="utf-8"))
        )
        if set(manifest) != {row["image_id"].strip() for row in rows}:
            raise ValueError("BRSET manifest image IDs do not match the source CSV")
        self.image_ids = np.asarray([row["image_id"].strip() for row in rows], dtype=object)
        self.patient_ids_array = np.asarray([row["patient_id"].strip() for row in rows], dtype=object)
        self.split_array = np.asarray(
            [manifest[str(image)]["split"] for image in self.image_ids], dtype=object
        )
        self.image_paths = [self.paths["images_dir"] / f"{image_id}.jpg" for image_id in self.image_ids]
        self.labels = np.asarray(
            [[float(row[name]) for name in LABEL_COLUMNS] for row in rows], dtype=np.float32
        )
        self.metadata = np.stack([vectorize_metadata(row, self.schema) for row in rows])
        self._patient_indices: dict[str, np.ndarray] = {
            patient: np.flatnonzero(self.patient_ids_array == patient).astype(np.int64)
            for patient in sorted(set(self.patient_ids_array))
        }

    def indices(self, split: str) -> np.ndarray:
        if split not in SPLIT_NAMES:
            raise ValueError(f"unknown BRSET split: {split}")
        return np.flatnonzero(self.split_array == split).astype(np.int64)

    def patient_ids(self, split: str) -> list[str]:
        return sorted({str(self.patient_ids_array[index]) for index in self.indices(split)})

    def indices_for_patients(self, patient_ids: Iterable[str]) -> np.ndarray:
        values = [self._patient_indices[str(patient)] for patient in patient_ids]
        if not values:
            return np.empty(0, dtype=np.int64)
        return np.sort(np.concatenate(values)).astype(np.int64)

    def patient_targets(self, patient_ids: Sequence[str]) -> np.ndarray:
        return np.stack(
            [self.labels[self._patient_indices[str(patient)]].max(axis=0) for patient in patient_ids]
        )

    def image_transform(self, *, train: bool):
        try:
            from torchvision import transforms
            from torchvision.models import ResNet50_Weights
        except ImportError as exc:  # pragma: no cover - dependency error is environment-specific
            raise RuntimeError("torchvision is required for BRSET image preprocessing") from exc
        preprocessing = self.config.get("preprocessing", {})
        resize = int(preprocessing.get("resize_size", 256))
        crop = int(preprocessing.get("image_size", 224))
        weights_transform = ResNet50_Weights.IMAGENET1K_V2.transforms()
        normalize = transforms.Normalize(
            mean=weights_transform.mean,
            std=weights_transform.std,
        )
        if train:
            return transforms.Compose(
                [
                    transforms.Resize(resize),
                    transforms.RandomResizedCrop(crop, scale=(0.80, 1.0)),
                    transforms.RandomRotation(30),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        return transforms.Compose(
            [transforms.Resize(resize), transforms.CenterCrop(crop), transforms.ToTensor(), normalize]
        )

    def make_loader(
        self,
        indices: Sequence[int],
        *,
        train: bool,
        batch_size: int,
        shuffle: bool,
    ) -> DataLoader:
        training = self.config.get("training", {})
        return DataLoader(
            BrsetDataset(self, indices, train=train),
            batch_size=int(batch_size),
            shuffle=bool(shuffle),
            num_workers=int(training.get("num_workers", 8)),
            pin_memory=bool(training.get("pin_memory", True)),
            persistent_workers=bool(training.get("num_workers", 8)),
            drop_last=False,
        )
