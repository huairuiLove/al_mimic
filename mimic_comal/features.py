"""Leakage-safe scratch text and structured multimodal feature cache."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from .data import MIMICRecord, load_records
from .config import require_multimodal_paths
from .multimodal import build_structured_modalities


def _fingerprint(records: list[MIMICRecord], cfg: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    update = digest.update
    # Batch the membership string to cut Python call overhead on 50k+ records.
    chunk: list[str] = []
    for index, record in enumerate(records):
        text_digest = hashlib.sha256(record.text.encode("utf-8")).hexdigest()
        chunk.append(
            f"{record.row_index}:{record.hadm_id}:{record.split}:{','.join(record.labels)}:{text_digest}\n"
        )
        if (index & 1023) == 1023:
            update("".join(chunk).encode())
            chunk.clear()
    if chunk:
        update("".join(chunk).encode())
    try:
        import orjson

        update(orjson.dumps(cfg, option=orjson.OPT_SORT_KEYS))
    except ImportError:  # pragma: no cover
        update(json.dumps(cfg, sort_keys=True).encode())
    return digest.hexdigest()


def _normalise(values: np.ndarray) -> np.ndarray:
    values32 = values.astype(np.float32, copy=False)
    norms = np.linalg.norm(values32, axis=1, keepdims=True)
    np.maximum(norms, 1e-12, out=norms)
    return (values32 / norms).astype(np.float16)


def _tfidf_features(records: list[MIMICRecord], cfg: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    train_mask = np.fromiter((record.split == "train" for record in records), dtype=bool, count=len(records))
    texts = [record.text for record in records]
    train_texts = [text for text, keep in zip(texts, train_mask) if keep]
    if not train_texts:
        raise RuntimeError("training split is empty")
    vectorizer = TfidfVectorizer(
        max_features=int(cfg.get("max_features", 100_000)),
        min_df=int(cfg.get("min_df", 2)),
        max_df=float(cfg.get("max_df", 0.995)),
        ngram_range=tuple(cfg.get("ngram_range", [1, 2])),
        sublinear_tf=True,
        dtype=np.float32,
        strip_accents="unicode",
    )
    train_matrix = vectorizer.fit_transform(train_texts)
    requested = int(cfg.get("dimension", 256))
    n_components = max(1, min(requested, train_matrix.shape[0] - 1, train_matrix.shape[1] - 1))
    # More power iterations improve SVD quality with negligible wall cost vs. text scan.
    svd = TruncatedSVD(
        n_components=n_components,
        random_state=int(cfg.get("seed", 17)),
        n_iter=int(cfg.get("svd_n_iter", 7)),
    )
    train_features = svd.fit_transform(train_matrix)
    all_features = np.empty((len(records), n_components), dtype=np.float32)
    train_indices = np.flatnonzero(train_mask)
    all_features[train_indices] = train_features
    non_train_indices = np.flatnonzero(~train_mask)
    if non_train_indices.size:
        batch = int(cfg.get("transform_batch_size", 8192))
        for start in range(0, int(non_train_indices.size), batch):
            selected = non_train_indices[start : start + batch]
            matrix = vectorizer.transform([texts[int(index)] for index in selected])
            all_features[selected] = svd.transform(matrix)
    return _normalise(all_features), {
        "encoder": "tfidf_svd",
        "dimension": n_components,
        "vocabulary_size": len(vectorizer.vocabulary_),
        "explained_variance": float(svd.explained_variance_ratio_.sum()),
        "vectorizer": vectorizer,
        "svd": svd,
    }


def _multimodal_features(
    records: list[MIMICRecord], config: dict[str, Any], cfg: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    text, text_metadata = _tfidf_features(records, cfg)
    paths = require_multimodal_paths(config)
    measurements, demographics, structured_metadata = build_structured_modalities(records, paths, cfg)
    text_stop = int(text.shape[1])
    measurement_stop = text_stop + int(measurements.shape[1])
    values = np.concatenate((text.astype(np.float32), measurements, demographics), axis=1).astype(np.float16)
    return values, {
        "encoder": "multimodal_scratch",
        "dimension": int(values.shape[1]),
        "initialization": "random",
        "pretrained_weights": False,
        "modalities": [
            {"name": "clinical_note", "start": 0, "stop": text_stop, "shape": [text_stop]},
            {
                "name": "icu_measurements",
                "start": text_stop,
                "stop": measurement_stop,
                "shape": structured_metadata["measurement_shape"],
            },
            {
                "name": "demographics",
                "start": measurement_stop,
                "stop": int(values.shape[1]),
                "shape": [int(demographics.shape[1])],
            },
        ],
        "text": {
            "dimension": text_stop,
            "vocabulary_size": text_metadata["vocabulary_size"],
            "explained_variance": text_metadata["explained_variance"],
        },
        "structured": structured_metadata,
        "vectorizer": text_metadata["vectorizer"],
        "svd": text_metadata["svd"],
    }


def build_features(
    config: dict[str, Any], prepared_dir: str | Path | None = None, output_dir: str | Path | None = None
) -> dict[str, Any]:
    dataset_cfg = config.get("dataset", {})
    prepared = Path(prepared_dir or dataset_cfg.get("prepared_dir", "prepared/mimic_iii"))
    output = Path(output_dir or dataset_cfg.get("feature_dir", prepared / "features"))
    output.mkdir(parents=True, exist_ok=True)
    records = load_records(prepared)
    feature_cfg = config.get("features", {})
    fingerprint_cfg = dict(feature_cfg)
    if str(feature_cfg.get("encoder", "multimodal_scratch")).lower() == "multimodal_scratch":
        for name, path in require_multimodal_paths(config).items():
            if name != "root":
                stat = path.stat()
                fingerprint_cfg[f"source_{name}"] = [str(path.resolve()), stat.st_size, stat.st_mtime_ns]
    fingerprint = _fingerprint(records, fingerprint_cfg)
    metadata_path = output / "metadata.json"
    feature_path = output / "features.npy"
    if metadata_path.is_file() and feature_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("fingerprint") == fingerprint:
            return metadata | {"output_dir": str(output), "cached": True}
    encoder = str(feature_cfg.get("encoder", "multimodal_scratch")).lower()
    if encoder in {"tfidf", "tfidf_svd", "svd"}:
        values, extra = _tfidf_features(records, feature_cfg)
        # Persist sklearn objects separately; metadata remains JSON serializable.
        joblib.dump(
            {"vectorizer": extra.pop("vectorizer"), "svd": extra.pop("svd")},
            output / "tfidf.joblib",
            compress=int(feature_cfg.get("joblib_compress", 1)),
        )
    elif encoder == "multimodal_scratch":
        values, extra = _multimodal_features(records, config, feature_cfg)
        joblib.dump(
            {"vectorizer": extra.pop("vectorizer"), "svd": extra.pop("svd")},
            output / "text_tfidf.joblib",
            compress=int(feature_cfg.get("joblib_compress", 1)),
        )
    else:
        raise ValueError("features.encoder must be multimodal_scratch or tfidf")
    np.save(feature_path, values)
    metadata = {
        "format_version": 2,
        "fingerprint": fingerprint,
        "records": len(records),
        "dtype": str(values.dtype),
        "output_dir": str(output),
        **extra,
    }
    try:
        import orjson

        metadata_path.write_bytes(orjson.dumps(metadata, option=orjson.OPT_INDENT_2) + b"\n")
    except ImportError:  # pragma: no cover
        metadata_path.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return metadata | {"cached": False}


def load_features(
    feature_dir: str | Path,
    expected_records: int | None = None,
    expected_encoder: str | None = None,
) -> np.ndarray:
    feature_dir = Path(feature_dir)
    path = feature_dir / "features.npy"
    if not path.is_file():
        raise FileNotFoundError(f"feature cache not found: {path}; run `features` first")
    # Prefer writable RAM copy on hosts with large memory; mmap stays available as fallback.
    try:
        values = np.load(path)
    except OSError:
        values = np.load(path, mmap_mode="r")
    if expected_records is not None and values.shape[0] != expected_records:
        raise ValueError(f"feature rows={values.shape[0]} but records={expected_records}")
    metadata_path = feature_dir / "metadata.json"
    if expected_encoder and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        actual = str(metadata.get("encoder", ""))
        expected = (
            "tfidf_svd" if expected_encoder.lower() in {"tfidf", "tfidf_svd", "svd"} else "multimodal_scratch"
        )
        if actual != expected:
            raise ValueError(
                f"feature cache encoder={actual!r}, expected {expected!r}; run `features` with the selected config"
            )
    return values
