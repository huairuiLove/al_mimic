"""Leakage-safe text feature extraction and memory-mapped feature cache."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .data import MIMICRecord, load_records


def _fingerprint(records: list[MIMICRecord], cfg: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(f"{record.row_index}:{record.hadm_id}:{record.split}\n".encode())
    digest.update(json.dumps(cfg, sort_keys=True).encode())
    return digest.hexdigest()


def _normalise(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values.astype(np.float32), axis=1, keepdims=True)
    return (values.astype(np.float32) / np.maximum(norms, 1e-12)).astype(np.float16)


def _tfidf_features(records: list[MIMICRecord], cfg: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    train_texts = [record.text for record in records if record.split == "train"]
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
    svd = TruncatedSVD(n_components=n_components, random_state=int(cfg.get("seed", 17)), n_iter=5)
    train_features = svd.fit_transform(train_matrix)
    all_features = np.empty((len(records), n_components), dtype=np.float32)
    train_indices = [index for index, record in enumerate(records) if record.split == "train"]
    train_index_set = set(train_indices)
    all_features[train_indices] = train_features
    for start in range(0, len(records), int(cfg.get("transform_batch_size", 2048))):
        indices = list(range(start, min(start + int(cfg.get("transform_batch_size", 2048)), len(records))))
        non_train = [index for index in indices if index not in train_index_set]
        if non_train:
            matrix = vectorizer.transform([records[index].text for index in non_train])
            all_features[non_train] = svd.transform(matrix)
    return _normalise(all_features), {
        "encoder": "tfidf_svd",
        "dimension": n_components,
        "vocabulary_size": len(vectorizer.vocabulary_),
        "explained_variance": float(svd.explained_variance_ratio_.sum()),
        "vectorizer": vectorizer,
        "svd": svd,
    }


def _bert_features(records: list[MIMICRecord], cfg: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    from concurrent.futures import ThreadPoolExecutor

    from transformers import AutoModel, AutoTokenizer

    model_path = str(cfg.get("model_path", "CoMAL-main/bert/bert-base-uncased"))
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=bool(cfg.get("local_files_only", True))
    )
    model = AutoModel.from_pretrained(model_path, local_files_only=bool(cfg.get("local_files_only", True)))
    device = torch.device(str(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")))
    model.to(device).eval()
    if device.type == "cuda":
        # Channels-last is a no-op for BERT, but mark the model for cudnn autotune paths.
        torch.backends.cudnn.benchmark = True
    batch_size = int(cfg.get("batch_size", 128))
    max_length = int(cfg.get("max_length", 256))
    precision = str(cfg.get("precision", "bf16")).lower()
    if device.type != "cuda":
        precision = "fp32"
    autocast = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16 if precision == "bf16" else torch.float16)
        if precision != "fp32"
        else torch.autocast(device_type="cpu", enabled=False)
    )

    def tokenize(start: int) -> dict[str, torch.Tensor]:
        texts = [record.text for record in records[start : start + batch_size]]
        return tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=max_length)

    starts = list(range(0, len(records), batch_size))
    features: list[np.ndarray] = []
    # Overlap CPU tokenization with GPU encode so both stay saturated.
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(tokenize, starts[0]) if starts else None
        for offset, start in enumerate(tqdm(starts, desc="encode MIMIC notes")):
            batch = pending.result() if pending is not None else tokenize(start)
            if offset + 1 < len(starts):
                pending = pool.submit(tokenize, starts[offset + 1])
            else:
                pending = None
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with torch.inference_mode(), autocast:
                output = model(**batch)
                hidden = getattr(output, "pooler_output", None)
                if hidden is None:
                    hidden = output.last_hidden_state[:, 0]
                hidden = F.normalize(hidden.float(), dim=1)
            # Async D2H via pinned staging keeps the next encode from waiting on numpy conversion.
            host = torch.empty_like(hidden, device="cpu", pin_memory=device.type == "cuda")
            host.copy_(hidden, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.current_stream().synchronize()
            features.append(host.numpy())
    values = np.concatenate(features, axis=0) if features else np.empty((0, 768), dtype=np.float32)
    return values.astype(np.float16), {
        "encoder": "bert",
        "dimension": int(values.shape[1]),
        "model_path": model_path,
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
    fingerprint = _fingerprint(records, feature_cfg)
    metadata_path = output / "metadata.json"
    feature_path = output / "features.npy"
    if metadata_path.is_file() and feature_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("fingerprint") == fingerprint:
            return metadata | {"output_dir": str(output), "cached": True}
    encoder = str(feature_cfg.get("encoder", "tfidf")).lower()
    if encoder in {"tfidf", "tfidf_svd", "svd"}:
        values, extra = _tfidf_features(records, feature_cfg)
        # Persist sklearn objects separately; metadata remains JSON serializable.
        joblib.dump(
            {"vectorizer": extra.pop("vectorizer"), "svd": extra.pop("svd")},
            output / "tfidf.joblib",
            compress=3,
        )
    elif encoder in {"bert", "bert-base-uncased"}:
        values, extra = _bert_features(records, feature_cfg)
    else:
        raise ValueError("features.encoder must be tfidf or bert")
    np.save(feature_path, values)
    metadata = {
        "format_version": 1,
        "fingerprint": fingerprint,
        "records": len(records),
        "dtype": str(values.dtype),
        "output_dir": str(output),
        **extra,
    }
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
    values = np.load(path, mmap_mode="r")
    if expected_records is not None and values.shape[0] != expected_records:
        raise ValueError(f"feature rows={values.shape[0]} but records={expected_records}")
    metadata_path = feature_dir / "metadata.json"
    if expected_encoder and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        actual = str(metadata.get("encoder", ""))
        expected = "tfidf_svd" if expected_encoder.lower() in {"tfidf", "tfidf_svd", "svd"} else "bert"
        if actual != expected:
            raise ValueError(
                f"feature cache encoder={actual!r}, expected {expected!r}; run `features` with the selected config"
            )
    return values
