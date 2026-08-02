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
from .runtime import effective_cpu_count


def _fingerprint(records: list[MIMICRecord], cfg: dict[str, Any]) -> str:
    # Keep sha256 so existing feature caches remain valid across adapter upgrades.
    digest = hashlib.sha256()
    update = digest.update
    for record in records:
        update(f"{record.row_index}:{record.hadm_id}:{record.split}\n".encode())
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


def _bert_features(records: list[MIMICRecord], cfg: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    from concurrent.futures import ThreadPoolExecutor

    from transformers import AutoModel, AutoTokenizer

    model_path = str(cfg.get("model_path", "CoMAL-main/bert/bert-base-uncased"))
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=bool(cfg.get("local_files_only", True))
    )
    model = AutoModel.from_pretrained(
        model_path,
        local_files_only=bool(cfg.get("local_files_only", True)),
        output_attentions=False,
        output_hidden_states=False,
    )
    device = torch.device(str(cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")))
    model.to(device).eval()
    model.requires_grad_(False)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        # Prefer cudnn SDPA / flash paths for encoder self-attention.
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        except Exception:
            pass
    batch_size = int(cfg.get("batch_size", 128))
    max_length = int(cfg.get("max_length", 256))
    precision = str(cfg.get("precision", "bf16")).lower()
    if device.type != "cuda":
        precision = "fp32"
    autocast_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    use_autocast = precision != "fp32" and device.type == "cuda"

    # Prefer the fast tokenizer backend and keep a deep CPU tokenize queue.
    tokenize_workers = max(1, min(2, effective_cpu_count() // 4 or 1))

    def tokenize(start: int) -> dict[str, torch.Tensor]:
        texts = [record.text for record in records[start : start + batch_size]]
        return tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
        )

    starts = list(range(0, len(records), batch_size))
    # Preallocate host buffer; avoids repeated concatenate + dtype casts.
    feature_dim = int(getattr(model.config, "hidden_size", 768))
    host_features = np.empty((len(records), feature_dim), dtype=np.float16)
    copy_stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None

    with ThreadPoolExecutor(max_workers=tokenize_workers) as pool:
        pending = [pool.submit(tokenize, start) for start in starts[:tokenize_workers]]
        next_submit = len(pending)
        for offset, start in enumerate(tqdm(starts, desc="encode MIMIC notes")):
            batch = pending[offset].result() if offset < len(pending) else tokenize(start)
            if next_submit < len(starts):
                pending.append(pool.submit(tokenize, starts[next_submit]))
                next_submit += 1
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with torch.inference_mode():
                if use_autocast:
                    with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                        output = model(**batch)
                else:
                    output = model(**batch)
                hidden = getattr(output, "pooler_output", None)
                if hidden is None:
                    hidden = output.last_hidden_state[:, 0]
                hidden = F.normalize(hidden.float(), dim=1)
            host = torch.empty((hidden.shape[0], feature_dim), dtype=torch.float16, pin_memory=device.type == "cuda")
            if copy_stream is not None:
                with torch.cuda.stream(copy_stream):
                    host.copy_(hidden.to(dtype=torch.float16), non_blocking=True)
                copy_stream.synchronize()
            else:
                host.copy_(hidden.to(dtype=torch.float16))
            stop = min(start + batch_size, len(records))
            host_features[start:stop] = host[: stop - start].numpy()
    return host_features, {
        "encoder": "bert",
        "dimension": feature_dim,
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
            compress=int(feature_cfg.get("joblib_compress", 1)),
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
        expected = "tfidf_svd" if expected_encoder.lower() in {"tfidf", "tfidf_svd", "svd"} else "bert"
        if actual != expected:
            raise ValueError(
                f"feature cache encoder={actual!r}, expected {expected!r}; run `features` with the selected config"
            )
    return values
