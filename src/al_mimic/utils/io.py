"""Small serialization helpers shared by task runners and evaluation."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


def jsonable(value: Any) -> Any:
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a base dependency
        torch = None
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): jsonable(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(child) for child in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def write_json(path: str | Path, value: Any) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = jsonable(value)
    try:
        import orjson
    except ImportError:  # pragma: no cover - optional acceleration
        output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    else:
        output.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2) + b"\n")
    return output


__all__ = ["jsonable", "write_json"]
