"""Runtime controls for accelerators and cgroup-limited hosts."""

from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import Any


def effective_cpu_count() -> int:
    for path in (Path("/sys/fs/cgroup/cpu.max"), Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")):
        try:
            if path.name == "cpu.max":
                text = path.read_text(encoding="utf-8").strip()
                if text and text != "max":
                    quota_text, period_text = text.split()
                    quota, period = int(quota_text), int(period_text)
                    if quota > 0 and period > 0:
                        return max(1, quota // period)
            else:
                quota = int(path.read_text(encoding="utf-8").strip())
                period = int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text(encoding="utf-8").strip())
                if quota > 0 and period > 0:
                    return max(1, quota // period)
        except (FileNotFoundError, OSError, ValueError):
            continue
    if hasattr(os, "sched_getaffinity"):
        try:
            return max(1, len(os.sched_getaffinity(0)))
        except OSError:
            pass
    return max(1, os.cpu_count() or 1)


def apply_host_thread_env(cpu_budget: int | None = None) -> dict[str, int]:
    budget = max(1, int(cpu_budget or effective_cpu_count()))
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ[key] = str(budget)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return {"cpu_budget": budget, "math_threads": budget}


def configure_runtime(config: dict[str, Any]) -> None:
    import torch

    training = config.get("training", {})
    budget = effective_cpu_count()
    apply_host_thread_env(budget)
    cpu_threads = int(training.get("cpu_threads", 0)) or budget
    cpu_threads = max(1, min(cpu_threads, budget))
    interop = int(training.get("interop_threads", max(1, min(4, budget))))
    torch.set_num_threads(cpu_threads)
    try:
        torch.set_num_interop_threads(max(1, min(interop, cpu_threads)))
    except RuntimeError:
        pass
    allow_tf32 = bool(training.get("allow_tf32", True))
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
        torch.backends.cudnn.benchmark = bool(training.get("cudnn_benchmark", True))
        torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")


def hardware_report(config: dict[str, Any] | None = None) -> dict[str, Any]:
    import torch

    training = (config or {}).get("training", {})
    report: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "logical_cpu_count": os.cpu_count(),
        "effective_cpu_count": effective_cpu_count(),
        "configured_workers": int(training.get("num_workers", 0)),
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
    }
    if torch.cuda.is_available():
        report["cuda_devices"] = [
            {
                "index": index,
                "name": torch.cuda.get_device_properties(index).name,
                "total_memory_gib": round(torch.cuda.get_device_properties(index).total_memory / 1024**3, 2),
            }
            for index in range(torch.cuda.device_count())
        ]
    return report


__all__ = [
    "apply_host_thread_env",
    "configure_runtime",
    "effective_cpu_count",
    "hardware_report",
]
