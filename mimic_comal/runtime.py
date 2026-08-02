"""Runtime controls for A800 GPUs and cgroup-limited high-core hosts."""

from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import Any

import torch


def effective_cpu_count() -> int:
    """Return the usable CPU budget, preferring cgroup quota over raw topology."""
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
    """Pin BLAS/OpenMP pools to the cgroup budget to avoid oversubscription thrash."""
    budget = max(1, int(cpu_budget or effective_cpu_count()))
    # Leave one logical slot for the Python coordinator when many workers exist.
    math_threads = max(1, budget)
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        os.environ[key] = str(math_threads)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # Keep the CUDA default connection pool; forcing 1 blocks copy/compute overlap.
    os.environ.pop("CUDA_DEVICE_MAX_CONNECTIONS", None)
    return {"cpu_budget": budget, "math_threads": math_threads}


def configure_runtime(config: dict[str, Any]) -> None:
    training = config.get("training", {})
    budget = effective_cpu_count()
    apply_host_thread_env(budget)

    cpu_threads = int(training.get("cpu_threads", 0))
    if cpu_threads <= 0:
        cpu_threads = budget
    else:
        cpu_threads = min(cpu_threads, budget)
    interop_threads = int(training.get("interop_threads", max(1, min(4, budget // 4 or 1))))
    interop_threads = max(1, min(interop_threads, cpu_threads))

    torch.set_num_threads(cpu_threads)
    try:
        torch.set_num_interop_threads(interop_threads)
    except RuntimeError:
        pass

    allow_tf32 = bool(training.get("allow_tf32", True))
    try:
        torch.backends.opt_einsum.enabled = True
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
        torch.backends.cudnn.benchmark = bool(training.get("cudnn_benchmark", True))
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        if hasattr(torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction"):
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")
        # Prefer max clocks when the driver exposes persistence/boost paths.
        try:
            torch.cuda.set_per_process_memory_fraction(1.0)
        except Exception:
            pass
        # Enable Flash/mem-efficient SDPA kernels when available (BERT encode path).
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
        except Exception:
            pass


def hardware_report(config: dict[str, Any]) -> dict[str, Any]:
    training = config.get("training", {})
    budget = effective_cpu_count()
    report: dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "logical_cpu_count": os.cpu_count(),
        "effective_cpu_count": budget,
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "configured_cpu_threads": int(training.get("cpu_threads", 0)),
        "configured_workers": int(training.get("num_workers", 0)),
        "gpu_resident_features": bool(training.get("gpu_resident_features", True)),
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
    }
    if hasattr(os, "sched_getaffinity"):
        report["cpu_affinity_count"] = len(os.sched_getaffinity(0))
    if torch.cuda.is_available():
        devices = []
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_gib": round(properties.total_memory / 1024**3, 2),
                    "compute_capability": f"{properties.major}.{properties.minor}",
                    "multiprocessors": properties.multi_processor_count,
                }
            )
        report["cuda_devices"] = devices
        report["tf32"] = bool(torch.backends.cuda.matmul.allow_tf32)
        report["cudnn_benchmark"] = bool(torch.backends.cudnn.benchmark)
    return report
