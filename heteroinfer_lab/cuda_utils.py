"""CUDA lifecycle helpers used by profiling benchmarks."""

from __future__ import annotations

import gc
from typing import Any

import torch


def require_cuda() -> None:
    """Raise a clear error when a benchmark is launched without CUDA."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; this benchmark expects a GPU runtime.")


def synchronize_if_cuda() -> None:
    """Synchronize CUDA work if CUDA is available."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def reset_peak_memory_stats() -> None:
    """Reset PyTorch peak CUDA memory counters."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def cleanup_cuda(*objects: Any) -> None:
    """Delete references owned by the caller and release cached CUDA blocks."""
    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
