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
    # CUDA kernels are asynchronous by default. Synchronization is required
    # before reading timers, otherwise Python may stop the clock too early.
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def reset_peak_memory_stats() -> None:
    """Reset PyTorch peak CUDA memory counters."""
    # Peak memory is measured per profiling run. Resetting here prevents the
    # previous configuration from contaminating the next configuration.
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def cleanup_cuda(*objects: Any) -> None:
    """Delete references owned by the caller and release cached CUDA blocks."""
    # The explicit del only removes this function's local references. Callers
    # should still delete large tensors in their own scope before calling this.
    for obj in objects:
        del obj
    # Python objects holding CUDA tensors may be released only after GC runs.
    gc.collect()
    if torch.cuda.is_available():
        # This returns unused cached blocks to the CUDA driver. It does not
        # reduce memory held by live tensors.
        torch.cuda.empty_cache()
