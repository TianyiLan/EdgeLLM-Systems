"""CUDA memory metric helpers."""

from __future__ import annotations

import torch


def get_gpu_memory_mb() -> float:
    """Return currently allocated CUDA memory in MB."""
    # This is PyTorch allocated memory, not total GPU process memory.
    return torch.cuda.memory_allocated() / 1024**2


def get_peak_gpu_memory_mb() -> float:
    """Return PyTorch peak allocated CUDA memory in MB."""
    # The caller should reset peak stats before each measured run.
    return torch.cuda.max_memory_allocated() / 1024**2
