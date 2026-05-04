"""CUDA memory metric helpers."""

from __future__ import annotations

import torch


def get_gpu_memory_mb() -> float:
    """Return currently allocated CUDA memory in MB."""
    return torch.cuda.memory_allocated() / 1024**2


def get_peak_gpu_memory_mb() -> float:
    """Return PyTorch peak allocated CUDA memory in MB."""
    return torch.cuda.max_memory_allocated() / 1024**2
