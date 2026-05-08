"""Model inspection helpers for experiment logs."""

from __future__ import annotations

from typing import Any

import torch


def count_parameters(model: Any) -> int:
    """Return the number of parameters visible on the loaded model."""
    return sum(param.numel() for param in model.parameters())


def parameter_size_mb(model: Any) -> float:
    """Return parameter payload size in MB for the loaded dtype/device layout."""
    total_bytes = sum(param.numel() * param.element_size() for param in model.parameters())
    return total_bytes / 1024**2


def dtype_name(dtype: Any) -> str:
    """Return a concise dtype name for CSV/log output."""
    if dtype == torch.float16:
        return "fp16"
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.float32:
        return "fp32"
    return str(dtype).replace("torch.", "")


def inspect_causal_lm(model: Any) -> dict[str, int | float | str]:
    """Collect model facts that should be recorded with benchmark results."""
    cfg = model.config
    hidden_size = int(getattr(cfg, "hidden_size", 0))
    attention_heads = int(getattr(cfg, "num_attention_heads", 0))
    kv_heads = int(getattr(cfg, "num_key_value_heads", attention_heads))
    head_dim = int(getattr(cfg, "head_dim", hidden_size // attention_heads if attention_heads else 0))
    layers = int(getattr(cfg, "num_hidden_layers", 0))

    first_param = next(model.parameters())
    param_count = count_parameters(model)
    param_mb = parameter_size_mb(model)

    kv_bytes_per_token = 2 * layers * kv_heads * head_dim * first_param.element_size()
    kv_mb_per_1k_tokens = kv_bytes_per_token * 1000 / 1024**2

    return {
        "model_type": str(getattr(cfg, "model_type", "unknown")),
        "layers": layers,
        "hidden_size": hidden_size,
        "attention_heads": attention_heads,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "torch_dtype": dtype_name(first_param.dtype),
        "parameter_count_b": round(param_count / 1e9, 3),
        "parameter_size_mb": round(param_mb, 1),
        "kv_mb_per_1k_tokens": round(kv_mb_per_1k_tokens, 2),
    }
