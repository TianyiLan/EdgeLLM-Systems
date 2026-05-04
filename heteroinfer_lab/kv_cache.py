"""KV cache sizing utilities."""

from __future__ import annotations

from typing import Any

import torch


def estimate_kv_cache_mb(model: Any, seq_len: int, batch_size: int = 1) -> float:
    """Estimate KV cache payload size from model config."""
    cfg = model.config
    num_layers = cfg.num_hidden_layers
    num_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    bytes_per_elem = 2

    total_bytes = (
        2
        * num_layers
        * num_kv_heads
        * head_dim
        * seq_len
        * batch_size
        * bytes_per_elem
    )
    return total_bytes / 1024**2


def kv_cache_size_from_past_key_values_mb(past_key_values: Any) -> float:
    """Measure pure KV cache payload by walking returned K/V tensors.

    This intentionally measures only tensor payload in ``past_key_values``.
    It does not include CUDA allocator state, temporary activations, logits,
    attention workspaces, or reserved memory.
    """
    candidates = [past_key_values]
    if hasattr(past_key_values, "to_legacy_cache"):
        try:
            candidates.append(past_key_values.to_legacy_cache())
        except Exception:
            pass

    seen_objects: set[int] = set()
    seen_tensors: set[int] = set()

    def walk(obj: Any) -> int:
        if obj is None:
            return 0

        obj_id = id(obj)
        if obj_id in seen_objects:
            return 0
        seen_objects.add(obj_id)

        if torch.is_tensor(obj):
            tensor_id = id(obj)
            if tensor_id in seen_tensors:
                return 0
            seen_tensors.add(tensor_id)
            return obj.numel() * obj.element_size()

        if isinstance(obj, dict):
            return sum(walk(v) for v in obj.values())
        if isinstance(obj, (list, tuple)):
            return sum(walk(v) for v in obj)
        if isinstance(obj, (str, bytes, int, float, bool)):
            return 0

        total = 0
        for name in (
            "key_cache",
            "value_cache",
            "keys",
            "values",
            "k_cache",
            "v_cache",
            "cache",
            "caches",
            "layers",
        ):
            if hasattr(obj, name):
                try:
                    total += walk(getattr(obj, name))
                except Exception:
                    pass

        if total == 0 and hasattr(obj, "__dict__"):
            total += walk(vars(obj))
        return total

    total_bytes = sum(walk(candidate) for candidate in candidates)
    if total_bytes == 0:
        cache_type = type(past_key_values).__name__
        public_attrs = [a for a in dir(past_key_values) if not a.startswith("_")][:30]
        raise ValueError(
            "No KV cache tensor was found in past_key_values; "
            f"cache_type={cache_type}, attrs={public_attrs}"
        )
    return total_bytes / 1024**2
