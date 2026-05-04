"""Prompt construction helpers."""

from __future__ import annotations

from typing import Any


DEFAULT_BASE_PROMPT = "The quick brown fox jumps over the lazy dog. "


def build_prompt(prompt_len: int, base_prompt: str = DEFAULT_BASE_PROMPT) -> str:
    """Build a repeated stable prompt long enough for token truncation."""
    repeat = prompt_len // 10 + 1
    return base_prompt * repeat


def build_prompt_inputs(
    tokenizer: Any,
    prompt_len: int,
    device: Any,
    base_prompt: str = DEFAULT_BASE_PROMPT,
) -> Any:
    """Tokenize a fixed prompt and truncate it to the requested length."""
    prompt = build_prompt(prompt_len, base_prompt)
    return tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=prompt_len,
    ).to(device)
