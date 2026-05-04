"""Model loading helpers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_causal_lm(
    model_id: str,
    model_dir: str | os.PathLike | None = None,
    torch_dtype: Any = torch.float16,
    device_map: str = "auto",
    local_files_only: bool = False,
) -> tuple[Any, Any, Path | str]:
    """Load tokenizer and causal LM, optionally caching under ``model_dir``."""
    model_path: Path | str = model_id

    if model_dir is not None:
        model_name = model_id.split("/")[-1]
        local_path = Path(model_dir) / model_name

        # Colab sessions are temporary, but Google Drive persists. Reusing a
        # Drive cache avoids downloading Gemma weights on every run.
        if local_path.is_dir() and any(local_path.iterdir()):
            model_path = local_path
        else:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                repo_id=model_id,
                local_dir=str(local_path),
                local_files_only=local_files_only,
            )
            model_path = local_path

    # Tokenizer and model must be loaded from the same path/revision.
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # device_map="auto" lets Accelerate place the model on the available GPU.
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    return tokenizer, model, model_path
