"""Run Experiment 001 Gemma baseline profiling with PKV KV cache metrics."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    # Allow running this file directly from the repo root or from Colab.
    sys.path.insert(0, str(ROOT))

from edge_llm_systems.cuda_utils import (  # noqa: E402
    cleanup_cuda,
    require_cuda,
    reset_peak_memory_stats,
    synchronize_if_cuda,
)
from edge_llm_systems.kv_cache import (  # noqa: E402
    estimate_kv_cache_mb,
    kv_cache_size_from_past_key_values_mb,
)
from edge_llm_systems.memory import get_peak_gpu_memory_mb  # noqa: E402
from edge_llm_systems.metrics import mean_metric, tokens_per_second  # noqa: E402
from edge_llm_systems.models import load_causal_lm  # noqa: E402
from edge_llm_systems.prompts import build_prompt_inputs  # noqa: E402


FIELDNAMES = [
    # Keep this order stable so result CSVs are easy to diff across runs.
    "prompt_len",
    "gen_len",
    "ttft_ms",
    "tpot_ms",
    "tokens_s",
    "peak_mem_mb",
    "kv_est_mb",
    "kv_pkv_prefill_mb",
    "kv_pkv_final_mb",
]


def load_config(path: Path) -> dict[str, Any]:
    """Load a YAML config via PyYAML."""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Please install pyyaml to read benchmark configs.") from exc

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dtype_from_config(name: str) -> Any:
    """Map config dtype string to torch dtype."""
    # Config files use strings because YAML cannot represent torch dtypes.
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported torch_dtype: {name}")
    return mapping[name]


def build_configs(config: dict[str, Any]) -> list[tuple[int, int]]:
    """Build the full profiling matrix without duplicate configs."""
    # The "orig" matrix reproduces short/medium prompts. The "ext" matrix adds
    # longer prompts for the same gen_len sweep.
    pairs = [
        (p, g)
        for p in config["prompt_lengths_orig"]
        for g in config["gen_lengths_orig"]
    ]
    pairs.extend(
        (p, g)
        for p in config["prompt_lengths_ext"]
        for g in config["gen_lengths_ext"]
    )

    # Avoid duplicate configs if a boundary prompt, such as 512, appears in both
    # the original and extended lists.
    seen = set()
    unique_pairs = []
    for pair in pairs:
        if pair not in seen:
            seen.add(pair)
            unique_pairs.append(pair)
    return unique_pairs


def measure_single(
    prompt_len: int,
    gen_len: int,
    model: Any,
    tokenizer: Any,
    device: Any,
    base_prompt: str,
) -> dict[str, float | int]:
    """Measure TTFT, TPOT, peak memory, and PKV payload for one config."""
    # Construct and tokenize one stable prompt at the requested token length.
    inputs = build_prompt_inputs(tokenizer, prompt_len, device, base_prompt)
    actual_prompt_len = inputs.input_ids.shape[1]

    # Start each measurement from a clean timing and peak-memory baseline.
    synchronize_if_cuda()
    reset_peak_memory_stats()

    # Prefill processes the whole prompt and returns the first cache.
    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model(input_ids=inputs.input_ids, use_cache=True)
    synchronize_if_cuda()
    ttft_ms = (time.perf_counter() - t0) * 1000

    # This is the pure KV payload after prompt tokens only.
    past_kv = outputs.past_key_values
    kv_pkv_prefill_mb = kv_cache_size_from_past_key_values_mb(past_kv)

    # Greedy next-token selection keeps generation deterministic and cheap.
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    # Decode generates one token per loop using the existing KV cache.
    decode_times = []
    out = None
    for _ in range(gen_len):
        t1 = time.perf_counter()
        with torch.no_grad():
            out = model(
                input_ids=next_token,
                past_key_values=past_kv,
                use_cache=True,
            )
        synchronize_if_cuda()
        decode_times.append((time.perf_counter() - t1) * 1000)
        past_kv = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    # Final PKV contains prompt tokens plus generated tokens.
    kv_pkv_final_mb = kv_cache_size_from_past_key_values_mb(past_kv)
    tpot_ms = sum(decode_times) / len(decode_times)

    # The formula should align with final PKV payload when cache accounting is correct.
    kv_est_mb = estimate_kv_cache_mb(model, actual_prompt_len + gen_len)

    result = {
        "prompt_len": actual_prompt_len,
        "gen_len": gen_len,
        "ttft_ms": round(ttft_ms, 2),
        "tpot_ms": round(tpot_ms, 2),
        "tokens_s": round(tokens_per_second(tpot_ms), 1),
        "peak_mem_mb": round(get_peak_gpu_memory_mb(), 1),
        "kv_est_mb": round(kv_est_mb, 2),
        "kv_pkv_prefill_mb": round(kv_pkv_prefill_mb, 2),
        "kv_pkv_final_mb": round(kv_pkv_final_mb, 2),
    }

    # Release large tensors before the next config.
    del inputs, outputs, out, past_kv, next_token
    cleanup_cuda()
    return result


def average_records(records: list[dict[str, float | int]]) -> dict[str, float | int]:
    """Average repeated measurements for one config."""
    # prompt_len/gen_len and theoretical KV are deterministic for one config,
    # while latency and peak memory are averaged over repeats.
    return {
        "prompt_len": records[0]["prompt_len"],
        "gen_len": records[0]["gen_len"],
        "ttft_ms": mean_metric(records, "ttft_ms"),
        "tpot_ms": mean_metric(records, "tpot_ms"),
        "tokens_s": mean_metric(records, "tokens_s", digits=1),
        "peak_mem_mb": mean_metric(records, "peak_mem_mb", digits=1),
        "kv_est_mb": records[0]["kv_est_mb"],
        "kv_pkv_prefill_mb": mean_metric(records, "kv_pkv_prefill_mb"),
        "kv_pkv_final_mb": mean_metric(records, "kv_pkv_final_mb"),
    }


def write_csv(path: Path, rows: list[dict[str, float | int]]) -> None:
    """Write benchmark rows to CSV."""
    # Parent directories are created here so callers only need to choose a path.
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    # CLI entrypoint for non-notebook runs.
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    require_cuda()

    # Load once, then reuse the same model for all warm-up and measured configs.
    tokenizer, model, model_path = load_causal_lm(
        model_id=config["model_id"],
        model_dir=config.get("model_dir"),
        torch_dtype=dtype_from_config(config.get("torch_dtype", "float16")),
        device_map=config.get("device_map", "auto"),
    )
    device = next(model.parameters()).device

    print(f"Model loaded: {config['model_id']}")
    print(f"Model path: {model_path}")
    print(f"Device: {device}")

    # Warm-up is intentionally excluded from CSV results.
    base_prompt = config.get("base_prompt")
    for prompt_len, gen_len in config["warmup_configs"]:
        try:
            _ = measure_single(prompt_len, gen_len, model, tokenizer, device, base_prompt)
            print(f"Warm-up done: prompt={prompt_len}, gen={gen_len}")
        except torch.cuda.OutOfMemoryError:
            cleanup_cuda()
            print(f"Warm-up OOM skipped: prompt={prompt_len}, gen={gen_len}")

    # Formal benchmark: each config is measured repeat times and averaged.
    rows = []
    for prompt_len, gen_len in build_configs(config):
        raw = []
        for repeat_idx in range(int(config["repeat"])):
            try:
                raw.append(
                    measure_single(
                        prompt_len,
                        gen_len,
                        model,
                        tokenizer,
                        device,
                        base_prompt,
                    )
                )
            except torch.cuda.OutOfMemoryError:
                cleanup_cuda()
                print(f"OOM skipped: prompt={prompt_len}, gen={gen_len}, repeat={repeat_idx + 1}")
                break
        if raw:
            row = average_records(raw)
            rows.append(row)
            print(row)

    output_csv = Path(config["output_csv"])
    write_csv(output_csv, rows)
    print(f"CSV saved: {output_csv}")


if __name__ == "__main__":
    main()
