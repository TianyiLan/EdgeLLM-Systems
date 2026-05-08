"""Run Experiment 001B v2.1 Gemma 2 model-scale profiling."""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from edge_llm_systems.cuda_utils import (  # noqa: E402
    cleanup_cuda,
    reset_peak_memory_stats,
    synchronize_if_cuda,
)
from edge_llm_systems.kv_cache import (  # noqa: E402
    estimate_kv_cache_mb,
    kv_cache_size_from_past_key_values_mb,
)
from edge_llm_systems.memory import get_peak_gpu_memory_mb  # noqa: E402
from edge_llm_systems.metrics import mean_metric, tokens_per_second  # noqa: E402
from edge_llm_systems.prompts import build_prompt_inputs  # noqa: E402


FIELDNAMES = [
    "prompt_len",
    "requested_gen_len",
    "actual_gen_len",
    "ttft_ms",
    "tpot_ms",
    "tokens_s",
    "peak_mem_mb",
    "kv_est_mb",
    "kv_pkv_prefill_mb",
    "kv_pkv_final_mb",
    "kv_peak_pct",
    "status",
    "message_zh",
    "repeat_completed",
]

PROMPT_LEN_OPTIONS = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
GEN_LEN_OPTIONS = [16, 32, 64, 128]
DEFAULT_PROMPT_LENS = [64, 128, 256, 512, 1024, 2048]
DEFAULT_GEN_LENS = [32, 64, 128]

OOM_MESSAGE_ZH = (
    "\u53d1\u751f CUDA \u663e\u5b58\u4e0d\u8db3\uff1b"
    "\u672c\u7ec4\u914d\u7f6e\u4e0d\u81ea\u52a8\u964d\u4f4e prompt_len\uff0c"
    "\u76f4\u63a5\u8bb0\u5f55\u4e3a OOM\u3002"
)
LOAD_OK_MESSAGE_ZH = "\u6a21\u578b\u52a0\u8f7d\u6210\u529f\uff1b\u672a\u6267\u884c\u63a8\u7406\u3002"
DONE_MESSAGE_ZH = "\u5b8c\u6210\u3002"
ERROR_PREFIX_ZH = "\u8fd0\u884c\u5931\u8d25\uff1a"


def dtype_from_config(name: str) -> Any:
    """Map a config string to a torch dtype."""
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


def build_test_configs(prompt_lengths: list[int], gen_lengths: list[int]) -> list[tuple[int, int]]:
    """Build the prompt_len x gen_len matrix without duplicate configs."""
    pairs = [(int(prompt_len), int(gen_len)) for prompt_len in prompt_lengths for gen_len in gen_lengths]
    seen = set()
    unique_pairs = []
    for pair in pairs:
        if pair not in seen:
            seen.add(pair)
            unique_pairs.append(pair)
    return unique_pairs


def choose_warmup_gen_len(gen_lengths: list[int]) -> int:
    """Choose a stable v1-like warm-up generation length."""
    if 64 in gen_lengths:
        return 64
    not_larger_than_64 = [gen_len for gen_len in gen_lengths if gen_len <= 64]
    if not_larger_than_64:
        return max(not_larger_than_64)
    return min(gen_lengths)


def make_load_status_row(status: str, message_zh: str) -> dict[str, Any]:
    """Create a load-only CSV row without model metadata."""
    return {
        "prompt_len": 0,
        "requested_gen_len": 0,
        "actual_gen_len": "",
        "ttft_ms": "",
        "tpot_ms": "",
        "tokens_s": "",
        "peak_mem_mb": "",
        "kv_est_mb": "",
        "kv_pkv_prefill_mb": "",
        "kv_pkv_final_mb": "",
        "kv_peak_pct": "",
        "status": status,
        "message_zh": message_zh,
        "repeat_completed": 0,
    }


def _base_row(prompt_len: int, requested_gen_len: int) -> dict[str, Any]:
    """Create an empty measurement row for one selected config."""
    return {
        "prompt_len": prompt_len,
        "requested_gen_len": requested_gen_len,
        "actual_gen_len": "",
        "ttft_ms": "",
        "tpot_ms": "",
        "tokens_s": "",
        "peak_mem_mb": "",
        "kv_est_mb": "",
        "kv_pkv_prefill_mb": "",
        "kv_pkv_final_mb": "",
        "kv_peak_pct": "",
        "status": "",
        "message_zh": "",
        "repeat_completed": 0,
    }


def measure_single_v2_1(
    prompt_len: int,
    gen_len: int,
    model: Any,
    tokenizer: Any,
    device: Any,
    base_prompt: str,
) -> dict[str, float | int]:
    """Measure one prompt/gen configuration with fixed-step manual decode."""
    inputs = build_prompt_inputs(tokenizer, prompt_len, device, base_prompt)
    actual_prompt_len = inputs.input_ids.shape[1]

    synchronize_if_cuda()
    reset_peak_memory_stats()

    t0 = time.perf_counter()
    with torch.no_grad():
        outputs = model(input_ids=inputs.input_ids, use_cache=True)
    synchronize_if_cuda()
    ttft_ms = (time.perf_counter() - t0) * 1000

    past_kv = outputs.past_key_values
    kv_pkv_prefill_mb = kv_cache_size_from_past_key_values_mb(past_kv)
    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    decode_times = []
    out = None
    actual_gen_len = 0
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
        actual_gen_len += 1
        past_kv = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

    kv_pkv_final_mb = kv_cache_size_from_past_key_values_mb(past_kv)
    tpot_ms = sum(decode_times) / len(decode_times)
    peak_mem_mb = get_peak_gpu_memory_mb()
    kv_est_mb = estimate_kv_cache_mb(model, actual_prompt_len + actual_gen_len)
    kv_peak_pct = kv_pkv_final_mb / peak_mem_mb * 100 if peak_mem_mb else 0.0

    result = {
        "prompt_len": actual_prompt_len,
        "requested_gen_len": gen_len,
        "actual_gen_len": actual_gen_len,
        "ttft_ms": round(ttft_ms, 2),
        "tpot_ms": round(tpot_ms, 2),
        "tokens_s": round(tokens_per_second(tpot_ms), 1),
        "peak_mem_mb": round(peak_mem_mb, 1),
        "kv_est_mb": round(kv_est_mb, 2),
        "kv_pkv_prefill_mb": round(kv_pkv_prefill_mb, 2),
        "kv_pkv_final_mb": round(kv_pkv_final_mb, 2),
        "kv_peak_pct": round(kv_peak_pct, 2),
    }

    del inputs, outputs, out, past_kv, next_token
    cleanup_cuda()
    return result


def average_measurements(records: list[dict[str, float | int]]) -> dict[str, float | int]:
    """Average repeated successful measurements."""
    return {
        "prompt_len": records[0]["prompt_len"],
        "requested_gen_len": records[0]["requested_gen_len"],
        "actual_gen_len": records[0]["actual_gen_len"],
        "ttft_ms": mean_metric(records, "ttft_ms"),
        "tpot_ms": mean_metric(records, "tpot_ms"),
        "tokens_s": mean_metric(records, "tokens_s", digits=1),
        "peak_mem_mb": mean_metric(records, "peak_mem_mb", digits=1),
        "kv_est_mb": records[0]["kv_est_mb"],
        "kv_pkv_prefill_mb": mean_metric(records, "kv_pkv_prefill_mb"),
        "kv_pkv_final_mb": mean_metric(records, "kv_pkv_final_mb"),
        "kv_peak_pct": mean_metric(records, "kv_peak_pct"),
    }


def make_oom_row(base_row: dict[str, Any], repeat_completed: int) -> dict[str, Any]:
    """Mark a configuration as OOM without changing the test matrix."""
    row = dict(base_row)
    row["status"] = "oom"
    row["repeat_completed"] = repeat_completed
    row["message_zh"] = OOM_MESSAGE_ZH
    return row


def make_error_row(base_row: dict[str, Any], message: str, repeat_completed: int) -> dict[str, Any]:
    """Mark a configuration as a non-OOM runtime error."""
    row = dict(base_row)
    row["status"] = "error"
    row["repeat_completed"] = repeat_completed
    row["message_zh"] = f"{ERROR_PREFIX_ZH}{message}"
    return row


def run_matrix(
    *,
    prompt_lengths: list[int],
    gen_lengths: list[int],
    model: Any,
    tokenizer: Any,
    device: Any,
    repeat: int,
    base_prompt: str,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """Run the selected prompt/gen matrix and keep OOM rows in the output."""
    rows: list[dict[str, Any]] = []
    configs = build_test_configs(prompt_lengths, gen_lengths)

    def emit(event: dict[str, Any]) -> None:
        if progress_callback is not None:
            progress_callback(event)

    emit({"event": "matrix_start", "config_total": len(configs), "repeat": repeat})

    for config_index, (prompt_len, gen_len) in enumerate(configs, start=1):
        base_row = _base_row(prompt_len=prompt_len, requested_gen_len=gen_len)
        raw = []
        emit(
            {
                "event": "config_start",
                "config_index": config_index,
                "config_total": len(configs),
                "prompt_len": prompt_len,
                "requested_gen_len": gen_len,
            }
        )
        try:
            for repeat_index in range(1, repeat + 1):
                emit(
                    {
                        "event": "repeat_start",
                        "config_index": config_index,
                        "config_total": len(configs),
                        "repeat_index": repeat_index,
                        "repeat_total": repeat,
                    }
                )
                measurement = measure_single_v2_1(
                    prompt_len,
                    gen_len,
                    model,
                    tokenizer,
                    device,
                    base_prompt,
                )
                raw.append(measurement)
                emit(
                    {
                        "event": "repeat_done",
                        "config_index": config_index,
                        "config_total": len(configs),
                        "repeat_index": repeat_index,
                        "repeat_total": repeat,
                        "row": measurement,
                    }
                )
        except torch.cuda.OutOfMemoryError:
            cleanup_cuda()
            row = make_oom_row(base_row, repeat_completed=len(raw))
            rows.append(row)
            emit({"event": "config_oom", "config_index": config_index, "config_total": len(configs), "row": row})
            continue
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                cleanup_cuda()
                row = make_oom_row(base_row, repeat_completed=len(raw))
                rows.append(row)
                emit({"event": "config_oom", "config_index": config_index, "config_total": len(configs), "row": row})
                continue
            cleanup_cuda()
            row = make_error_row(base_row, str(exc), repeat_completed=len(raw))
            rows.append(row)
            emit({"event": "config_error", "config_index": config_index, "config_total": len(configs), "row": row})
            continue

        avg = average_measurements(raw)
        row = dict(base_row)
        row.update(avg)
        row["status"] = "ok"
        row["repeat_completed"] = len(raw)
        row["message_zh"] = DONE_MESSAGE_ZH
        rows.append(row)
        emit({"event": "config_done", "config_index": config_index, "config_total": len(configs), "row": row})

    emit({"event": "matrix_done", "config_total": len(configs), "rows": rows})
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write v2.1 rows to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)