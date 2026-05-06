"""Plot Experiment 001 PKV profiling results from CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd


def _detect_hardware_label() -> str:
    """Return the current CUDA device name for figure titles when available."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "GPU unknown"


def _detect_gpu_memory_gb() -> int:
    """Return total GPU memory in whole GB, or 16 as a safe default."""
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return round(props.total_memory / 1024**3)
    except Exception:
        pass
    return 16


def plot_exp001(csv_path: Path, output_path: Path, hardware_label: str | None = None) -> None:
    """Create the Experiment 001 PKV overview figure."""
    # The plotting script consumes a result CSV only. It does not rerun models.
    if hardware_label is None:
        hardware_label = _detect_hardware_label()
    df = pd.read_csv(csv_path)

    # Use linear x-axis spacing so prompt length growth is visually meaningful.
    all_prompt_lens = sorted(df["prompt_len"].unique())
    gen_len_list = sorted(df["gen_len"].unique())

    # Stable colors make figures comparable across reruns.
    color_map = {32: "#4C9BE8", 64: "#F5A623", 128: "#7ED321"}

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(
        "Experiment 001 PKV: Gemma 2-2B Profiling\n"
        f"({hardware_label}, FP16, KV from past_key_values)",
        fontsize=13,
        fontweight="bold",
    )

    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.42, wspace=0.35)
    ax1 = fig.add_subplot(gs[0, 0:2])
    ax2 = fig.add_subplot(gs[0, 2:4])
    ax3 = fig.add_subplot(gs[1, 0:2])
    ax4a = fig.add_subplot(gs[1, 2])
    ax4b = fig.add_subplot(gs[1, 3])

    # 1. Prefill latency: should grow with prompt length.
    for gl in gen_len_list:
        sub = df[df["gen_len"] == gl].sort_values("prompt_len")
        ax1.plot(
            sub["prompt_len"],
            sub["ttft_ms"],
            marker="o",
            color=color_map.get(gl),
            label=f"gen={gl}",
            linewidth=2,
        )
    ax1.set_xlabel("Prompt Length (tokens)")
    ax1.set_ylabel("TTFT (ms)")
    ax1.set_title("1. TTFT vs Prompt Length")
    ax1.set_xticks(all_prompt_lens)
    ax1.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Decode latency: expected to stay relatively flat in this range.
    for gl in gen_len_list:
        sub = df[df["gen_len"] == gl].sort_values("prompt_len")
        ax2.plot(
            sub["prompt_len"],
            sub["tpot_ms"],
            marker="s",
            color=color_map.get(gl),
            label=f"gen={gl}",
            linewidth=2,
        )
    ax2.set_xlabel("Prompt Length (tokens)")
    ax2.set_ylabel("TPOT (ms)")
    ax2.set_title("2. TPOT vs Prompt Length")
    ax2.set_xticks(all_prompt_lens)
    ax2.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax2.set_ylim(0, 100)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. System-level memory pressure, not pure KV cache payload.
    for gl in gen_len_list:
        sub = df[df["gen_len"] == gl].sort_values("prompt_len")
        ax3.plot(
            sub["prompt_len"],
            sub["peak_mem_mb"],
            marker="^",
            color=color_map.get(gl),
            label=f"gen={gl}",
            linewidth=2,
        )
    gpu_mem_gb = _detect_gpu_memory_gb()
    ax3.axhline(y=gpu_mem_gb * 1024, color="red", linestyle="--", alpha=0.5, label=f"{gpu_mem_gb} GB limit")
    ax3.set_xlabel("Prompt Length (tokens)")
    ax3.set_ylabel("Peak GPU Memory (MB)")
    ax3.set_title("3. Peak Memory vs Prompt Length")
    ax3.set_xticks(all_prompt_lens)
    ax3.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4-A. Compare formula and measured PKV on the longest generation length.
    representative_gen = max(gen_len_list)
    sub_rep = df[df["gen_len"] == representative_gen].sort_values("prompt_len")
    ax4a.plot(
        sub_rep["prompt_len"],
        sub_rep["kv_pkv_final_mb"],
        marker="o",
        color="#4C9BE8",
        label="PKV measured",
        linewidth=2,
    )
    ax4a.plot(
        sub_rep["prompt_len"],
        sub_rep["kv_est_mb"],
        marker="D",
        color="#F5A623",
        linestyle="--",
        label="Formula estimate",
        linewidth=2,
    )
    ax4a.set_xlabel("Prompt Length (tokens)")
    ax4a.set_ylabel("KV Cache Size (MB)")
    ax4a.set_title(f"4-A. PKV vs Formula (gen={representative_gen})")
    ax4a.set_xticks(all_prompt_lens)
    ax4a.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax4a.tick_params(axis="x", labelrotation=45)
    ax4a.legend(fontsize=8)
    ax4a.grid(True, alpha=0.3)

    # 4-B. Show how generation length shifts final PKV payload.
    for gl in gen_len_list:
        sub = df[df["gen_len"] == gl].sort_values("prompt_len")
        ax4b.plot(
            sub["prompt_len"],
            sub["kv_pkv_final_mb"],
            marker="o",
            linewidth=2,
            color=color_map.get(gl),
            label=f"gen={gl}",
        )
    ax4b.set_xlabel("Prompt Length (tokens)")
    ax4b.set_ylabel("Final PKV Cache Size (MB)")
    ax4b.set_title("4-B. PKV Payload vs Prompt Length")
    ax4b.set_xticks(all_prompt_lens)
    ax4b.tick_params(axis="x", labelrotation=45)
    ax4b.grid(True, alpha=0.3)
    ax4b.legend(title="Gen Length", fontsize=8, title_fontsize=8, loc="upper left")

    # Keep file creation inside the script so callers can pass new output paths.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved: {output_path}")


def main() -> None:
    # CLI wrapper; notebooks can import and call plot_exp001 directly.
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    plot_exp001(args.csv, args.output)


if __name__ == "__main__":
    main()
