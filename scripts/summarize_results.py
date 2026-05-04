"""Print a compact summary for an Experiment 001 CSV file."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def as_float(row: dict[str, str], key: str) -> float:
    """Read one CSV field as float."""
    return float(row[key])


def summarize(csv_path: Path) -> None:
    """Print high-level ranges for quick sanity checks."""
    # This script uses only the standard library so it can run in minimal envs.
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print(f"No rows found: {csv_path}")
        return

    # Pull each metric into a list so min/max ranges are easy to inspect.
    ttft = [as_float(row, "ttft_ms") for row in rows]
    tpot = [as_float(row, "tpot_ms") for row in rows]
    peak = [as_float(row, "peak_mem_mb") for row in rows]
    kv_final = [as_float(row, "kv_pkv_final_mb") for row in rows]

    print(f"CSV: {csv_path}")
    print(f"Rows: {len(rows)}")
    print(f"Prompt lengths: {sorted({int(float(row['prompt_len'])) for row in rows})}")
    print(f"Generation lengths: {sorted({int(float(row['gen_len'])) for row in rows})}")
    print(f"TTFT range: {min(ttft):.2f} - {max(ttft):.2f} ms")
    print(f"TPOT range: {min(tpot):.2f} - {max(tpot):.2f} ms/token")
    print(f"Peak memory range: {min(peak):.1f} - {max(peak):.1f} MB")
    print(f"Final PKV range: {min(kv_final):.2f} - {max(kv_final):.2f} MB")


def main() -> None:
    # CLI wrapper for quick local checks.
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path)
    args = parser.parse_args()
    summarize(args.csv)


if __name__ == "__main__":
    main()
