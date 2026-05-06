# Experiment 001 Results

This directory contains Experiment 001 baseline profiling results and PKV correction results for Gemma 2-2B inference profiling.

## Directory Layout

```text
results/exp001/
├── csv/                 # final official CSV results
├── figures/             # final official figures
└── temp/                # historical / temporary results kept for reference
    ├── csv/
    └── figures/
```

## Final Official Results

Use these files as the current Experiment 001-PKV final results:

* `csv/exp001_results_pkv_modular.csv`
* `figures/exp001_profiling_results_pkv_modular.png`

These results come from the modular PKV benchmark flow and should be used for Stage 1 baseline profiling conclusions.

## Historical / Temporary Results

The following files are retained under `temp/` for experiment history and comparison only:

* `temp/csv/gemma_baseline_profiling.csv`
* `temp/csv/exp001_results_v2.csv`
* `temp/csv/exp001_results_pkv.csv`
* `temp/figures/exp001_profiling_results.png`
* `temp/figures/exp001_profiling_results_v2.png`
* `temp/figures/exp001_profiling_results_v3.png`
* `temp/figures/exp001_profiling_results_pkv.png`

## Measurement Note

The older CUDA memory-delta KV measurement is kept as a historical system-level memory observation only.

It should not be used as the basis for pure KV cache conclusions, because CUDA memory delta and peak memory include non-KV runtime, allocator, and temporary tensor overhead.

For pure KV cache size, use the modular PKV payload result in `csv/exp001_results_pkv_modular.csv`.
