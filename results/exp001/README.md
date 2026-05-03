# Experiment 001 Results

This directory contains Experiment 001 baseline profiling results and PKV correction results for Gemma 2-2B inference profiling.

## Latest Recommended Files

Use these files for the current Experiment 001-PKV measurement protocol:

* `exp001_results_pkv.csv`
* `figures/exp001_profiling_results_pkv.png`

## Historical Files

The following files are retained for experiment history and comparison:

* `gemma_baseline_profiling.csv`
* `exp001_results_v2.csv`
* `figures/exp001_profiling_results.png`
* `figures/exp001_profiling_results_v2.png`
* `figures/exp001_profiling_results_v3.png`

## Measurement Note

The older V2 memory-delta KV measurement is kept as a historical system-level memory observation only.

It should not be used as the basis for pure KV cache conclusions, because CUDA memory delta and peak memory include non-KV runtime, allocator, and temporary tensor overhead.

For pure KV cache size, use the PKV payload results in `exp001_results_pkv.csv`.
