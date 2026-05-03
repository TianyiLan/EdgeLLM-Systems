# Experiment 001 Results

This directory contains Experiment 001 baseline profiling results and the PKV correction results for Gemma 2-2B inference profiling.

Recommended latest files:

* `exp001_results_pkv.csv`
* `figures/exp001_profiling_results_pkv.png`

Historical files:

* `gemma_baseline_profiling.csv`
* `exp001_results_v2.csv`
* `figures/exp001_profiling_results.png`
* `figures/exp001_profiling_results_v2.png`
* `figures/exp001_profiling_results_v3.png`

The older memory-delta KV measurement is kept as historical record only. It should no longer be used as the basis for pure KV cache conclusions, because CUDA memory delta and peak memory include non-KV runtime, allocator, and temporary tensor overhead.
