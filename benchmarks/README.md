# Benchmarks

This directory contains standard benchmark entrypoints and configs.

Benchmark scripts should read a config or notebook-provided runtime settings,
call reusable utilities from `edge_llm_systems`, and write structured outputs
such as CSV files. Low-level model loading, prompt construction, CUDA cleanup,
model inspection, and KV cache accounting should stay in `edge_llm_systems`.

Current benchmark entries:

* `configs/exp001_gemma2_t4.yaml`
  * v1 Gemma 2 2B FP16 PKV baseline config.
* `profiling/run_exp001_profile.py`
  * v1 Experiment 001 PKV profiling entrypoint.
* `configs/exp001_v2_1.yaml`
  * v2.1 Gemma 2 model-scale probe defaults.
* `profiling/run_exp001_v2_1.py`
  * v2.1 notebook-facing benchmark helpers for Gemma 2 2B/9B/27B T4 pressure tests.

Example v1 CLI run:

```bash
python benchmarks/profiling/run_exp001_profile.py \
  --config benchmarks/configs/exp001_gemma2_t4.yaml
```

The v2.1 workflow is intended to start from the Colab notebook because model
choice and test matrix are exposed as notebook form controls.
