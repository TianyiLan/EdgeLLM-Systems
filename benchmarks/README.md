# Benchmarks

This directory contains standard benchmark entrypoints and configs.

Benchmark scripts should read a config, call reusable utilities from
`heteroinfer_lab`, and write structured outputs such as CSV files. Low-level
model loading, prompt construction, CUDA cleanup, and KV cache accounting should
stay in `heteroinfer_lab`.

Current benchmark:

* `configs/exp001_gemma2_t4.yaml`
* `profiling/run_exp001_profile.py`

Example:

```bash
python benchmarks/profiling/run_exp001_profile.py \
  --config benchmarks/configs/exp001_gemma2_t4.yaml
```
