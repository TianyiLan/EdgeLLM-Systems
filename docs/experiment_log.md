# Experiment Log

## Stage 1: Performance Characterization

### Current Status

| Item | Status |
|---|---|
| Stage 1A | Completed |
| Stage 1A final experiment | Experiment 001A / Experiment 001-PKV Modular |
| Stage 1A final data | `results/exp001/csv/exp001_results_pkv_modular.csv` |
| Stage 1A final figure | `results/exp001/figures/exp001_profiling_results_pkv_modular.png` |
| Stage 1B | In progress / supplementary model-scale stress baseline |
| Stage 1B current experiment | Experiment 001B v2.1 |
| Stage 1B current data | `results/exp001b/` |
| Primary platform | Google Colab, NVIDIA Tesla T4, FP16, batch size = 1 |

Stage 1 当前拆分为 Stage 1A 和 Stage 1B。

Stage 1A 已完成 Gemma 2 2B IT 的可信 baseline profiling 与 KV Cache 测量协议校准。Stage 1B 在同一测量协议基础上，补充 Gemma family model-scale stress baseline，用于观察 T4 FP16 环境下的模型规模部署边界。

在是否补充 Gemma 4 probing 明确之前，Stage 1 不应写作完全关闭。

---

## Stage 1A: Experiment 001A / PKV Modular Baseline

### Goal

Stage 1A 的目标是建立可复现 FP16 inference baseline，并修正 KV cache 测量协议。最终协议使用 `past_key_values` payload accounting 作为 pure KV cache measurement；CUDA peak memory 只作为 system-level memory pressure metric。

### Setup

| Item | Value |
|---|---|
| Model | `google/gemma-2-2b-it` |
| Platform | Google Colab, NVIDIA Tesla T4 |
| Precision | FP16 |
| Batch size | 1 |
| Prompt lengths | 64 / 128 / 256 / 512 / 1024 / 2048 |
| Generation lengths | 32 / 64 / 128 |
| Final result file | `results/exp001/csv/exp001_results_pkv_modular.csv` |
| Final overview figure | `results/exp001/figures/exp001_profiling_results_pkv_modular.png` |

### Main Findings

1. TTFT 主要随 `prompt_len` 增长，反映 prefill cost。
2. TPOT 在当前测试区间内相对稳定，约为 50 ms/token。
3. `past_key_values` payload 与 theoretical KV cache formula 高度一致。
4. CUDA peak memory 显著大于 pure KV payload，不能作为 pure KV cache measurement。
5. Prefill PKV 小于 final PKV 是合理现象，因为 prefill cache length = `prompt_len`，final cache length = `prompt_len + gen_len`。
6. 早期 `gen_len=64` tokens/s 异常未在最终 modular PKV 结果中复现，归类为 early measurement noise / runtime jitter。

### Deprecated Historical Measurement

早期 CUDA memory-delta based KV cache measurement 已废弃为 pure KV cache 测量方法。该方法混入 model weights、allocator behavior、temporary activations、logits、attention workspace、runtime buffers 和 fragmentation 等 non-KV overhead。历史 V1/V2 文件仅作为开发记录和 system-level memory observation 保留。

Stage 1A 已完成。

---

## Stage 1B: Experiment 001B v2.1 Gemma 2 Model-Scale Stress Baseline

### Goal

Experiment 001B 关注更大 Gemma 模型在 Tesla T4 FP16 环境中的部署边界。它复用 Stage 1A 的 PKV measurement protocol，不自动降低精度、不缩小测试矩阵、不启用量化或 offload。

关系如下：

- Stage 1A 验证 baseline profiling 与 PKV measurement protocol；
- Stage 1B 使用该协议观察模型规模增长带来的部署边界。

### Result Directory

```text
results/exp001b/
```

结果按模型分组：

```text
results/exp001b/gemma_2_2B_IT/
results/exp001b/gemma_2_9B_IT/
```

### Gemma 2 2B IT Result

| Item | Value |
|---|---|
| Model | `google/gemma-2-2b-it` |
| Status | Completed |
| Recommended CSV | `results/exp001b/gemma_2_2B_IT/csv/exp001b_gemma2_2b_it_t4_fp16.csv` |
| Original CSV | `results/exp001b/gemma_2_2B_IT/csv/exp001B_v2_1_gemma-2-2b-it_p64-128-256-512-1024-2048_g32-64-128_20260507_142512.csv` |
| Recommended figure | `results/exp001b/gemma_2_2B_IT/figures/exp001b_gemma2_2b_it_t4_fp16.png` |
| Original figure | `results/exp001b/gemma_2_2B_IT/figures/exp001B_v2_1_gemma-2-2b-it_p64-128-256-512-1024-2048_g32-64-128_20260507_142512.png` |

新的 2B run 与 Stage 1A baseline 高度一致：

- TTFT 仍随 prompt length 增长；
- TPOT 仍约为 50 ms/token；
- `actual_gen_len` 与 `requested_gen_len` 一致；
- PKV payload 仍与 theoretical KV cache formula 对齐；
- peak memory 仍显著大于 pure KV payload。

在当前矩阵下，Gemma 2 2B IT FP16 没有触及 T4 显存边界。peak memory 约为 5.1 GB 到 7.2 GB，最大 final PKV payload 约为 221 MB，KV payload 占 peak memory 的比例最高约 3.1%。

### Gemma 2 9B IT Result

| Item | Value |
|---|---|
| Model | `google/gemma-2-9b-it` |
| Status | CUDA OOM during model loading |
| OOM CSV | `results/exp001b/gemma_2_9B_IT/csv/exp001B_v2_1_gemma-2-9b-it_p64-128-256-512-1024-2048_g32-64-128_20260507_150118.csv` |

Gemma 2 9B IT FP16 在 Tesla T4 上模型加载阶段发生 CUDA OOM，尚未进入 prompt/generation matrix。该结果是有效的 deployment-boundary evidence，不是 benchmark 逻辑失败。

本次 benchmark 按设计直接记录 OOM，不自动执行以下行为：

- 缩小 `prompt_len`；
- 缩小 `gen_len`；
- 切换 precision；
- 启用 quantization；
- 使用 CPU/disk offload。

该结果说明 Gemma 2 9B IT FP16 已在 model-load stage 触发 T4 部署边界。

### Current Stage 1B Conclusion

当前 Stage 1B 证据支持：

- Gemma 2 2B IT FP16 可在 T4 上稳定运行当前单 batch 测试矩阵；
- Gemma 2 9B IT FP16 在 T4 上发生 load-stage OOM；
- 2B 未触及 T4 显存边界，9B 已触发 FP16 部署边界。

这为 Stage 2 的 deployment boundary analysis 提供了直接依据。但在是否补充 Gemma 4 probing 明确之前，Stage 1B 仍保持开放状态。

---

## Deferred Items

- INT8 / INT4 quantization 延后到 deployability、compression 或 optimization 实验。
- Cross-GPU P100 baseline 不是 Stage 1A 完成条件，仅在后续有合适环境时补充。
- Gemma 4 model-scale probing 可作为 Stage 1B v2.2 扩展，目前尚未记录为已完成。

---

## Stage 2 Preparation

Stage 2 将聚焦 memory-constrained inference analysis。它应基于 Stage 1A 的 PKV baseline 和 Stage 1B 的 deployment-boundary evidence，继续分析部署限制主要来自 model weights、KV cache growth、temporary runtime overhead、bandwidth，还是具体实现路径。
