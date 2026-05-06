# Experiment Log

## Stage 1：Performance Characterization（性能表征）

### 阶段状态

| 项目 | 内容 |
|---|---|
| 阶段状态 | 已完成 |
| 最终实验 | Experiment 001-PKV Modular |
| 最终数据源 | `results/exp001/csv/exp001_results_pkv_modular.csv` |
| 最终图像 | `results/exp001/figures/exp001_profiling_results_pkv_modular.png` |
| 模型 | `google/gemma-2-2b-it` |
| 主要平台 | Google Colab, NVIDIA Tesla T4, FP16, batch size = 1 |

Stage 1 已正式结束。本阶段的目标是建立可信的基线性能测量流程，并完成 KV Cache 测量协议校准。

从本日志版本开始，Stage 1 的最终结论统一基于以下模块化 PKV 结果文件：

```text
results/exp001/csv/exp001_results_pkv_modular.csv
```

早期结果文件仅作为历史记录和对照材料保留，不再作为 Stage 1 的最终基线数据来源。

---

## 1. Experiment 001-PKV Modular：Gemma 基线性能分析与 KV Cache 测量校准

### 1.1 实验背景

EdgeLLM-Systems 聚焦资源受限边缘环境下的大模型推理系统研究。在进入瓶颈分析、GPU 软件优化或异构硬件探索之前，项目需要先建立一个稳定、可复现、指标定义清晰的基线测量体系。

Experiment 001 的目标是回答以下基础问题：

1. TTFT、TPOT、吞吐率、峰值显存和 KV Cache payload 如何随 `prompt_len` 与 `gen_len` 变化？
2. 是否可以使用 `past_key_values` 直接、可靠地测量 KV Cache payload？
3. 如何区分 pure KV Cache payload 与 system-level GPU memory pressure？
4. 哪些现象足够稳定，可以作为 Stage 2 的基线依据？

Stage 1 不负责证明最终瓶颈。本阶段的任务是完成测量协议校准与基线性能表征。真正的瓶颈识别从 Stage 2 开始。

---

## 2. 实验设置

### 2.1 模型设置

| 项目 | 内容 |
|---|---|
| 模型 | `google/gemma-2-2b-it` |
| 架构类型 | Decoder-only LLM |
| 精度 | FP16 |
| Batch size | 1 |
| 推理方式 | 自回归生成 |
| KV Cache | 启用 |

选择 Gemma 2 2B Instruct 的原因是：该模型可以在 16GB Tesla T4 上稳定运行，同时能够体现典型的 prefill、decode 和 KV Cache 行为。

### 2.2 平台设置

| 项目 | 内容 |
|---|---|
| 主要运行环境 | Google Colab |
| GPU | NVIDIA Tesla T4 |
| GPU 显存 | 16GB |
| 运行时 | CUDA-enabled PyTorch environment |
| 框架 | Hugging Face Transformers + PyTorch |

需要精确复现实验时，应以 benchmark 运行日志或 notebook 输出中的 package version 为准。本日志只记录最终数据解释，不手动固化环境版本。

### 2.3 参数矩阵

| 变量 | 取值 |
|---|---|
| `prompt_len` | 64 / 128 / 256 / 512 / 1024 / 2048 |
| `gen_len` | 32 / 64 / 128 |
| `batch_size` | 1 |
| `dtype` | FP16 |

最终模块化 PKV 实验共包含 18 组配置。

---

## 3. 测量协议

### 3.1 延迟指标

| 指标 | 含义 |
|---|---|
| `ttft_ms` | Time To First Token，首 token 延迟，主要反映 prefill 阶段耗时。 |
| `tpot_ms` | Time Per Output Token，单 token 解码延迟，主要反映 decode 阶段耗时。 |
| `tokens_s` | 输出吞吐率，通常近似等于 `1000 / TPOT(ms)`。 |

### 3.2 显存指标

| 指标 | 含义 |
|---|---|
| `peak_mem_mb` | CUDA peak allocated memory，用作系统级显存压力指标。 |
| `kv_est_mb` | 根据模型结构和总序列长度计算得到的理论 KV Cache 大小。 |
| `kv_pkv_prefill_mb` | Prefill 后 `past_key_values` 中 K/V tensor 的实际 payload。此时 cache length 等于 `prompt_len`。 |
| `kv_pkv_final_mb` | 生成结束后 `past_key_values` 中 K/V tensor 的实际 payload。此时 cache length 等于 `prompt_len + gen_len`。 |

### 3.3 KV Cache 测量方法

Stage 1 的最终协议采用 `past_key_values` payload accounting 作为 pure KV Cache 的主测量方法。

对 `outputs.past_key_values` 中每个 K/V tensor，payload 按如下方式计算：

```text
numel × element_size
```

该方法直接统计 decode 阶段后续需要保留的 K/V tensor payload，不包含 CUDA allocator、临时 activation、logits、attention workspace、runtime buffer 或显存碎片。

因此：

- `kv_pkv_final_mb` 是最终 pure KV Cache payload 测量值；
- `kv_est_mb` 是理论参考值；
- `peak_mem_mb` 不是 pure KV Cache，而是 system-level memory pressure 指标。

### 3.4 历史显存差分法的处理

早期使用的 CUDA memory-delta 方法不再作为 pure KV Cache 测量方法。

原因是 CUDA memory delta 和 CUDA peak memory 会混入大量 non-KV runtime overhead，包括：

- 模型权重；
- PyTorch allocator 行为；
- temporary activation；
- logits；
- attention workspace；
- CUDA runtime buffer；
- memory fragmentation。

因此，最终 Stage 1 结论统一以模块化 PKV 结果为准。

---

## 4. 最终实验数据

数据来源：

```text
results/exp001/csv/exp001_results_pkv_modular.csv
```

| prompt_len | gen_len | TTFT(ms) | TPOT(ms) | tokens/s | peak_mem(MB) | kv_est(MB) | kv_pkv_prefill(MB) | kv_pkv_final(MB) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 32 | 60.05 | 53.90 | 18.8 | 5065.4 | 9.75 | 6.5 | 9.75 |
| 64 | 64 | 66.38 | 47.17 | 21.2 | 5065.4 | 13.00 | 6.5 | 13.00 |
| 64 | 128 | 57.02 | 51.91 | 19.3 | 5065.4 | 19.50 | 6.5 | 19.50 |
| 128 | 32 | 62.14 | 49.16 | 20.3 | 5133.2 | 16.25 | 13.0 | 16.25 |
| 128 | 64 | 70.80 | 54.82 | 18.3 | 5133.2 | 19.50 | 13.0 | 19.50 |
| 128 | 128 | 65.30 | 53.74 | 18.8 | 5133.2 | 26.00 | 13.0 | 26.00 |
| 256 | 32 | 89.72 | 46.89 | 21.3 | 5273.7 | 29.25 | 26.0 | 29.25 |
| 256 | 64 | 91.70 | 51.97 | 19.3 | 5273.7 | 32.50 | 26.0 | 32.50 |
| 256 | 128 | 90.71 | 49.64 | 20.3 | 5273.7 | 39.00 | 26.0 | 39.00 |
| 512 | 32 | 182.01 | 54.82 | 18.5 | 5548.9 | 55.25 | 52.0 | 55.25 |
| 512 | 64 | 181.23 | 50.23 | 20.1 | 5548.9 | 58.50 | 52.0 | 58.50 |
| 512 | 128 | 184.41 | 49.77 | 20.1 | 5548.9 | 65.00 | 52.0 | 65.00 |
| 1024 | 32 | 433.78 | 51.82 | 19.5 | 6108.1 | 107.25 | 104.0 | 107.25 |
| 1024 | 64 | 437.15 | 47.49 | 21.1 | 6108.1 | 110.50 | 104.0 | 110.50 |
| 1024 | 128 | 437.31 | 50.54 | 19.8 | 6108.1 | 117.00 | 104.0 | 117.00 |
| 2048 | 32 | 1096.18 | 53.97 | 18.7 | 7211.6 | 211.25 | 208.0 | 211.25 |
| 2048 | 64 | 1106.62 | 54.02 | 18.7 | 7211.6 | 214.50 | 208.0 | 214.50 |
| 2048 | 128 | 1107.20 | 52.61 | 19.0 | 7211.6 | 221.00 | 208.0 | 221.00 |

---

## 5. 聚合结果

### 5.1 按 Prompt Length 聚合

| prompt_len | avg TTFT(ms) | avg TPOT(ms) | avg tokens/s | peak_mem(MB) | PKV prefill(MB) | avg PKV final(MB) |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 61.15 | 50.99 | 19.77 | 5065.4 | 6.5 | 14.08 |
| 128 | 66.08 | 52.57 | 19.13 | 5133.2 | 13.0 | 20.58 |
| 256 | 90.71 | 49.50 | 20.30 | 5273.7 | 26.0 | 33.58 |
| 512 | 182.55 | 51.61 | 19.57 | 5548.9 | 52.0 | 59.58 |
| 1024 | 436.08 | 49.95 | 20.13 | 6108.1 | 104.0 | 111.58 |
| 2048 | 1103.33 | 53.53 | 18.80 | 7211.6 | 208.0 | 215.58 |

### 5.2 按 Generation Length 聚合

| gen_len | avg TTFT(ms) | avg TPOT(ms) | avg tokens/s | avg peak_mem(MB) | avg PKV final(MB) |
|---:|---:|---:|---:|---:|---:|
| 32 | 320.65 | 51.76 | 19.52 | 5723.48 | 71.50 |
| 64 | 325.65 | 50.95 | 19.78 | 5723.48 | 74.75 |
| 128 | 323.66 | 51.37 | 19.55 | 5723.48 | 81.25 |

---

## 6. 结果分析

### 6.1 TTFT 主要受 prompt_len 影响

TTFT 随 `prompt_len` 增长显著上升。

按 `prompt_len` 聚合后的平均值如下：

```text
prompt_len=64    → avg TTFT = 61.15 ms
prompt_len=128   → avg TTFT = 66.08 ms
prompt_len=256   → avg TTFT = 90.71 ms
prompt_len=512   → avg TTFT = 182.55 ms
prompt_len=1024  → avg TTFT = 436.08 ms
prompt_len=2048  → avg TTFT = 1103.33 ms
```

TTFT 与 `gen_len` 没有稳定对应关系。这符合预期，因为 TTFT 主要反映 prefill 阶段，而不是后续 decode 阶段。

在长 prompt 区间，TTFT 增长明显加速：

```text
512  → 1024 tokens: 182.55 ms → 436.08 ms
1024 → 2048 tokens: 436.08 ms → 1103.33 ms
```

这说明长上下文 prefill 成本在 Stage 2 中需要继续分析。

### 6.2 TPOT 在当前测试区间内基本稳定

18 组配置中：

```text
TPOT min = 46.89 ms
TPOT max = 54.82 ms
TPOT avg ≈ 51.36 ms
```

按 `gen_len` 聚合：

```text
gen_len=32  → avg TPOT = 51.76 ms
gen_len=64  → avg TPOT = 50.95 ms
gen_len=128 → avg TPOT = 51.37 ms
```

这表明在当前模型、当前平台、当前序列长度范围内，decode 阶段单 token 延迟相对稳定。

### 6.3 早期 gen_len=64 tokens/s 异常不作为最终结论

早期单次实验曾出现 `gen_len=64` 下 tokens/s 下降的现象。模块化 PKV 最终数据没有复现该规律。

按 `gen_len` 聚合：

```text
gen_len=32  → avg tokens/s = 19.52
gen_len=64  → avg tokens/s = 19.78
gen_len=128 → avg tokens/s = 19.55
```

因此，早期 `gen_len=64` 异常归类为 early measurement noise 或 runtime jitter，不作为 Stage 1 unresolved issue 保留。

### 6.4 Peak memory 主要受 prompt_len 影响

在最终模块化 PKV 数据中，相同 `prompt_len` 下不同 `gen_len` 的 `peak_mem_mb` 完全一致。

按 `prompt_len` 观察：

```text
prompt_len=64    → 5065.4 MB
prompt_len=128   → 5133.2 MB
prompt_len=256   → 5273.7 MB
prompt_len=512   → 5548.9 MB
prompt_len=1024  → 6108.1 MB
prompt_len=2048  → 7211.6 MB
```

这说明在当前配置下，系统级峰值显存主要由模型权重、prefill 阶段显存压力、CUDA runtime 和 allocator 行为共同决定。生成阶段的 KV Cache 增量可以在 PKV payload 中观察到，但没有主导 CUDA peak memory。

### 6.5 PKV final 与理论 KV Cache 完全一致

最终数据中，每一行均满足：

```text
kv_pkv_final_mb == kv_est_mb
```

最大观测差异为：

```text
0.0 MB
```

这验证了 `past_key_values` payload accounting 的准确性。

### 6.6 PKV prefill 随 prompt_len 线性增长

Prefill 后 KV Cache payload 呈严格线性增长：

```text
prompt_len=64    → 6.5 MB
prompt_len=128   → 13.0 MB
prompt_len=256   → 26.0 MB
prompt_len=512   → 52.0 MB
prompt_len=1024  → 104.0 MB
prompt_len=2048  → 208.0 MB
```

观测斜率为：

```text
0.1015625 MB/token
```

这说明 KV Cache payload 与序列长度之间满足线性关系。

### 6.7 PKV prefill 小于 PKV final 是合理现象

对每组配置都有：

```text
kv_pkv_prefill_mb < kv_pkv_final_mb
```

原因是：

```text
prefill cache length = prompt_len
final cache length   = prompt_len + gen_len
```

二者差值对应 decode 阶段新增 token 写入 KV Cache 的 payload。

### 6.8 CUDA peak memory 不能被解释为 pure KV Cache

以最大测试配置为例：

```text
prompt_len=2048, gen_len=128
peak_mem_mb      = 7211.6 MB
kv_pkv_final_mb  = 221.0 MB
```

pure KV Cache payload 远小于 CUDA peak memory。因此，CUDA peak memory 只能作为 system-level memory pressure 指标，不能作为 pure KV Cache 测量值。

这是 Stage 1 最重要的方法论结论。

---

## 7. 历史文件解释策略

以下文件仅作为历史记录保留：

```text
results/exp001/gemma_baseline_profiling.csv
results/exp001/temp/csv/exp001_results_v2.csv
results/exp001/temp/csv/exp001_results_pkv.csv
results/exp001/temp/figures/exp001_profiling_results.png
results/exp001/temp/figures/exp001_profiling_results_v2.png
results/exp001/temp/figures/exp001_profiling_results_v3.png
results/exp001/temp/figures/exp001_profiling_results_pkv.png
```

Stage 1 的最终数据源为：

```text
results/exp001/csv/exp001_results_pkv_modular.csv
results/exp001/figures/exp001_profiling_results_pkv_modular.png
```

历史数据不能覆盖最终模块化 PKV 数据结论。

### 7.1 早期单次实验结果

早期单次实验用于搭建第一版 profiling workflow，但可靠性弱于最终模块化 benchmark 输出。

早期出现的 `gen_len=64` tokens/s 下降现象不作为最终稳定结论保留。

### 7.2 V2 memory-delta 结果

V2 结果扩展了 prompt length 到 1024 / 2048，并引入 CUDA memory-delta 观察。但后续证明 CUDA memory-delta 会混入明显 non-KV overhead。

因此，V2 memory-delta 只作为历史系统级显存观察，不作为 pure KV Cache 依据。

### 7.3 非模块化 PKV correction 结果

非模块化 PKV correction 结果验证了 `past_key_values` payload measurement 的可行性，是重要的中间步骤。

最终 Stage 1 baseline 以模块化 PKV 结果为准，因为它符合当前项目 benchmark 架构。

---

## 8. 已解决与延期事项

### 8.1 CUDA memory-delta 不是 pure KV Cache 指标

**状态**：已解决

CUDA memory-delta 和 peak memory 包含 non-KV runtime overhead。pure KV Cache 测量已更正为 `past_key_values` payload accounting。

### 8.2 `gen_len=64` tokens/s 异常

**状态**：已解决

最终模块化 PKV 数据没有显示稳定的 `gen_len=64` tokens/s 下降。该早期现象归类为 early measurement noise / runtime jitter。

### 8.3 INT8 / INT4 量化

**状态**：延期

INT8 / INT4 量化不是 Stage 1 必做内容，应进入后续 deployability、compression 或 optimization 实验。

### 8.4 Cross-GPU P100 baseline

**状态**：延期

P100 cross-GPU baseline 曾被考虑，但不是 Stage 1 完成条件。后续如需要跨设备对照，可在更合适的平台或订阅环境中补充。

### 8.5 历史图片字体或格式问题

**状态**：不影响最终结论

早期图像生成与字体问题不影响最终模块化 PKV 数据源。

---

## 9. Stage 1 退出条件

Stage 1 满足以下退出条件：

- [x] 建立稳定的 FP16 baseline profiling workflow。
- [x] 系统记录 TTFT、TPOT、tokens/s、peak memory 和 KV Cache 指标。
- [x] 系统控制 prompt length 与 generation length。
- [x] 实现并验证 `past_key_values` payload measurement。
- [x] 理论 KV Cache 与 PKV payload 在最终数据中完全一致。
- [x] 明确区分 CUDA peak memory 与 pure KV Cache payload。
- [x] 早期不稳定观察已被解释、解决或降级为历史说明。
- [x] 最终 Stage 1 数据源固定为模块化 PKV 结果。

---

## 10. Stage 1 最终结论

Experiment 001-PKV Modular 为 EdgeLLM-Systems 建立了第一版可信 baseline。

Stage 1 的最终结论如下：

1. TTFT 主要由 `prompt_len` 决定，并在长 prompt 区间显著上升。
2. TPOT 在当前测试范围内基本稳定，约为 50 ms/token。
3. 最终数据不支持 `gen_len=64` tokens/s 稳定下降。
4. CUDA peak memory 主要随 `prompt_len` 增长，反映系统级显存压力。
5. Pure KV Cache payload 随 sequence length 线性增长。
6. `past_key_values` payload 与理论 KV Cache 估算完全一致。
7. CUDA peak memory 不能作为 pure KV Cache 测量值。
8. Stage 1 提供了可靠 baseline，但不负责单独完成最终瓶颈识别。

Stage 1 正式关闭。

---

## 11. 进入 Stage 2

下一阶段为：

```text
Stage 2：Memory-Constrained Inference Analysis
```

Stage 2 将基于 Stage 1 modular PKV baseline，从测量协议校准进入瓶颈识别。

Stage 2 的核心问题包括：

1. 在资源受限边缘场景中，最先限制部署的是模型权重、KV Cache、activation/runtime overhead、带宽，还是具体实现？
2. Context length、batch size、precision 和 KV Cache 增长如何共同影响 deployment boundary？
3. 在更长上下文或更大 batch 下，decode latency 是否逐步表现出 memory-bound 特征？
4. KV Cache compression、migration、offloading 或 alternative implementation 能否改善部署边界或推理性能？
5. 在进入 GPU software optimization 和 heterogeneous hardware exploration 前，最应该优先验证哪条优化路径？

Stage 2 应从 deployment boundary analysis 开始，而不是立即进入 CUDA 或 FPGA 优化。
