# Experiment 001B Results

Experiment 001B 记录 Gemma family model-scale stress baseline on T4，用于在 Stage 1A 的 Gemma 2 2B PKV baseline 基础上，继续观察更大模型在 Tesla T4 FP16 环境中的部署边界。

## 实验目的

Experiment 001B 不是新的 KV cache 测量协议。它复用 Stage 1A 已校准的 `past_key_values` payload measurement，并进一步回答：

> 在相同 Tesla T4 FP16 条件下，不改变精度、不启用 offload、不缩小测试矩阵时，不同规模 Gemma 模型能否完成加载和基础 profiling？

## 目录结构

```text
results/exp001b/
├── gemma_2_2B_IT/
│   ├── csv/
│   ├── figures/
│   └── exp_info/
└── gemma_2_9B_IT/
    ├── csv/
    ├── figures/
    └── exp_info/
```

结果按模型分组保存，避免成功运行结果和 OOM 边界记录混在同一层目录中。

## Gemma 2 2B IT

状态：Tesla T4 / FP16 / batch size = 1 下成功完成。

推荐 CSV：

```text
gemma_2_2B_IT/csv/exp001b_gemma2_2b_it_t4_fp16.csv
```

原始长文件名 CSV：

```text
gemma_2_2B_IT/csv/exp001B_v2_1_gemma-2-2b-it_p64-128-256-512-1024-2048_g32-64-128_20260507_142512.csv
```

推荐总览图片：

```text
gemma_2_2B_IT/figures/exp001b_gemma2_2b_it_t4_fp16.png
```

原始长文件名图片：

```text
gemma_2_2B_IT/figures/exp001B_v2_1_gemma-2-2b-it_p64-128-256-512-1024-2048_g32-64-128_20260507_142512.png
```

2B 结果完成了与 Stage 1A baseline 相同的 prompt/generation matrix。整体趋势与 Stage 1A 高度一致：TTFT 随 prompt length 增长，TPOT 约为 50 ms/token，PKV payload 与理论 KV cache 估算保持一致。

## Gemma 2 9B IT

状态：Tesla T4 / FP16 下模型加载阶段发生 CUDA OOM。

OOM CSV：

```text
gemma_2_9B_IT/csv/exp001B_v2_1_gemma-2-9b-it_p64-128-256-512-1024-2048_g32-64-128_20260507_150118.csv
```

9B 在进入 prompt/generation matrix 前已经加载失败。因此该结果应作为有效的部署边界记录，而不是 benchmark 程序失败。本次实验刻意不自动缩小 prompt_len、不改变精度、不启用量化、不使用 offload。

## 与 Experiment 001A 的关系

Experiment 001A 位于 `results/exp001/`，用于建立可信 Gemma 2 2B PKV baseline，并校准测量协议。

Experiment 001B 位于 `results/exp001b/`，用于在同一测量协议下观察模型规模增大时的 T4 FP16 部署边界。

因此：

- Experiment 001A 回答测量协议是否可靠；
- Experiment 001B 回答模型规模增大后部署边界从哪里开始出现。

## 当前结论

Gemma 2 2B IT FP16 在当前单 batch 测试矩阵下没有触及 Tesla T4 显存边界；Gemma 2 9B IT FP16 已在模型加载阶段触发 T4 FP16 部署边界。
