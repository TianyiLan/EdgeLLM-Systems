# README.md

# HeteroInfer-Lab

面向边缘侧大模型推理的异构推理系统研究项目。

本项目聚焦资源受限环境（单卡 GPU、边缘服务器、小型工作站、FPGA、NPU 等）中的大模型推理性能问题，围绕 LLM Inference 的真实性能瓶颈展开系统性研究。

核心研究方向包括：

* Prefill / Decode 性能分析
* KV Cache 管理与优化
* CUDA Decode Kernel Optimization
* GPU / FPGA / NPU 异构执行
* FPGA HLS 与 Dataflow Generation
* MLIR-based Optimization

项目整体研究路径为：

> Profiling → KV Cache → Decode Optimization → Heterogeneous Execution → FPGA/HLS Dataflow Generation

目标是构建可持续迭代的研究型系统项目，形成可复现、可验证、可扩展的系统研究成果。

---

# 当前 Focus

## Experiment 001：Gemma Baseline Profiling

当前阶段的核心任务，是建立第一版 LLM inference profiling framework。Experiment 001 仍处于 Stage 1 的 measurement calibration 阶段，当前重点是收尾并固化可信的测量协议，而不是进入 Stage 2 优化分析。

使用 Google Gemma 系列小模型进行 baseline profiling。长期目标模型优先选择 Gemma 4 E2B，用于后续系统分析与异构硬件优化研究。考虑到当前阶段使用 Colab 免费版 Tesla T4（16GB 显存）进行实验，Experiment 001 第一阶段优先采用Gemma 2 2B Instruct 版本建立稳定、可复现的 profiling baseline。待实验框架稳定后，再扩展至 Gemma 4 E2B 进行对照分析。
重点分析：

* Prefill latency
* Decode latency
* TTFT / TPOT
* GPU 显存峰值
* KV Cache payload 开销
* Prompt Length / Generation Length 对性能的影响

最近一次校准中，原来的 CUDA memory-delta based KV cache measurement 被证明会混入大量 non-KV runtime overhead，包括 allocator 行为、临时 activation、logits、attention workspace 等，因此实测值显著高于理论 KV Cache 大小。当前已改用 `past_key_values` payload measurement：直接统计 K/V tensors 的 `numel × element_size`，作为 pure KV cache measurement。

当前核心发现：

* PKV measured KV cache 与 theoretical formula 基本一致；
* CUDA peak memory 显著大于纯 KV payload；
* peak memory 适合作为 system-level memory pressure 指标，但不适合作为 pure KV cache measurement；
* prefill PKV 小于 final PKV 是合理的，因为 prefill 只包含 prompt tokens，而 final PKV 包含 prompt + generated tokens；
* Stage 1 measurement protocol calibration is being finalized。

核心研究问题：

> 在资源受限 GPU 环境下，小型 decoder-only LLM 的 prefill latency、decode latency 与 KV Cache 显存开销将如何变化？

当前阶段重点放在：

* 测量（Measure）
* 记录（Record）
* 分析（Analyze）
* 可视化（Visualize）
* 瓶颈识别（Identify Bottlenecks）

---

# 当前阶段暂不涉及

为保证研究目标清晰，第一阶段暂不涉及：

* 模型训练
* 模型微调（Fine-tuning）
* RAG 系统
* Agent 系统
* Web 服务与前端 UI
* Docker 工程化部署
* FPGA kernel 实现
* MLIR Pass 开发

当前重点是建立真实、可复现的 baseline，而不是过早进入复杂优化阶段。

---

# 项目结构

```text id="l0yr5c"
HeteroInfer-Lab/
│
├── README.md
│
├── heteroinfer_lab/
│   ├── models.py
│   ├── prompts.py
│   ├── metrics.py
│   ├── memory.py
│   ├── kv_cache.py
│   └── cuda_utils.py
│
├── docs/
│   ├── roadmap.md
│   └── experiment_log.md
│
├── benchmarks/
│   ├── configs/
│   └── profiling/
│
├── scripts/
│
├── notebooks/
│
├── results/
│
└── requirements.txt
```

---

# 项目定位

本项目定位为面向资源受限 LLM 推理的系统研究与实验平台，核心目标包括：

* 构建持续迭代的研究型系统项目
* 形成可复现的实验结果与系统分析能力
* 建立 AI System 与异构计算方向的研究基础
* 为后续性能分析、系统优化与异构硬件实验提供基础

研究工作的重点，在于持续形成具有明确价值和可验证结果的技术资产。
