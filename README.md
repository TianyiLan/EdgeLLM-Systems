# EdgeLLM-Systems

中文名：边缘大模型推理系统

A research-oriented system project for memory-constrained edge LLM inference, profiling, optimization, and heterogeneous acceleration.

EdgeLLM-Systems 是一个面向资源受限边缘环境的大模型推理系统研究项目，聚焦部署边界、性能瓶颈、软件优化与异构硬件加速。项目关注 LLM（Large Language Model，大语言模型）在单卡 GPU（Graphics Processing Unit，图形处理器）、边缘服务器、小型工作站、FPGA（Field Programmable Gate Array，现场可编程门阵列）、NPU（Neural Processing Unit，神经网络处理器）等平台上的真实系统行为。
项目关注的核心问题是：

> 在有限显存、有限带宽、低 batch（批量）和低延迟约束下，如何让大模型稳定部署，并进一步提升推理效率。

项目强调基于真实实验数据进行分析与优化，逐步形成从性能测量、瓶颈识别、软件优化到异构硬件映射的完整研究链路。

---

## Research Scope

EdgeLLM-Systems 当前聚焦以下方向：

* LLM Inference Profiling（大模型推理性能分析）
* Prefill / Decode 阶段性能分析
* KV Cache（Key-Value Cache，键值缓存）测量、管理与优化
* Memory-bound Decode（受访存限制的解码阶段）行为分析
* CUDA（Compute Unified Device Architecture，英伟达并行计算平台）Decode Kernel 优化
* GPU / FPGA / NPU 异构执行
* HLS（High-Level Synthesis，高层次综合）与 FPGA Dataflow（数据流）生成
* MLIR（Multi-Level Intermediate Representation，多层中间表示）相关优化

整体研究路径为：

> Profiling → Bottleneck Analysis → Software Optimization → Heterogeneous Execution → FPGA Compiler & Dataflow Mapping

---

## Research Questions

本项目围绕以下问题展开：

1. **部署边界**：在资源受限平台上，模型权重、KV Cache、context length（上下文长度）和 batch size 如何共同决定模型能否运行？
2. **性能瓶颈**：Prefill（预填充）与 Decode（逐 token 解码）阶段的主要瓶颈分别来自计算、显存容量、访存带宽还是运行时开销？
3. **KV Cache 影响**：KV Cache 在延迟、显存占用和带宽压力中分别占据什么位置？
4. **软件优化收益**：CUDA kernel、KV layout、PagedAttention、compression 和 runtime scheduling 等软件方法能够带来多少实际收益？
5. **异构硬件价值**：FPGA / NPU 等专用硬件能否突破 GPU 平台的结构性限制？
6. **方法论扩展**：如何将单个硬件加速案例提升为可复用的 FPGA dataflow mapping 与 compiler-assisted optimization 框架？

---

## Current Status

项目已完成第一阶段 baseline profiling 与 KV Cache 测量协议校准，后续任务状态以 `docs/roadmap.md`、`docs/experiment_log.md` 以及 GitHub Issues 为准。

当前已完成的关键基础包括：

* 建立 Gemma 2 2B Instruct 在 Colab + Tesla T4 环境下的 baseline profiling 流程
* 测量 TTFT（Time To First Token，首 token 延迟）、TPOT（Time Per Output Token，单 token 延迟）、tokens/s、peak GPU memory 等基础指标
* 修正原 CUDA memory-delta KV Cache 测量方法中的 non-KV runtime overhead 问题
* 使用 `past_key_values` payload 作为 pure KV cache measurement protocol
* 验证 PKV measured KV cache 与 theoretical formula 高度一致
* 将 CUDA peak memory 明确区分为 system-level memory pressure 指标

---

## Project Structure

```text
EdgeLLM-Systems/
│
├── README.md
│
├── edge_llm_systems/
│   ├── models.py          # 模型加载与配置
│   ├── prompts.py         # Prompt 构造与控制变量生成
│   ├── metrics.py         # TTFT / TPOT / tokens/s 等指标计算
│   ├── memory.py          # GPU 显存测量与运行时观测
│   ├── kv_cache.py        # KV Cache 理论估算与 PKV payload 测量
│   └── cuda_utils.py      # CUDA 环境与同步辅助函数
│
├── docs/
│   ├── roadmap.md         # 技术路线图
│   └── experiment_log.md  # 实验日志与阶段性结论
│
├── benchmarks/
│   ├── configs/           # Benchmark 配置文件
│   └── profiling/         # 标准化 profiling 入口
│
├── scripts/               # 绘图、汇总、结果整理等辅助脚本
│
├── notebooks/             # 探索性实验与可视化 notebook
│
├── results/               # 实验数据、图表与结果说明
│
└── requirements.txt
```

---

## Methodology

本项目采用分阶段推进方式。

### 1. Measurement

建立可复现的测量流程，统一输入、指标和输出格式，确保不同实验之间能够横向比较。

### 2. Characterization

分析模型结构、prompt length、generation length、batch size、dtype、KV Cache 和运行时行为对推理性能的影响。

### 3. Bottleneck Identification

结合实验数据、理论公式和 roofline analysis（屋顶线分析）判断系统瓶颈的来源，包括显存容量、访存带宽、kernel 实现和运行时调度。

### 4. Optimization Validation

在 GPU 平台上验证 CUDA kernel、KV Cache compression、layout redesign、PagedAttention、offloading 等优化路径的实际收益。

### 5. Heterogeneous Realization

探索 FPGA / NPU 等异构硬件对 Decode 阶段的适用性，并进一步研究 FPGA dataflow mapping、HLS 与 MLIR-based compiler optimization。

---

## Completed Baseline: Experiment 001-PKV

Experiment 001-PKV 用于校准 KV Cache 测量协议，并建立第一版可信 baseline。

主要结论：

* `past_key_values` payload 可以作为 pure KV Cache 的直接测量方式
* theoretical KV Cache formula 与 PKV measurement 基本一致
* CUDA peak memory 包含 allocator、temporary activation、logits、attention workspace 等运行时开销
* peak memory 适合作为系统级显存压力指标
* Prompt length 显著影响 TTFT 与 peak memory
* TPOT 在当前实验区间内相对稳定，可作为后续 Decode 分析的 baseline

相关结果位于：

```text
results/exp001/
```

---

## Development Principles

### 1. Measurement before optimization

所有优化工作均基于可复现的 profiling 数据展开。

### 2. Problem-driven implementation

CUDA、FPGA、NPU、HLS 和 MLIR 均服务于具体系统问题，优先级由实验结论决定。

### 3. Reproducibility first

实验代码、配置、数据和图表应保持结构化，便于复现、复查和后续扩展。

### 4. Minimal but extensible engineering

项目工程结构保持简洁，同时保留扩展至 CUDA microbenchmark、FPGA kernel、NPU simulation 和 compiler framework 的接口。

---

## Non-goals

当前项目不聚焦以下方向：

* 模型训练
* 通用模型微调
* RAG（Retrieval-Augmented Generation，检索增强生成）应用系统
* Agent 应用框架
* Web 前端或产品化服务
* 通用 Docke
