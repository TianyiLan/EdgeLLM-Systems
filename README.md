# EdgeLLM-HeteroLab

面向边缘侧大模型推理部署、KV Cache 管理与异构硬件加速的研究型实验框架。

本项目的核心目标，不是简单完成“大模型部署”，而是通过系统化的 profiling、benchmark 与实验分析，研究资源受限环境下 LLM inference 的真实性能瓶颈，并逐步延伸到 KV Cache 优化、CUDA microbenchmark、FPGA/HLS decode kernel 设计，以及面向异构硬件的 dataflow 自动生成。

本项目首先是：

**科研型开源项目（Research-oriented Open Source Project）**

而不是：

**产品型工程项目（Product-oriented Engineering Project）**

重点是构建可复现、可扩展、可用于博士申请展示的科研成果。

---

# 一、研究背景

随着大模型（LLM）逐步从云端走向边缘侧部署，低资源设备（如单卡 GPU、小型工作站、边缘服务器、嵌入式平台）上的推理性能问题越来越重要。

与云端高吞吐场景不同，边缘侧更关注：

- 单请求延迟（Latency）
- 首 Token 时间（TTFT, Time To First Token）
- 单 Token 解码延迟（TPOT, Time Per Output Token）
- 显存占用（Memory Footprint）
- KV Cache 开销（KV Cache Overhead）
- Prefill / Decode 阶段性能差异
- 量化（Q4 / Q8 / FP16）带来的性能权衡

尤其在 Decode 阶段，KV Cache 的存储、读取、迁移与压缩，往往成为系统性能瓶颈。

因此，本项目希望从：

**GPU + CUDA + LLM inference profiling**

作为第一阶段切入点，而不是直接从 FPGA 开始。

因为：

- GPU 更接近真实系统瓶颈
- 能更快形成实验结果
- CUDA 是 AI System 的基础能力
- 后续 FPGA/HLS 优化必须建立在真实瓶颈分析之上

---

# 二、第一阶段目标（Stage 1）

## LLM Inference Profiling Framework

构建一个可复现的实验框架，用于分析：

- Prefill latency
- Decode latency
- KV Cache behavior
- Quantization impact
- GPU memory bottleneck
- 不同 prompt / output length 对性能的影响

重点研究问题：

> 在资源受限 GPU 环境下，小型 decoder-only LLM 的 prefill/decode latency 与 KV Cache 显存开销，如何随输入长度、输出长度和量化方式变化？

这是整个项目的第一阶段核心问题。

---

# 三、当前实验（Experiment 001）

## Gemma Baseline Profiling on Colab

使用 Google Gemma 小模型，在 Colab GPU 环境下建立 baseline profiling。

## 实验目标

建立边缘侧 LLM inference 的基础性能画像（baseline）。

## 主要指标

### 延迟指标

- TTFT（Time To First Token）
- TPOT（Time Per Output Token）
- Tokens/s（吞吐率）

### 显存指标

- Peak GPU Memory
- KV Cache Estimated Size

### 控制变量

- Prompt Length
- Generation Length
- Batch Size（后续）
- FP16 / INT8 / INT4（后续）

---

# 四、当前项目结构

```text
EdgeLLM-HeteroLab/
│
├── README.md
│
├── docs/
│   ├── roadmap.md
│   └── experiment_log.md
│
├── benchmarks/
│
├── scripts/
│
├── notebooks/
│
├── results/
│
└── requirements.txt