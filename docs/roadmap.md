# docs/roadmap.md

# HeteroInfer-Lab 技术路线图

## 一、项目目标

HeteroInfer-Lab 聚焦于资源受限环境下的大模型推理系统研究，面向单卡 GPU、边缘服务器、小型工作站以及 FPGA、NPU 等异构硬件平台，围绕 LLM Inference 的真实性能瓶颈展开系统分析与优化。

项目重点关注以下方向：

* LLM Inference Profiling
* Prefill / Decode 性能分析
* KV Cache 管理与优化
* CUDA Decode Kernel Optimization
* GPU / FPGA / NPU 异构执行
* HLS / MLIR / FPGA Dataflow Generation

整体技术路线为：

> Profiling → KV Cache → Decode Optimization → Heterogeneous Execution → FPGA/HLS Dataflow Generation

核心目标是建立一套可复现、可扩展、具备持续研究价值的异构推理实验框架。

---

## 二、系统结构

项目长期分为三个核心模块：

### Module A：Measurement Layer（测量层）

负责系统基础性能测量与运行行为采集，包括：

* Benchmark
* Profiling
* Latency Measurement
* Memory Trace
* GPU Runtime Observation

主要任务是获取真实系统行为数据，建立后续分析工作的基础。

核心问题：

> 系统瓶颈具体出现在哪里？

---

### Module B：Analysis Layer（分析层）

负责系统瓶颈分析与性能解释，包括：

* KV Cache Analysis
* Memory Bottleneck Identification
* Decode Bottleneck Explanation
* Roofline-based Analysis
* System-level Performance Interpretation

主要任务是解释系统性能问题产生的原因。

核心问题：

> 为什么系统会在这里变慢？

---

### Module C：Optimization Layer（优化层）

负责面向瓶颈的优化设计与实现，包括：

* CUDA Optimization
* Decode Kernel Analysis
* FPGA / HLS Implementation
* MLIR-based Optimization
* Heterogeneous Execution Design

主要任务是将性能分析结果转化为可验证的优化方案。

核心问题：

> 哪些优化路径具有实际收益？

---

## 三、阶段推进路线

---

# Stage 1：Performance Profiling（当前阶段）

## Experiment 001-PKV：KV cache measurement correction with past_key_values

Current sub-stage: Experiment 001-PKV: KV cache measurement correction with past_key_values.

当前 baseline profiling 优先使用 Gemma 2 2B Instruct 版本完成实验框架搭建，Gemma 4 E2B 作为后续对照实验模型，优先保证第一阶段实验稳定性与可复现性，再逐步扩展至更复杂模型。在 Colab GPU 环境下建立基础 profiling 框架。

当前状态为 Experiment 001-PKV：KV cache measurement correction with `past_key_values`。Stage 1 仍处于 measurement protocol calibration 收尾阶段，重点是修正 KV Cache 测量协议，并确认 pure KV payload 与理论公式一致。

核心任务：

* Prefill latency 分析
* Decode latency 分析
* TTFT / TPOT 测量
* GPU 显存峰值统计
* KV Cache theoretical formula 与 `past_key_values` payload 对齐
* Prompt Length / Generation Length 影响分析

核心问题：

> 在资源受限 GPU 环境下，小型 decoder-only LLM 的 prefill latency、decode latency 与 KV Cache 显存开销将如何变化？

阶段目标：

建立第一版可复现的 baseline profiling framework。

Stage 1 exit criteria：

* baseline inference reproducible；
* TTFT / TPOT measured with repeated runs；
* theoretical KV cache and `past_key_values` payload are aligned；
* CUDA peak memory overhead is explained as non-KV runtime / allocator / temporary tensor overhead；
* `experiment_log` records the measurement correction process。

当前尚未正式进入 Stage 2。只有在上述条件满足后，才开始 Stage 2：KV Cache System Analysis。

---

# Stage 2：KV Cache System Analysis（Next Stage）

Stage 2 是下一阶段工作。在完成 Stage 1 measurement protocol calibration 并满足退出条件后，才进一步深入 KV Cache。

核心任务：

* KV Cache Memory Footprint Analysis
* KV Cache Layout Analysis
* Decode Memory Access Pattern
* KV Compression 可行性分析
* KV Migration / Offloading 思路验证

核心问题：

> Decode 阶段中，KV Cache 在多大程度上构成系统性能限制？

阶段目标：

明确 KV Cache 对系统延迟与显存占用的实际影响。

---

# Stage 3：CUDA + Decode Optimization

围绕 decode 阶段进入优化阶段。

核心任务：

* Decode Kernel Microbenchmark
* FlashAttention / Paged Attention 对比分析
* Memory-bound Kernel Analysis
* CUDA Decode Optimization
* Small-batch Inference Optimization

核心问题：

> 哪些 decode kernel 最值得进行硬件级优化？

阶段目标：

建立 GPU Profiling 与后续异构硬件实现之间的技术桥梁。

---

# Stage 3.5：NPU / Architecture Exploration

围绕专用推理架构进行验证性研究。

核心任务：

* NPU Decode Path Simulation
* Specialized Architecture Behavior Analysis
* GPU vs NPU Decode Bottleneck Comparison
* NPU-side KV Management Analysis

核心问题：

> 在专用推理架构中，Decode bottleneck 将如何变化？

阶段目标：

评估 NPU 架构在边缘侧 LLM Decode 阶段的适用性。

说明：

该阶段作为架构验证模块存在，不作为当前主线推进。

---

# Stage 4：FPGA / HLS / MLIR Extension

进入异构硬件实现阶段。

核心任务：

* Decode Kernel HLS Implementation
* FPGA Dataflow Mapping
* MLIR-based Optimization
* Automatic Dataflow Generation

核心问题：

> 如何将 decode kernel 映射到 FPGA，并形成高效 dataflow execution？

阶段目标：

形成完整的异构推理优化闭环。

---

## 四、当前执行重点

当前阶段唯一重点任务：

# 完成 Experiment 001-PKV measurement calibration

包括：

* 跑通 Gemma baseline inference
* 获取 repeated-run TTFT / TPOT 基础数据
* 记录 CUDA peak memory 与 pure KV payload
* 输出 PKV correction profiling 图表
* 建立结构化实验记录
* 固化 `past_key_values` payload 作为 pure KV cache measurement protocol

当前阶段不进入：

* 大规模 CUDA 优化
* FPGA Kernel 开发
* MLIR Pass 实现

所有后续工作均建立在第一阶段真实实验结果与校准后的测量协议之上。

---

## 五、执行原则

### 原则一：先测量，再优化

在未完成 profiling 之前，不进入复杂优化阶段。

性能分析是系统研究的前提。

---

### 原则二：保持单阶段推进

当前只推进 Stage 1，不并行启动多个方向。

避免研究范围失控。

---

### 原则三：以实验结果驱动研究

代码、工具和框架都服务于实验结果。

研究推进的核心资产是：

* 数据
* 图表
* 分析结论
* 可复现实验流程

而不是环境配置本身。

---

## 六、近期目标

近期目标集中于：

## 建立稳定、可复现的 LLM Profiling Baseline

完成第一版实验框架与 measurement protocol calibration 后，再逐步进入 KV Cache 深入分析与异构优化阶段。
