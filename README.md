# EdgeLLM-HeteroLab

面向边缘侧大模型推理部署、KV Cache 管理与异构硬件加速的研究型实验框架。

本项目聚焦资源受限环境（单卡 GPU、边缘服务器、小型工作站等）中的大模型推理性能问题，围绕 LLM inference 的实际瓶颈展开系统性研究，重点关注：

* Prefill latency 的增长规律
* Decode latency 的主要影响因素
* KV Cache 的显存占用与带宽压力
* 不同量化方式（FP16 / INT8 / INT4）带来的性能变化
* Decode kernel 的 CUDA 优化潜力与 FPGA/HLS 映射可行性

在此基础上，项目将逐步扩展至：

* CUDA microbenchmark
* KV Cache Compression / Migration
* GPU / FPGA 异构执行
* Decode Kernel 的 HLS 实现
* 面向边缘侧 LLM 的 Heterogeneous Inference System

本项目定位为长期持续迭代的科研型开源项目，目标是形成可复现、可扩展、具备博士申请展示价值的系统性研究成果。

---

# 一、当前核心任务（Current Focus）

## Experiment 001：Gemma Baseline Profiling Framework

第一阶段的重点是建立基础实验框架，围绕一个明确问题开展分析：

> 在资源受限 GPU 环境下，小型 decoder-only LLM 的 prefill latency、decode latency 与 KV Cache 显存开销，将如何随着输入长度（Prompt Length）和输出长度（Generation Length）变化？

这一问题构成整个项目的起点，也是后续 CUDA 优化、KV Cache 管理和 FPGA/HLS 设计的基础。

当前阶段的主要任务是：

* 建立可复现的 profiling 流程
* 获取结构化实验数据
* 输出性能图表与分析结论
* 明确系统中的主要瓶颈位置

研究工作的前提是对问题本身有准确判断，因此第一阶段的重点放在测量与分析，而不是直接进入优化阶段。

---

# 二、从 GPU Profiling 开始的原因

项目第一阶段选择：

## GPU + CUDA + LLM Inference Profiling

主要考虑如下：

* GPU 更接近真实部署环境中的性能瓶颈
* 实验周期更短，更容易形成初步结果
* CUDA 是后续 AI System 研究的重要基础能力
* FPGA/HLS 优化需要建立在真实性能分析之上

如果无法明确：

* 延迟主要集中在哪个阶段
* 显存瓶颈出现在哪里
* KV Cache 是否构成核心限制因素

那么后续硬件优化工作很容易偏离真正的问题。

Profiling 的意义就在于建立这一层判断依据。

---

# 三、当前实验（Experiment 001）

## Gemma Baseline Profiling on Colab

使用 Google Gemma 小模型（优先选择 Gemma 4 E2B），在 Colab GPU 环境下完成 baseline profiling。

## 实验目标

建立边缘侧 LLM inference 的基础性能画像，重点分析：

* Prefill latency
* Decode latency
* KV Cache behavior
* GPU Memory bottleneck

通过基础实验，形成对小模型推理行为的初步认识，并为后续扩展实验提供基线参考。

---

## 主要观测指标

### 延迟指标

* TTFT（Time To First Token）
* TPOT（Time Per Output Token）
* Tokens/s（吞吐率）

### 显存指标

* Peak GPU Memory
* GPU Memory Allocation
* KV Cache Estimated Size

### 控制变量

* Prompt Length
* Generation Length
* Batch Size（后续阶段）
* Quantization（后续阶段）

---

# 四、当前阶段的研究边界

为保证研究工作具备明确目标，第一阶段暂不涉及以下内容：

* 模型训练
* 模型微调（Fine-tuning）
* RAG 系统
* Agent 系统
* 聊天机器人产品开发
* 前端 UI 与 Web 服务
* Docker 工程化部署
* FPGA kernel 实现
* MLIR Pass 开发

当前阶段集中于：

* 测量（Measure）
* 记录（Record）
* 分析（Analyze）
* 可视化（Visualize）
* 瓶颈识别（Identify Bottlenecks）

重点在于理解系统行为，而非过早进入复杂优化。

---

# 五、当前项目结构

```text id="oq5ckl"
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
```

---

# 六、后续研究方向（Stage 2+）

在完成 Experiment 001 后，项目将继续推进：

* FlashAttention / Paged Attention 分析
* KV Cache Compression
* KV Cache Migration
* Decode Kernel CUDA Profiling
* GPU / FPGA 异构执行
* Decode Kernel 的 HLS 实现
* FPGA Dataflow 自动生成
* MLIR / Polyhedral Optimization

并进一步与以下方向形成研究衔接：

* POM
* CODO
* MLIR-based FPGA Compiler
* FPGA Dataflow Optimization

整体推进方式遵循逐步深入原则，以实验结果为基础持续扩展研究范围。

---

# 七、项目定位

本项目服务于博士申请阶段的长期科研积累，核心目标包括：

* 构建持续迭代的独立研究项目
* 形成可展示的科研成果与系统能力
* 建立 AI System 与异构计算方向的研究基础
* 为后续学术研究与独立开发提供长期支撑

研究工作的重点不在于短期完成多少功能，而在于持续形成具有明确价值和可验证结果的技术资产。
