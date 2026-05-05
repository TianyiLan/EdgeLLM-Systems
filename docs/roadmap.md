# EdgeLLM-Systems 技术路线图

## 一、项目目标

EdgeLLM-Systems（边缘大模型推理系统）是一个面向资源受限边缘环境的大模型推理系统研究项目，聚焦部署边界、性能瓶颈、软件优化与异构硬件加速。

项目面向两类主要边缘平台：

1. **Host-centric Edge Platforms（主机式边缘平台）**  
   指以通用主机系统为中心，通过离散 GPU、FPGA 或其他加速卡执行本地大模型推理的平台，包括个人 PC、小型工作站、边缘服务器、工控机、本地私有化推理节点，以及 x86 / ARM 主机 + NVIDIA GPU / FPGA 加速卡等形态。

2. **SoC-integrated Edge Platforms（片上集成式边缘平台）**  
   指以 SoC（System on Chip，片上系统）为中心，将 CPU、GPU、NPU、DSP 和内存控制器等模块集成在同一片上系统中的低功耗边缘平台，包括手机、机器人、智能座舱、车载系统、Jetson / Orin 类嵌入式 AI 设备，以及 ARM + NPU 的端侧推理平台。

项目关注 LLM（Large Language Model，大语言模型）Inference（推理）在上述平台上的真实性能瓶颈与系统优化路径。

项目的核心问题是：

> 在有限显存、有限带宽、低 batch（批量）和低延迟约束下，如何让大模型稳定部署，并进一步提升推理效率。

重点研究方向包括：

- LLM Inference Profiling（大模型推理性能分析）
- Prefill / Decode 阶段性能分析
- KV Cache（Key-Value Cache，键值缓存）测量、管理与优化
- Memory-bound Decode（受访存限制的解码阶段）行为分析
- CUDA（Compute Unified Device Architecture，英伟达并行计算平台）Decode Kernel Optimization
- Host-centric Edge Platforms 上的 GPU / FPGA 异构执行
- SoC-integrated Edge Platforms 上的 CPU / GPU / NPU 异构执行
- HLS（High-Level Synthesis，高层次综合）与 FPGA Dataflow（数据流）生成
- MLIR（Multi-Level Intermediate Representation，多层中间表示）相关优化

整体技术路线为：

> Profiling → Bottleneck Identification → GPU Software Optimization → Heterogeneous Hardware Architecture Exploration → FPGA Compiler & Dataflow Mapping

核心目标是建立一套可复现、可扩展、具备持续研究价值的边缘大模型推理系统实验框架。

---

## 二、系统结构

### Module A：Measurement Layer（测量层）

Measurement Layer 负责系统基础性能测量与运行行为采集，包括：

- Benchmark
- Profiling
- Latency Measurement
- Memory Trace
- GPU Runtime Observation
- KV Cache Payload Measurement
- Reproducible Experiment Configuration

主要任务是获取真实系统行为数据，建立后续分析工作的基础。

核心问题：

> 系统当前的真实运行状态是什么？

---

### Module B：Analysis Layer（分析层）

Analysis Layer 负责系统瓶颈分析与性能解释，包括：

- Deployment Boundary Analysis（部署边界分析）
- KV Cache Analysis
- Memory Bottleneck Identification
- Decode Bottleneck Explanation
- Roofline-based Analysis（屋顶线分析）
- System-level Performance Interpretation
- Host-centric Edge Platforms 与 SoC-integrated Edge Platforms 的平台差异分析

主要任务是解释系统性能问题产生的原因，明确不同资源约束对部署边界和推理效率的影响。

核心问题：

> 为什么系统会在特定阶段出现容量压力、带宽压力或延迟瓶颈？

---

### Module C：Optimization Layer（优化层）

Optimization Layer 负责面向瓶颈的优化设计与实现，包括：

- CUDA Optimization
- Decode Kernel Analysis
- KV Cache Layout / Compression Optimization
- Runtime Scheduling Optimization
- FPGA / HLS Implementation
- NPU / SoC-side Architecture Exploration
- MLIR-based Optimization
- Heterogeneous Execution Design

主要任务是将性能分析结果转化为可验证的优化方案。

核心问题：

> 哪些优化路径具有实际收益？

---

### Module D：Compiler & Mapping Layer（编译器与映射层）

Compiler & Mapping Layer 负责将单点硬件加速经验提升为可复用方法，包括：

- FPGA Dataflow Mapping
- HLS Scheduling
- Loop Transformation（循环变换）
- Memory Hierarchy Mapping（存储层次映射）
- Pipeline Balancing（流水线平衡）
- Compiler-assisted Spatialization（编译器辅助空间化）
- MLIR-based Hardware Code Generation

主要任务是将具体 decode kernel 的硬件加速经验抽象为系统性方法论。

核心问题：

> 如何将单个 FPGA 加速案例推广为可复用的 dataflow mapping 与 compiler optimization 框架？

---

## 三、阶段推进路线

## Stage 1：Performance Characterization

### Experiment 001-PKV：KV Cache Measurement Calibration

#### 阶段目标

建立可信 baseline（基线），完成测量协议校准。

#### 核心任务

- Prefill latency 分析
- Decode latency 分析
- TTFT（Time To First Token，首 token 延迟）/ TPOT（Time Per Output Token，单 token 延迟）测量
- tokens/s 吞吐率测量
- GPU 显存峰值统计
- KV Cache theoretical formula 与 PKV（past_key_values）payload 对齐
- Prompt Length / Generation Length 影响分析

#### 核心问题

> 系统当前长什么样？

#### 阶段产出

建立第一版可复现的 baseline profiling framework，并固化可信 KV Cache 测量协议。

#### 当前结论

Experiment 001-PKV 已完成 KV Cache 测量协议校准，`past_key_values` payload 可作为 pure KV Cache 的直接测量方式。CUDA peak memory 不再被视为 pure KV Cache 指标，而是作为 system-level memory pressure（系统级显存压力）指标使用。

---

## Stage 2：Memory-Constrained Inference Analysis

#### 阶段目标

明确资源受限环境下限制系统部署与推理效率的核心 bottleneck（瓶颈），判断部署边界与优化优先级。

#### 核心任务

- Deployment Boundary Analysis
- KV Cache Memory Footprint Analysis
- Decode Memory Access Pattern
- Roofline-based Decode Analysis
- KV Compression 可行性分析
- KV Migration / Offloading（迁移 / 卸载）思路验证
- Alternative Implementation Comparison（FlashAttention / PagedAttention / vLLM）
- Host-centric Edge Platforms 与 SoC-integrated Edge Platforms 的部署差异建模

#### 核心问题

> Decode 阶段中，真正限制系统性能的核心 bottleneck 是什么？  
> KV Cache 在其中占据什么位置？  
> 当前问题首先体现为 capacity problem（容量问题）、bandwidth problem（带宽问题）、kernel implementation problem（算子实现问题），还是 runtime scheduling problem（运行时调度问题）？

#### 阶段产出

明确容量瓶颈、带宽瓶颈与实现瓶颈的边界，并建立后续优化路径的理论依据。

#### 说明

Stage 2 不直接进行优化，而是为 Stage 3 的软件优化、Stage 4 的异构硬件架构探索提供问题定义和实验依据。

Stage 2 与 Stage 3 共同构成第一条完整主线课题：

> 从问题识别到优化验证的完整闭环

---

## Stage 3：GPU Software Optimization

#### 阶段目标

在 Host-centric Edge Platforms 上验证 GPU 软件系统优化的实际收益。

#### 核心任务

- Decode Kernel Microbenchmark
- FlashAttention / PagedAttention 对比分析
- Memory-bound Kernel Analysis
- CUDA Decode Kernel Optimization
- Small-batch Inference Optimization
- KV Layout Redesign
- KV Cache Compression / Quantization
- Runtime Scheduling Optimization
- GPU-only baseline 与 optimized GPU path 对比

#### 核心问题

> 在不改变硬件的前提下，软件优化最多能解决多少问题？  
> 哪些 CUDA kernel、KV layout 或 runtime scheduling 策略最值得优先实施？

#### 阶段产出

完成第一轮可验证的软件优化闭环，明确 GPU 软件优化的收益上限与残余瓶颈。

#### 说明

Stage 3 的目的不是单纯展示 CUDA 优化，而是回答：

> 当系统瓶颈被识别后，现有 GPU 软件栈能否有效缓解这些瓶颈？

如果 Stage 3 后仍存在结构性限制，则进入 Stage 4 的异构硬件架构探索。

---

## Stage 4：Heterogeneous Hardware Architecture Exploration

Stage 4 是异构硬件架构探索层，用于评估专用硬件是否能够突破 GPU 平台的结构性限制。

该阶段不再将 FPGA 与 NPU 混合为单一方向，而是拆分为两个相对独立的 hardware track（硬件路线）。

---

### Track 4A：FPGA-based Host-centric Edge Acceleration

#### 阶段目标

验证 FPGA 是否能在 Host-centric Edge Platforms 上作为 GPU 的异构加速器，缓解 Decode 阶段的 memory-bound bottleneck。

#### 典型平台

- x86 / ARM host
- NVIDIA GPU
- FPGA accelerator card
- PCIe-based host-device interconnect
- Local edge inference server / workstation

#### 核心任务

- FPGA Decode Path Exploration
- GPU + FPGA 协同执行分析
- Decode Kernel Offloading
- KV Cache 数据搬移与缓存策略分析
- PCIe / host-device transfer 开销分析
- GPU-only vs GPU+FPGA 对比
- HLS kernel 原型实现
- FPGA-side memory bandwidth / on-chip buffer 利用分析

#### 核心问题

> 在 Host-centric Edge Platforms 上，FPGA 是否能作为 GPU 的有效补充，缓解 Decode 阶段的访存压力与延迟瓶颈？

#### 阶段产出

形成 FPGA-based heterogeneous decode acceleration 的架构验证结果，明确 FPGA 在主机式边缘平台中的收益、适用边界与工程代价。

---

### Track 4B：NPU-based SoC-integrated Edge Inference

#### 阶段目标

分析 SoC-integrated Edge Platforms 上 CPU / GPU / NPU 的异构协同机制，评估 NPU 对边缘侧 LLM 推理部署能力和推理效率的影响。

#### 典型平台

- Mobile SoC
- Embedded AI SoC
- ARM CPU + GPU + NPU
- Unified Memory Architecture（统一内存架构）
- Low-power edge AI device
- Jetson / Orin-like embedded platform

#### 核心任务

- NPU Decode Path Simulation
- CPU / GPU / NPU Operator Placement
- NPU Tensor Shape Sensitivity Analysis
- GPU-NPU Synchronization Overhead Analysis
- Unified Memory Behavior Analysis
- Embedded SoC-side Prefill / Decode Scheduling
- NPU-side KV Management Analysis
- Simulation-based Architecture Evaluation

#### 核心问题

> 在 SoC-integrated Edge Platforms 上，NPU、GPU、CPU 如何协同执行 LLM 推理，才能提升部署能力和端侧推理效率？

#### 阶段产出

形成 NPU-based embedded heterogeneous inference 的架构分析结果，明确 NPU 路线的适用场景、系统瓶颈与真实约束。

---

### Stage 4 总体说明

Stage 4 的核心目标是回答：

> 当 GPU 软件优化接近上限后，是否需要专用异构硬件来解决问题？  
> 如果需要，FPGA 与 NPU 分别适合解决哪类边缘部署问题？

Track 4A 和 Track 4B 对应两类不同边缘平台，不应混合评价：

- FPGA 路线主要服务 Host-centric Edge Platforms
- NPU 路线主要服务 SoC-integrated Edge Platforms

二者都属于异构硬件架构探索，但平台假设、实验方法、优化目标和工程约束不同。

---

## Stage 5：FPGA Compiler & Dataflow Mapping Framework

#### 阶段目标

在 Track 4A 的 FPGA 加速验证基础上，从单个 FPGA acceleration case（加速案例）走向系统性可复用的方法论。

#### 核心任务

- Decode Kernel HLS Implementation
- FPGA Dataflow Mapping
- MLIR-based Optimization
- Automatic Dataflow Generation
- Loop Transformation（循环变换）
- Memory Hierarchy Mapping（存储层次映射）
- Pipeline Balancing（流水线平衡）
- Compiler-assisted Spatialization（编译器辅助空间化）
- Design Space Exploration（设计空间探索）
- Hardware Code Generation

#### 核心问题

> 如何系统性地将 decode kernel 映射到 FPGA，并自动或半自动生成高效 dataflow execution（数据流执行）？

#### 阶段产出

形成完整的异构推理优化闭环与可复用 FPGA dataflow mapping 框架。

#### 说明

Stage 5 是方法论阶段，不再停留于单个 FPGA demo，而是形成 compiler + architecture 层面的系统成果。

Stage 5 主要承接 Track 4A，不强行覆盖 NPU / SoC 路线。NPU 方向若继续深入，可独立发展为 SoC-side heterogeneous runtime / scheduling framework，但不作为当前 Stage 5 的默认目标。

---

## 四、执行原则

### 原则一：先测量，再优化

在完成 profiling 与测量协议校准前，不进入复杂优化阶段。

性能分析是系统研究的前提。

---

### 原则二：问题驱动，工具服务问题

CUDA、FPGA、NPU、HLS 和 MLIR 均服务于具体系统问题。

所有优化都应建立在真实 bottleneck 的基础之上。

---

### 原则三：区分平台假设

Host-centric Edge Platforms 与 SoC-integrated Edge Platforms 是两类不同边缘平台。

不同平台的硬件约束、软件栈、内存结构、功耗限制和异构协同方式不同，不应混用同一套评价逻辑。

---

### 原则四：保持单阶段推进

当前只推进一个主阶段，其他方向保留规划与接口。

该原则用于降低研究范围失控风险，保证每一阶段形成明确产出。

---

### 原则五：以实验结果驱动研究

研究推进的核心资产包括：

- 数据
- 图表
- 分析结论
- Benchmark
- 可复现实验流程
- 可迁移的方法论

环境配置、工具链配置和工程脚手架均服务于上述资产。

---

## 五、近期目标

近期目标集中于：

## 完成 Stage 1 收尾并正式进入 Stage 2

包括：

- 固化 baseline profiling benchmark
- 完成 PKV measurement protocol 文档化
- 统一 README / roadmap / experiment log 状态
- 建立 Stage 2 的标准化实验入口
- 开始 Deployment Boundary 与 Decode Bottleneck 分析

后续所有 CUDA、FPGA、NPU 与 MLIR 工作，均建立在可信 baseline 与明确 bottleneck 的基础之上。