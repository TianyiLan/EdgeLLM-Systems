# EdgeLLM-Systems 技术路线图

## 一、项目目标

EdgeLLM-Systems（边缘大模型推理系统）是一个面向资源受限边缘环境的大模型推理系统研究项目，聚焦部署边界、性能瓶颈、软件优化与异构硬件加速。项目面向单卡 GPU（Graphics Processing Unit，图形处理器）、边缘服务器、小型工作站、FPGA（Field Programmable Gate Array，现场可编程门阵列）、NPU（Neural Processing Unit，神经网络处理器）等异构硬件平台，围绕 LLM（Large Language Model，大语言模型）Inference（推理）的真实性能瓶颈展开系统分析与优化。

项目关注的核心问题是：

> 在有限显存、有限带宽、低 batch（批量）和低延迟约束下，如何让大模型稳定部署，并进一步提升推理效率。

重点研究方向包括：

* LLM Inference Profiling（大模型推理性能分析）
* Prefill / Decode 阶段性能分析
* KV Cache（Key-Value Cache，键值缓存）管理与优化
* CUDA（Compute Unified Device Architecture，英伟达并行计算平台）Decode Kernel Optimization
* GPU / FPGA / NPU 异构执行
* HLS（High-Level Synthesis，高层次综合）与 FPGA Dataflow（数据流）生成
* MLIR（Multi-Level Intermediate Representation，多层中间表示）相关优化

整体技术路线为：

> Profiling → Bottleneck Identification → Optimization Validation → Heterogeneous Execution → FPGA Compiler & Dataflow Mapping

核心目标是建立一套可复现、可扩展、具备持续研究价值的异构推理实验框架。

---

## 二、系统结构

### Module A：Measurement Layer（测量层）

Measurement Layer 负责系统基础性能测量与运行行为采集，包括：

* Benchmark
* Profiling
* Latency Measurement
* Memory Trace
* GPU Runtime Observation

主要任务是获取真实系统行为数据，建立后续分析工作的基础。

核心问题：

> 系统当前的真实运行状态是什么？

---

### Module B：Analysis Layer（分析层）

Analysis Layer 负责系统瓶颈分析与性能解释，包括：

* Deployment Boundary Analysis（部署边界分析）
* KV Cache Analysis
* Memory Bottleneck Identification
* Decode Bottleneck Explanation
* Roofline-based Analysis（屋顶线分析）
* System-level Performance Interpretation

主要任务是解释系统性能问题产生的原因，明确不同资源约束对部署边界和推理效率的影响。

核心问题：

> 为什么系统会在特定阶段出现容量压力、带宽压力或延迟瓶颈？

---

### Module C：Optimization Layer（优化层）

Optimization Layer 负责面向瓶颈的优化设计与实现，包括：

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

## Stage 1：Performance Characterization

### Experiment 001-PKV：KV Cache Measurement Calibration

#### 阶段目标

建立可信 baseline（基线），完成测量协议校准。

#### 核心任务

* Prefill latency 分析
* Decode latency 分析
* TTFT（Time To First Token，首 token 延迟）/ TPOT（Time Per Output Token，单 token 延迟）测量
* GPU 显存峰值统计
* KV Cache theoretical formula 与 PKV（past_key_values）payload 对齐
* Prompt Length / Generation Length 影响分析

#### 核心问题

> 系统当前长什么样？

#### 阶段产出

建立第一版可复现的 baseline profiling framework。

---

## Stage 2：Memory-Constrained Inference Analysis

#### 阶段目标

明确资源受限环境下限制系统部署与推理效率的核心 bottleneck（瓶颈），判断部署边界与优化优先级。

#### 核心任务

* Deployment Boundary Analysis
* KV Cache Memory Footprint Analysis
* Decode Memory Access Pattern
* Roofline-based Decode Analysis
* KV Compression 可行性分析
* KV Migration / Offloading（迁移 / 卸载）思路验证
* Alternative Implementation Comparison（FlashAttention / PagedAttention / vLLM）

#### 核心问题

> Decode 阶段中，真正限制系统性能的核心 bottleneck 是什么？
> KV Cache 在其中占据什么位置？

#### 阶段产出

明确容量瓶颈、带宽瓶颈与实现瓶颈的边界，并建立后续优化路径的理论依据。

#### 说明

Stage 2 与 Stage 3 共同构成第一条完整主线课题：

> 从问题识别到优化验证的完整闭环

---

## Stage 3：Software Optimization

#### 阶段目标

在 GPU 平台上验证软件系统优化的实际收益。

#### 核心任务

* Decode Kernel Microbenchmark
* FlashAttention / PagedAttention 对比分析
* Memory-bound Kernel Analysis
* CUDA Decode Optimization
* Small-batch Inference Optimization
* KV Layout Redesign
* Runtime Scheduling Optimization

#### 核心问题

> 哪些软件级优化最值得优先实施？

#### 阶段产出

完成第一轮可验证的软件优化闭环。

---

## Stage 3.5：Architecture Branch（异构架构验证）

#### 阶段目标

验证专用异构架构是否能够突破 GPU 平台的结构性限制。

#### 核心任务

* FPGA Decode Path Exploration
* GPU + FPGA 协同执行分析
* GPU vs FPGA Decode Bottleneck Comparison
* NPU Decode Path Simulation
* Specialized Architecture Behavior Analysis
* Memory-centric Architecture Exploration

#### 核心问题

> 当 GPU 优化接近上限后，是否需要专用异构架构来解决问题？

#### 阶段产出

完成架构级验证，明确 FPGA / NPU 的实际收益与适用边界。

#### 说明

该阶段属于主线中的 architecture branch（架构分支），用于连接软件优化结果与后续 FPGA 编译器 / 数据流映射研究。

该阶段具备独立形成 architecture / system paper（体系结构 / 系统论文）的潜力。

---

## Stage 4：FPGA Compiler & Dataflow Mapping Framework

#### 阶段目标

从单个 FPGA acceleration case（加速案例）走向系统性可复用的方法论。

#### 核心任务

* Decode Kernel HLS Implementation
* FPGA Dataflow Mapping
* MLIR-based Optimization
* Automatic Dataflow Generation
* Loop Transformation（循环变换）
* Memory Hierarchy Mapping（存储层次映射）
* Pipeline Balancing（流水线平衡）
* Compiler-assisted Spatialization（编译器辅助空间化）

#### 核心问题

> 如何系统性地将 decode kernel 映射到 FPGA，并自动生成高效 dataflow execution（数据流执行）？

#### 阶段产出

形成完整的异构推理优化闭环与可复用方法框架。

#### 说明

Stage 4 是方法论阶段，目标是形成 compiler + architecture 层面的系统成果，将单点硬件加速经验提升为可迁移、可复用的 FPGA dataflow mapping 框架。

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

### 原则三：保持单阶段推进

当前只推进一个主阶段，其他方向保留规划与接口。

该原则用于降低研究范围失控风险，保证每一阶段形成明确产出。

---

### 原则四：以实验结果驱动研究

研究推进的核心资产包括：

* 数据
* 图表
* 分析结论
* Benchmark
* 可复现实验流程
* 可迁移的方法论

环境配置、工具链配置和工程脚手架均服务于上述资产。

---

## 五、近期目标

近期目标集中于：

## 完成 Stage 1 收尾并正式进入 Stage 2

包括：

* 固化 baseline profiling benchmark
* 完成 PKV measurement protocol 文档化
* 统一 README / roadmap / experiment log 状态
* 建立 Stage 2 的标准化实验入口
* 开始 Deployment Boundary 与 Decode Bottleneck 分析

后续所有 CUDA、FPGA 与 MLIR 工作，均建立在可信 baseline 与明确 bottleneck 的基础之上。
