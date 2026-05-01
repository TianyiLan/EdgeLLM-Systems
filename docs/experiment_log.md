# Experiment Log

## Experiment 001: Gemma Baseline Profiling

### 1. 实验背景（Background）

本项目聚焦于边缘侧大模型推理系统研究。当前第一阶段的核心任务是建立基础性能画像（Baseline Profiling），为后续系统优化提供可复现、可量化的基准依据。

在进入 CUDA 优化、KV Cache 管理以及 FPGA/HLS 实现之前，需要首先明确以下问题：

- 推理延迟主要集中于哪个阶段；
- Prefill 与 Decode 阶段之间的性能差异；
- KV Cache 在显存占用中的比例；
- Prompt Length 与 Generation Length 对系统性能的影响。

如果缺乏对系统真实瓶颈的明确判断，后续优化工作将难以形成有效目标。因此，Experiment 001 的主要任务是建立第一版可复现的 LLM inference profiling baseline。

### 2. 实验目标（Objective）

本实验计划使用 Google Gemma 小模型（优先选择 Gemma 4 E2B），在资源受限的单卡 GPU 环境下完成基础 profiling。

重点分析指标包括：

- Prefill latency；
- Decode latency；
- TTFT（Time To First Token）；
- TPOT（Time Per Output Token）；
- GPU 显存峰值；
- KV Cache 基础开销。

同时，实验将观察以下变量对系统性能的影响：

- Prompt Length；
- Generation Length。

本实验阶段不以优化为目标，而是以建立系统行为的基础认知和性能基准为主要目标。

### 3. 实验环境（Environment）

#### 当前实际运行环境（Colab Baseline）

运行平台：

- Google Colab（免费版）

GPU：
- Tesla T4
- CUDA Available: True
- CUDA Version: 12.8
- GPU Memory: 16 GB

软件环境：
- Python: 3.12.13
- PyTorch: 2.10.0+cu128
- Transformers: 5.0.0

说明：
当前实验使用 Colab 免费版 T4 GPU 作为 baseline profiling 平台。  
该环境能够稳定支持 Gemma 小模型推理实验，并具备后续 TTFT、TPOT、显存峰值及 KV Cache 分析所需的基础条件。

本实验后续所有 baseline 数据，均以该环境作为统一测试平台。


#### 模型

- Google Gemma（优先选择 Gemma 4 E2B）；
- 若环境兼容性或资源限制导致模型无法稳定运行，则使用 Gemma 2B / 4B 作为替代模型。

#### 工具

- PyTorch；
- Hugging Face Transformers；
- CUDA Runtime Observation；
- GPU Memory Monitoring；
- Python Profiling Scripts。

#### 预期输出

- latency 数据表；
- memory usage 数据表；
- profiling 图表；
- baseline analysis summary。

### 4. 核心观测指标（Metrics）

#### 4.1 延迟指标

**TTFT（Time To First Token）**

首个 Token 的输出延迟，主要用于衡量 Prefill 阶段的性能表现。

**TPOT（Time Per Output Token）**

平均每个输出 Token 的生成延迟，主要用于衡量 Decode 阶段的性能表现。

**Tokens/s**

整体吞吐率，用于衡量系统整体推理效率。

#### 4.2 显存指标

**Peak GPU Memory**

推理过程中的显存峰值，用于衡量系统整体资源占用情况。

**KV Cache Estimated Size**

KV Cache 的基础显存估算值，用于判断 Decode 阶段是否主要受 KV Cache 限制。

### 5. 控制变量（Variables）

#### 5.1 Prompt Length

计划测试以下输入长度：

- Short Prompt；
- Medium Prompt；
- Long Prompt。

该变量用于观察输入长度对 Prefill latency 与显存占用的影响。

#### 5.2 Generation Length

计划测试以下输出长度：

- Short Generation；
- Medium Generation；
- Long Generation。

该变量用于观察输出长度对 Decode latency 与 KV Cache 增长的影响。

### 6. 当前状态（Current Status）

#### 已完成

- GitHub 仓库初始化；
- 项目结构建立；
- README 编写；
- roadmap 制定；
- 技术路线明确；
- Experiment 001 定义完成。

#### 当前进行中

- Colab baseline 环境准备；
- Gemma 模型选择与加载验证；
- 第一版 Profiling Script 设计。

#### 尚未开始

- 正式实验数据采集；
- profiling 图表输出；
- 第一版结果分析。

当前阶段重点是完成第一组 baseline 数据采集。

### 7. 预期问题（Expected Challenges）

#### 7.1 显存限制

Colab 免费版 GPU 显存有限，可能导致以下问题：

- 模型加载失败；
- Prompt 长度受限；
- Batch Size 受限。

应对策略：优先采用小模型与单 batch 配置完成 baseline 测试。

#### 7.2 测量误差

单次实验结果可能受到以下因素影响：

- Runtime 抖动；
- Colab 后台资源波动；
- CUDA 初始化开销。

应对策略：进行多次重复实验，并使用平均值或稳定区间作为分析依据。

#### 7.3 KV Cache 难以直接观测

KV Cache 显存通常需要通过间接方式估算，而非直接读取。

应对策略：结合理论估算结果与实际显存变化进行综合分析。

### 8. 下一步计划（Next Steps）

下一阶段任务如下：

1. 完成 Gemma baseline inference 跑通。
2. 记录第一组基础数据，包括 TTFT、TPOT 与 Peak GPU Memory。
3. 建立 Prompt Length / Generation Length 对比实验。
4. 输出第一版 profiling 图表。
5. 进入 Stage 2：KV Cache System Analysis，继续深入分析系统瓶颈。

### 9. 当前原则（Guiding Principle）

当前阶段只聚焦一项任务：完成 Experiment 001。

在 baseline profiling 完成之前，暂不提前进入以下工作：

- CUDA 深度优化；
- FPGA Kernel 开发；
- MLIR Pass 实现。

研究推进遵循以下原则：

> 先测量，再优化。
