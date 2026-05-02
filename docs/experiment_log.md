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

本项目长期目标模型优先选择 Google Gemma 4 E2B，用于后续边缘侧推理系统分析与异构硬件优化研究。

考虑到当前阶段使用 Google Colab 免费版 Tesla T4（16GB 显存）进行 baseline profiling，优先保证实验的稳定性、可复现性与低成本验证，因此 Experiment 001 第一阶段采用 Gemma 2 2B Instruct 版本作为基础测试模型。

待 profiling framework 完成并验证稳定后，将进一步扩展至 Gemma 4 E2B 进行对照实验与系统分析。

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
- Colab baseline 环境准备；
- Gemma 模型选择与加载验证；
- 第一版 Profiling Script 设计。
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

### 10.实验记录

#### Experiment 001：Gemma 2-2B 基线性能 Profiling

**日期**：2025年5月  
**环境**：Google Colab 免费版 · Tesla T4 (16GB) · FP16  
**模型**：google/gemma-2-2b-it  
**框架**：HuggingFace Transformers + PyTorch  

---

#### 实验目的

建立边缘侧 LLM 推理的基线性能画像，为后续 KV Cache 优化、
decode kernel 分析、FPGA/HLS 映射提供可对比的参考基准。

---

#### 实验设计

**扫描参数矩阵**：

| 参数 | 取值 |
|---|---|
| Prompt Length | 64 / 128 / 256 / 512 tokens |
| Gen Length | 32 / 64 / 128 tokens |
| Batch Size | 1（固定） |
| 精度 | FP16（固定） |

共 12 组实验。

**测量指标**：TTFT、TPOT、Tokens/s、Peak GPU Memory、KV Cache 估算大小

**Warm-up 策略**：正式实验前跑 (64,32)、(256,64)、(512,32) 三组预热，
覆盖短/中/长序列的 CUDA kernel 编译，消除冷启动影响。

---

#### 实验结果

| prompt | gen | TTFT(ms) | TPOT(ms) | tok/s | peak(MB) | KV_est(MB) |
|---|---|---|---|---|---|---|
| 64 | 32 | 59.38 | 50.20 | 19.9 | 5065.4 | 9.75 |
| 64 | 64 | 74.23 | 62.03 | 16.1 | 5065.4 | 13.0 |
| 64 | 128 | 61.09 | 48.61 | 20.6 | 5065.4 | 19.5 |
| 128 | 32 | 62.69 | 48.42 | 20.7 | 5133.2 | 16.25 |
| 128 | 64 | 62.10 | 64.34 | 15.5 | 5133.2 | 19.5 |
| 128 | 128 | 60.62 | 49.69 | 20.1 | 5133.2 | 26.0 |
| 256 | 32 | 93.14 | 50.01 | 20.0 | 5273.7 | 29.25 |
| 256 | 64 | 92.48 | 60.25 | 16.6 | 5273.7 | 32.5 |
| 256 | 128 | 97.33 | 51.38 | 19.5 | 5273.7 | 39.0 |
| 512 | 32 | 190.05 | 49.25 | 20.3 | 5548.9 | 55.25 |
| 512 | 64 | 186.98 | 58.11 | 17.2 | 5548.9 | 58.5 |
| 512 | 128 | 185.24 | 51.65 | 19.4 | 5548.9 | 65.0 |

可视化图表见：`docs/figures/exp001_profiling_results.png`

---

#### 核心结论

**结论1：TTFT 随 prompt 长度线性增长**
64 tokens → ~60ms
128 tokens → ~62ms
256 tokens → ~94ms
512 tokens → ~187ms

符合理论预期：prefill 阶段对所有输入 token 并行计算，
计算量正比于序列长度，因此延迟线性增长。
TTFT 与 gen_len 无关（三条曲线基本重合）。

**结论2：TPOT 稳定，decode 是 memory-bandwidth bound**

所有 12 组实验的 TPOT 集中在 48~65ms，
与 prompt 长度和生成长度几乎无关。

这说明在当前实验范围内（序列总长 ≤ 640 tokens），
T4 的 decode 阶段瓶颈在于显存带宽，而非计算量。
每步 decode 只需读取一次 KV Cache，带宽跑满后
增加序列长度并不会显著影响单步延迟。

**结论3：Peak Memory 只随 prompt 增长，gen_len 影响可忽略**
prompt=64  → 5065 MB
prompt=128 → 5133 MB（+68MB）
prompt=256 → 5274 MB（+141MB）
prompt=512 → 5549 MB（+275MB）

同一 prompt 下不同 gen_len 的 peak_mem 完全相同，
说明在当前规模下，decode 阶段追加的 KV Cache 相对
模型权重（~4.5GB）可忽略不计。

**结论4：KV Cache 理论大小远小于实测峰值显存**

KV Cache 估算最大值为 65MB（prompt=512, gen=128），
而实测峰值显存约 5549MB，差值约 5484MB 为模型权重占用。
这个比例关系将在更长序列下发生变化——KV Cache 会成为
显存瓶颈，这是 Stage 2 的核心研究问题。

---

#### 遗留问题 / 下一步

- [ ] TPOT 在部分短 prompt 组合下存在轻微抖动（±15ms），
      原因待查（可能是 T4 调度机制，与序列长度无关）
- [ ] 当前只测了 FP16，INT8/INT4 量化下的性能变化留待后续
- [ ] prompt_len 上限只到 512，更长序列（1024/2048）的
      KV Cache 压力尚未测量 → Stage 2 重点
- [ ] Tokens/s 在 gen=64 时出现规律性下降，原因待分析

---

#### 引出 Stage 2 的问题

基于本次实验，Stage 2 的核心问题自然浮现：

> **当序列长度继续增长（1024/2048/4096 tokens）时，
> KV Cache 将从"可忽略"变成"显存主要占用者"，
> 此时 decode 的 memory-bandwidth bound 特征会如何演变？
> KV Cache 的布局和压缩方式能否缓解这一瓶颈？**

这是 Stage 2 KV Cache System Analysis 的出发点。
