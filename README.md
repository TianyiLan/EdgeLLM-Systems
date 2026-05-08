# EdgeLLM-Systems

中文名：边缘大模型推理系统

A research-oriented system project for memory-constrained edge LLM inference, profiling, optimization, and heterogeneous acceleration.

EdgeLLM-Systems 是一个面向资源受限边缘环境的大模型推理系统研究项目，聚焦部署边界、性能瓶颈、软件优化与异构硬件加速。

项目关注 LLM（Large Language Model，大语言模型）在两类边缘平台上的真实系统行为：

1. **Host-centric Edge Platforms（主机式边缘平台）**  
   指以通用主机系统为中心，通过离散 GPU、FPGA 或其他加速卡执行本地大模型推理的平台，包括个人 PC、小型工作站、边缘服务器、工控机、本地私有化推理节点，以及 x86 / ARM 主机 + NVIDIA GPU / FPGA 加速卡等形态。

2. **SoC-integrated Edge Platforms（片上集成式边缘平台）**  
   指以 SoC（System on Chip，片上系统）为中心，将 CPU、GPU、NPU、DSP 和内存控制器等模块集成在同一片上系统中的低功耗边缘平台，包括手机、机器人、智能座舱、车载系统、Jetson / Orin 类嵌入式 AI 设备，以及 ARM + NPU 的端侧推理平台。

项目关注的核心问题是：

> 在有限显存、有限带宽、低 batch（批量）和低延迟约束下，如何让大模型稳定部署，并进一步提升推理效率。

项目强调基于真实实验数据进行分析与优化，逐步形成从性能测量、瓶颈识别、软件优化、异构硬件架构探索到 FPGA 编译器与数据流映射的完整研究链路。

---

## Research Scope

EdgeLLM-Systems 当前聚焦以下方向：

- LLM Inference Profiling（大模型推理性能分析）
- Prefill / Decode 阶段性能分析
- KV Cache（Key-Value Cache，键值缓存）测量、管理与优化
- Memory-bound Decode（受访存限制的解码阶段）行为分析
- CUDA（Compute Unified Device Architecture，英伟达并行计算平台）Decode Kernel 优化
- Host-centric Edge Platforms 上的 GPU / FPGA 异构执行
- SoC-integrated Edge Platforms 上的 CPU / GPU / NPU 异构执行
- HLS（High-Level Synthesis，高层次综合）与 FPGA Dataflow（数据流）生成
- MLIR（Multi-Level Intermediate Representation，多层中间表示）相关优化

整体研究路径为：

> Profiling → Bottleneck Analysis → GPU Software Optimization → Heterogeneous Hardware Architecture Exploration → FPGA Compiler & Dataflow Mapping

---

## Research Questions

本项目围绕以下问题展开：

1. **部署边界**  
   在资源受限平台上，模型权重、KV Cache、context length（上下文长度）和 batch size 如何共同决定模型能否运行？

2. **性能瓶颈**  
   Prefill（预填充）与 Decode（逐 token 解码）阶段的主要瓶颈分别来自计算、显存容量、访存带宽、kernel 实现还是运行时开销？

3. **KV Cache 影响**  
   KV Cache 在延迟、显存占用和带宽压力中分别占据什么位置？

4. **软件优化收益**  
   CUDA kernel、KV layout、PagedAttention、compression 和 runtime scheduling 等软件方法能够带来多少实际收益？

5. **FPGA 异构加速价值**  
   在 Host-centric Edge Platforms 中，FPGA 能否作为 GPU 的有效补充，缓解 Decode 阶段的 memory-bound bottleneck？

6. **NPU / SoC 异构推理价值**  
   在 SoC-integrated Edge Platforms 中，CPU / GPU / NPU 如何协同执行 LLM 推理，才能提升端侧部署能力和推理效率？

7. **方法论扩展**  
   如何将单个 FPGA 硬件加速案例提升为可复用的 FPGA dataflow mapping 与 compiler-assisted optimization 框架？

---

## Current Status

The project has completed Stage 1A baseline profiling and KV cache measurement protocol calibration with Gemma 2 2B IT on Google Colab Tesla T4.

Current Stage 1 work is now split into:

- **Stage 1A: Experiment 001A / PKV Modular Baseline**  
  Completed. The final trusted baseline uses `past_key_values` payload accounting as the primary pure KV cache metric. CUDA peak memory is treated as a system-level memory pressure metric rather than pure KV cache.

- **Stage 1B: Experiment 001B / Gemma Model-Scale Stress Baseline**  
  In progress. The current v2.1 results show that Gemma 2 2B IT FP16 completes the T4 test matrix and remains consistent with the Stage 1A baseline, while Gemma 2 9B IT FP16 triggers CUDA OOM during model loading. This provides early evidence for the T4 FP16 deployment boundary.

Key current artifacts:

- Stage 1A final CSV: `results/exp001/csv/exp001_results_pkv_modular.csv`
- Stage 1A final figure: `results/exp001/figures/exp001_profiling_results_pkv_modular.png`
- Stage 1B results: `results/exp001b/`

Stage 1 should not be described as fully closed until the remaining Stage 1B scope, especially whether to add Gemma 4 probing, is explicitly finalized.

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