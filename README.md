# EdgeLLM-HeteroLab
Research framework for edge LLM inference, KV cache optimization, and heterogeneous acceleration.

## Stage 1
LLM inference profiling on resource-constrained GPU environments.

Focus:
- Prefill latency
- Decode latency
- KV cache behavior
- Quantization impact
- Memory bottleneck analysis

## Current Target
Experiment 001:
Gemma baseline profiling on Colab

Metrics:
- TTFT
- TPOT
- tokens/s
- peak GPU memory
- KV cache estimation