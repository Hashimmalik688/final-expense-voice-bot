#!/usr/bin/env bash
# Start vLLM at 0.85 GPU utilization so Parakeet TDT can run on CUDA alongside it.
# GPU budget (RTX 4090 24 GB):
#   vLLM KV cache + Llama-3.1-8B weights  ~20.9 GB  (0.85 × 24.6 GB)
#   Parakeet TDT 0.6B (CUDA)              ~1.7 GB
#   CUDA context / overhead               ~0.6 GB
#   ─────────────────────────────────────────────
#   Total                                 ~23.2 GB  (1.4 GB headroom)

source /venv/main/bin/activate

exec python -m vllm.entrypoints.openai.api_server \
    --model /workspace/models/llama \
    --served-model-name meta-llama/Llama-3.1-8B-Instruct \
    --port 8000 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 2048 \
    --max-num-seqs 32 \
    --enforce-eager
