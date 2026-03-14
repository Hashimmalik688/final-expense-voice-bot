#!/usr/bin/env bash
# Start vLLM capped at ~10 GB VRAM so Parakeet TDT and Kokoro have plenty of
# headroom on the same GPU.
# GPU budget example (RTX 4090 24 GB):
#   vLLM KV cache + Llama-3.1-8B weights  ~9.8 GB   (0.40 × 24.6 GB)
#   Parakeet TDT 0.6B (CUDA)              ~1.7 GB
#   Kokoro TTS (CUDA)                     ~1.5 GB
#   CUDA context / overhead               ~0.6 GB
#   ─────────────────────────────────────────────
#   Total                                 ~13.6 GB  (10.4 GB headroom)

source /venv/main/bin/activate

exec python -m vllm.entrypoints.openai.api_server \
    --model /workspace/models/llama-awq \
    --served-model-name meta-llama/Llama-3.1-8B-Instruct \
    --quantization awq \
    --port 8000 \
    --dtype float16 \
    --gpu-memory-utilization 0.40 \
    --max-model-len 4096 \
    --max-num-seqs 16 \
    --enforce-eager
