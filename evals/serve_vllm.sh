#!/usr/bin/env bash
# Start a vLLM OpenAI-compatible server for evals (run on the GPU server).
#
# Usage: evals/serve_vllm.sh [MODEL_DIR]
#   MODEL_DIR defaults to /workspace/models/Qwen3-30B-A3B. Point it at a fake-quantized checkpoint
#   (same layout, BF16 weights holding quantized values) to evaluate a quantization variant.
#
# Env overrides: PORT (8000), MAX_MODEL_LEN (32768), GPU_UTIL (0.90)
# The served name is fixed so run_evals.py doesn't change between variants.
set -euo pipefail

MODEL_DIR=${1:-/workspace/models/Qwen3-30B-A3B}

# --generation-config vllm: ignore the model's default sampling settings so temperature 0 is pure greedy
exec vllm serve "$MODEL_DIR" \
  --served-model-name qwen3-30b-a3b \
  --host 127.0.0.1 \
  --port "${PORT:-8000}" \
  --dtype bfloat16 \
  --max-model-len "${MAX_MODEL_LEN:-32768}" \
  --gpu-memory-utilization "${GPU_UTIL:-0.90}" \
  --generation-config vllm \
  --seed 0
