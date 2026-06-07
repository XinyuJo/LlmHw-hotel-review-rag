#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/root/sxy/Qwen3-4B-Instruct}"
LORA_PATH="${LORA_PATH:-hyde_sft_lora}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-hyde_sft_lora}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_LORA_RANK="${MAX_LORA_RANK:-32}"

python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --served-model-name "Qwen3-4B-Instruct" \
  --enable-lora \
  --max-lora-rank "${MAX_LORA_RANK}" \
  --lora-modules "${SERVED_MODEL_NAME}=${LORA_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --trust-remote-code
