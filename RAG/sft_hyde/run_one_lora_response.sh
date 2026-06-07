#!/usr/bin/env bash
set -euo pipefail
R=${1:?rank required}
cd /root/sxy/hotel-review-rag
source /miniconda3/etc/profile.d/conda.sh
set -a; source .env; set +a
export LD_LIBRARY_PATH=/miniconda3/envs/unsloth_env/lib:${LD_LIBRARY_PATH:-}
MODEL=/root/Exp4/Qwen3-4B-Instruct-2507
PORT=8000
NAME=hyde_sft_r${R}
LORA=RAG/data/evaluation/hyde_sft_lora_r${R}
LOGDIR=RAG/data/evaluation/logs
mkdir -p "$LOGDIR"
pkill -f "vllm.entrypoints.openai.api_server" || true
sleep 8
nohup conda run -n unsloth_env python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name Qwen3-4B-Instruct \
  --enable-lora \
  --lora-modules "${NAME}=${LORA}" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --trust-remote-code > "${LOGDIR}/vllm_${NAME}.log" 2>&1 &
SERVER_PID=$!
for i in $(seq 1 180); do
  if /usr/bin/curl -sf http://127.0.0.1:${PORT}/v1/models >/dev/null 2>&1; then
    echo "${NAME} ready"
    break
  fi
  sleep 2
  if [[ $i -eq 180 ]]; then
    echo "${NAME} failed to start" >&2
    tail -n 120 "${LOGDIR}/vllm_${NAME}.log" >&2 || true
    exit 1
  fi
done
conda run -n unsloth_env python -u RAG/sft_hyde/run_hyde_responses.py \
  --name qwen3_sft_hyde_r${R} \
  --hyde-backend vllm \
  --vllm-base-url http://127.0.0.1:${PORT}/v1 \
  --vllm-model "$NAME" \
  --workers 2 \
  --retries 3 \
  --retry-sleep 8 2>&1 | tee "${LOGDIR}/responses_qwen3_sft_hyde_r${R}.log"
pkill -f "vllm.entrypoints.openai.api_server" || true
sleep 5
