#!/usr/bin/env bash
set -euo pipefail
cd /root/sxy/hotel-review-rag
source /miniconda3/etc/profile.d/conda.sh
set -a
source .env
set +a
export LD_LIBRARY_PATH=/miniconda3/envs/unsloth_env/lib:${LD_LIBRARY_PATH:-}
mkdir -p RAG/data/evaluation/logs
MODEL=/root/Exp4/Qwen3-4B-Instruct-2507
PORT=8000
BASE_URL=http://127.0.0.1:${PORT}/v1

wait_server() {
  local name="$1"
  for i in $(seq 1 120); do
    if curl -sf "${BASE_URL}/models" >/dev/null 2>&1; then
      echo "${name} vLLM ready"
      return 0
    fi
    sleep 2
  done
  echo "${name} vLLM not ready" >&2
  return 1
}

stop_server() {
  pkill -f "vllm.entrypoints.openai.api_server" || true
  sleep 5
}

run_base() {
  stop_server
  nohup conda run -n unsloth_env python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --served-model-name Qwen3-4B-Instruct \
    --host 0.0.0.0 \
    --port "$PORT" \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85 \
    --trust-remote-code > RAG/data/evaluation/logs/vllm_qwen3_base.log 2>&1 &
  wait_server base
  conda run -n unsloth_env python RAG/sft_hyde/run_hyde_responses.py \
    --name qwen3_base_hyde \
    --hyde-backend vllm \
    --vllm-base-url "$BASE_URL" \
    --vllm-model Qwen3-4B-Instruct \
    --workers 2 2>&1 | tee RAG/data/evaluation/logs/responses_qwen3_base_hyde.log
  stop_server
}

run_lora() {
  local r="$1"
  local lora="RAG/data/evaluation/hyde_sft_lora_r${r}"
  local name="hyde_sft_r${r}"
  stop_server
  nohup conda run -n unsloth_env python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --served-model-name Qwen3-4B-Instruct \
    --enable-lora \
    --lora-modules "${name}=${lora}" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85 \
    --trust-remote-code > "RAG/data/evaluation/logs/vllm_${name}.log" 2>&1 &
  wait_server "$name"
  conda run -n unsloth_env python RAG/sft_hyde/run_hyde_responses.py \
    --name "qwen3_sft_hyde_r${r}" \
    --hyde-backend vllm \
    --vllm-base-url "$BASE_URL" \
    --vllm-model "$name" \
    --workers 2 2>&1 | tee "RAG/data/evaluation/logs/responses_qwen3_sft_hyde_r${r}.log"
  stop_server
}

run_base
for R in 8 16 32; do
  run_lora "$R"
done

conda run -n unsloth_env python RAG/sft_hyde/evaluate_hyde_rankings.py \
  --configs no_hyde full qwen3_base_hyde qwen3_sft_hyde_r8 qwen3_sft_hyde_r16 qwen3_sft_hyde_r32 \
  --output RAG/data/evaluation/hyde_ranking_results.json \
  --summary-output RAG/data/evaluation/hyde_ranking_summary.json \
  --model qwen-max 2>&1 | tee RAG/data/evaluation/logs/evaluate_hyde_rankings.log
