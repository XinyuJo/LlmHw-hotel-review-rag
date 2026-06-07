#!/usr/bin/env bash
set -euo pipefail
cd /root/sxy/hotel-review-rag
source /miniconda3/etc/profile.d/conda.sh
export LD_LIBRARY_PATH=/miniconda3/envs/unsloth_env/lib:${LD_LIBRARY_PATH:-}
MODEL=/root/Exp4/Qwen3-4B-Instruct-2507
DATA=RAG/data/evaluation/hyde_sft_full.train.jsonl
VALID=RAG/data/evaluation/hyde_sft_full.valid.jsonl
mkdir -p RAG/data/evaluation/logs
for R in 8 16 32; do
  OUT=RAG/data/evaluation/hyde_sft_lora_r${R}
  echo "==== train LoRA rank ${R} ===="
  conda run -n unsloth_env python RAG/sft_hyde/train_hyde_lora.py \
    --model "$MODEL" \
    --data "$DATA" \
    --eval-data "$VALID" \
    --output "$OUT" \
    --run-name "hyde_sft_r${R}" \
    --max-steps 400 \
    --batch-size 2 \
    --grad-accum 4 \
    --learning-rate 2e-4 \
    --lora-r "$R" \
    --lora-alpha "$R" \
    --eval-steps 50 \
    --save-steps 100 \
    --offline 2>&1 | tee "RAG/data/evaluation/logs/train_lora_r${R}.log"
done
