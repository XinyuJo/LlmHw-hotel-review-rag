#!/usr/bin/env bash
set -euo pipefail
cd /root/sxy/hotel-review-rag
mkdir -p RAG/data/evaluation/logs
source /miniconda3/etc/profile.d/conda.sh
set -a
source .env
set +a
export LD_LIBRARY_PATH=/miniconda3/envs/unsloth_env/lib:${LD_LIBRARY_PATH:-}
exec conda run -n unsloth_env python RAG/sft_hyde/build_hyde_sft_dataset.py \
  --input RAG/data/processed/reverse_queries.csv \
  --output RAG/data/evaluation/hyde_sft_full.jsonl \
  --use-api \
  --max-chars 160 \
  --min-chars 20 \
  --sleep 0.02 \
  --resume \
  --shuffle \
  --seed 42 \
  --valid-ratio 0.1 \
  --workers 8
