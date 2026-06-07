#!/usr/bin/env bash
set -euo pipefail
cd /root/sxy/hotel-review-rag
mkdir -p RAG/data/evaluation/logs
while pgrep -f "python RAG/sft_hyde/build_hyde_sft_dataset.py" >/dev/null; do
  date
  wc -l RAG/data/evaluation/hyde_sft_full.jsonl 2>/dev/null || true
  sleep 60
done
if [[ ! -f RAG/data/evaluation/hyde_sft_full.train.jsonl || ! -f RAG/data/evaluation/hyde_sft_full.valid.jsonl ]]; then
  echo "dataset split files missing" >&2
  tail -n 80 RAG/data/evaluation/logs/build_hyde_sft_full.log >&2 || true
  exit 1
fi
bash RAG/sft_hyde/run_rank_sweep.sh
