"""生成 HyDE 对比实验的 RAG 回复缓存。

示例：
  # 无 HyDE
  python RAG/sft_hyde/run_hyde_responses.py --name no_hyde --disable-hyde

  # DashScope 通用 LLM + HyDE
  python RAG/sft_hyde/run_hyde_responses.py --name dashscope_hyde --hyde-backend dashscope

  # 本地 Qwen3 / SFT Qwen3 + HyDE
  python RAG/sft_hyde/run_hyde_responses.py --name qwen3_base_hyde --hyde-backend vllm --vllm-model Qwen3-4B-Instruct
  python RAG/sft_hyde/run_hyde_responses.py --name qwen3_sft_hyde --hyde-backend vllm --vllm-model hyde_sft_lora
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
SERVICE_DIR = ROOT / "rag-service"
sys.path.insert(0, str(SERVICE_DIR))

from modules.rag_system import HotelReviewRAG  # noqa: E402


def sanitize(obj):
    """递归转换 numpy/pandas 类型，便于 JSON 序列化。"""
    if isinstance(obj, dict):
        return {k: sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    if isinstance(obj, tuple):
        return [sanitize(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return None if np.isnan(value) else value
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if pd.isna(obj) if not isinstance(obj, (str, bytes, list, dict, tuple)) else False:
        return None
    return obj


def load_questions(path: Path, limit: int) -> list[dict]:
    questions = json.loads(path.read_text(encoding="utf-8"))[:90]
    return questions[:limit] if limit else questions


def build_rag(args: argparse.Namespace) -> HotelReviewRAG:
    required = ["DASHSCOPE_API_KEY", "DASHVECTOR_API_KEY", "DASHVECTOR_HOTEL_ENDPOINT"]
    missing = [key for key in required if not os.getenv(key)]
    if missing:
        raise RuntimeError(f"缺少环境变量: {', '.join(missing)}")

    comments_path = ROOT / args.comments_csv
    df_comments = None
    if comments_path.exists():
        df_comments = pd.read_csv(comments_path)
        if "_id" in df_comments.columns:
            df_comments = df_comments.set_index("_id")
        print(f"从本地 CSV 加载评论数据: {comments_path} ({len(df_comments)} 条)", flush=True)

    return HotelReviewRAG(
        api_key=os.environ["DASHSCOPE_API_KEY"],
        dashvector_api_key=os.environ["DASHVECTOR_API_KEY"],
        dashvector_endpoint=os.environ["DASHVECTOR_HOTEL_ENDPOINT"],
        data_dir=SERVICE_DIR / "data",
        df_comments=df_comments,
        intl_api_key=os.getenv("DASHSCOPE_INTL_API_KEY") or None,
        hyde_backend=args.hyde_backend,
        hyde_vllm_base_url=args.vllm_base_url,
        hyde_vllm_model=args.vllm_model,
        hyde_vllm_api_key=args.vllm_api_key,
    )


def run_one(rag: HotelReviewRAG, question: dict, args: argparse.Namespace) -> dict:
    start = time.time()
    last_error = None
    for attempt in range(args.retries + 1):
        try:
            result = rag.query(
                question["question"],
                enable_hyde=not args.disable_hyde,
                enable_generation=True,
                print_response=False,
            )
            return {
                "question_id": question["question_id"],
                "question_type": question.get("question_type"),
                "question": question["question"],
                "response": result["response"],
                "references": sanitize(result["references"]),
                "query_processing": sanitize(result["query_processing"]),
                "timing": sanitize(result["timing"]),
                "error": None,
                "wall_time": time.time() - start,
                "attempts": attempt + 1,
            }
        except Exception as exc:
            last_error = str(exc)
            if attempt < args.retries:
                time.sleep(args.retry_sleep)

    return {
        "question_id": question["question_id"],
        "question_type": question.get("question_type"),
        "question": question["question"],
        "response": "",
        "references": {},
        "query_processing": {},
        "timing": {},
        "error": last_error,
        "wall_time": time.time() - start,
        "attempts": args.retries + 1,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--eval-set", default="RAG/data/evaluation/eval_set.json")
    parser.add_argument("--output-dir", default="RAG/data/evaluation")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--comments-csv", default="RAG/data/processed/enriched_comments.csv")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed-only", action="store_true")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=5.0)
    parser.add_argument("--disable-hyde", action="store_true")
    parser.add_argument("--hyde-backend", choices=["dashscope", "vllm"], default="dashscope")
    parser.add_argument("--vllm-base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--vllm-model", default="Qwen3-4B-Instruct")
    parser.add_argument("--vllm-api-key", default="EMPTY")
    args = parser.parse_args()

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"responses_{args.name}.json"
    existing_by_id = {}
    if args.resume and output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
        existing_by_id = {row["question_id"]: row for row in existing}
        print(f"断点续跑: 已加载 {len(existing_by_id)} 条已有结果", flush=True)

    questions = load_questions(ROOT / args.eval_set, args.limit)
    if existing_by_id:
        if args.retry_failed_only:
            questions = [
                q for q in questions
                if existing_by_id.get(q["question_id"], {}).get("error")
            ]
        else:
            questions = [
                q for q in questions
                if q["question_id"] not in existing_by_id
            ]
    print(f"本次待运行问题数: {len(questions)}", flush=True)
    rag = build_rag(args)

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_one, rag, q, args) for q in questions]
        for future in tqdm(as_completed(futures), total=len(futures), desc=args.name):
            results.append(future.result())

    merged = dict(existing_by_id)
    for row in results:
        merged[row["question_id"]] = row
    results = list(merged.values())
    results.sort(key=lambda x: x["question_id"])
    output_path.write_text(
        json.dumps(sanitize(results), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    errors = sum(1 for item in results if item["error"])
    print(f"已保存: {output_path}")
    print(f"总数: {len(results)}, 错误: {errors}")


if __name__ == "__main__":
    main()
