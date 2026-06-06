"""
Rerank 多样性优化离线评估脚本

用法：
    1. 首次跑：生成 candidates pool（每个 query 的 Top-100 召回结果）
       python eval_rerank.py --build-pool

    2. 跑评估：对多个 ranker 变体计算指标
       python eval_rerank.py --eval

设计思路：
- 召回阶段（耗时）只跑一次，缓存所有 90 个 query 的 Top-100 候选
- 排序阶段（快）对每个 ranker 变体重跑一遍，计算指标
- 这样可以快速迭代 ranker 参数，避免重复调用召回 API
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

# 让脚本可以独立运行
SCRIPT_DIR = Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "rag-service"))

from metrics import evaluate_single, aggregate_metrics

# ──────────────────────────────────────────────
# 配置
# ──────────────────────────────────────────────

EVAL_SET_PATH = SCRIPT_DIR / "eval_set.json"
CANDIDATES_POOL_PATH = SCRIPT_DIR / "candidates_pool.json"
RELEVANCE_LABELS_PATH = SCRIPT_DIR / "relevance_labels.json"
REPORT_PATH = SCRIPT_DIR / "eval_rerank_report.md"

# 评估前 90 个问题（与 RAG/模型评估.md 一致，跳过房型约束类）
EVAL_QUESTION_LIMIT = 90

# Ranker 变体配置
RANKER_VARIANTS = {
    "baseline":      {"enable_mmr": False, "enable_category_diversity": False, "relevance_floor": 0.0},
    "floor_0.15":    {"enable_mmr": False, "enable_category_diversity": False, "relevance_floor": 0.15},
    "mmr_lambda0.9": {"enable_mmr": True,  "mmr_lambda": 0.9, "enable_category_diversity": False},
    "mmr_lambda0.7": {"enable_mmr": True,  "mmr_lambda": 0.7, "enable_category_diversity": False},
    "mmr_lambda0.5": {"enable_mmr": True,  "mmr_lambda": 0.5, "enable_category_diversity": False},
    "cat_max2":      {"enable_mmr": False, "enable_category_diversity": True, "max_per_category": 2},
    "mmr_0.7+cat_2": {"enable_mmr": True,  "mmr_lambda": 0.7, "enable_category_diversity": True, "max_per_category": 2},
}


# ──────────────────────────────────────────────
# 1. 构建候选池（一次性，调用 RAG 系统的检索阶段）
# ──────────────────────────────────────────────

def build_candidates_pool(rag_system, eval_questions, limit=EVAL_QUESTION_LIMIT):
    """
    对每个评估问题跑一遍检索，保存 Top-100 候选用于后续 ranker 评估
    """
    from concurrent.futures import ThreadPoolExecutor

    print(f"[BuildPool] 开始为 {min(limit, len(eval_questions))} 个问题构建候选池...")
    pool = {}

    def _process(item):
        qid = item['question_id']
        query = item['question']
        try:
            # 跑完整链路但不调用 ranking（直接拿到 retrieval Top-100）
            result = rag_system.query(
                query,
                enable_ranking=False,    # 不排序，直接拿 Top-100
                enable_generation=False, # 不生成
                ranking_topk=100,
                print_response=False,
            )
            return qid, result['references']['comments'], result['query_processing']
        except Exception as e:
            print(f"[BuildPool] qid={qid} failed: {e}")
            return qid, None, None

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(_process, item) for item in eval_questions[:limit]]
        for i, future in enumerate(futures, 1):
            qid, candidates, query_proc = future.result()
            if candidates is not None:
                pool[str(qid)] = {
                    'candidates': candidates,
                    'time_sensitivity': (query_proc or {}).get('intent_detection', {}).get('time_sensitivity'),
                }
            if i % 10 == 0:
                print(f"[BuildPool] {i}/{min(limit, len(eval_questions))} done")

    with open(CANDIDATES_POOL_PATH, 'w', encoding='utf-8') as f:
        json.dump(pool, f, ensure_ascii=False, indent=2)
    print(f"[BuildPool] 已保存 {len(pool)} 条候选池到 {CANDIDATES_POOL_PATH}")


# ──────────────────────────────────────────────
# 2. 跑评估
# ──────────────────────────────────────────────

def load_eval_data():
    """加载评估问题集和候选池"""
    with open(EVAL_SET_PATH, 'r', encoding='utf-8') as f:
        eval_questions = json.load(f)[:EVAL_QUESTION_LIMIT]

    if not CANDIDATES_POOL_PATH.exists():
        raise FileNotFoundError(
            f"候选池不存在：{CANDIDATES_POOL_PATH}\n"
            f"请先运行 python eval_rerank.py --build-pool"
        )
    with open(CANDIDATES_POOL_PATH, 'r', encoding='utf-8') as f:
        candidates_pool = json.load(f)

    # 相关性标注（可选）
    relevance_labels = {}
    if RELEVANCE_LABELS_PATH.exists():
        with open(RELEVANCE_LABELS_PATH, 'r', encoding='utf-8') as f:
            relevance_labels = json.load(f)
        print(f"[Eval] 已加载相关性标注：{len(relevance_labels)} 个 query")
    else:
        print("[Eval] 未找到相关性标注文件，将跳过 NDCG/MRR/MAP 指标")

    return eval_questions, candidates_pool, relevance_labels


def run_variant(variant_name, variant_config, eval_questions, candidates_pool,
                relevance_labels, rag_system):
    """
    对单个 ranker 变体跑评估
    
    返回：
        per_query_metrics: list[dict]
        per_query_metrics_by_type: dict[str -> list[dict]]
    """
    from modules.ranker import MultiFactorRanker

    print(f"\n[Eval] >>> 评估变体: {variant_name}")
    print(f"       配置: {variant_config}")

    # 构建 ranker
    base_kwargs = {
        'w_relevance': 0.40, 'w_quality': 0.25,
        'w_length': 0.05, 'w_review': 0.05, 'w_useful': 0.05,
        'w_recency': 0.20,
        'base_decay': 0.5, 'implied_boost': 0.5, 'clear_boost': 0.5,
        'half_life_days': 180,
        'dataset_latest_date': datetime(2025, 4, 17),
        'embedding_client': rag_system.embedding_client if variant_config.get('enable_mmr') else None,
    }
    ranker = MultiFactorRanker(rag_system.reranker, **{**base_kwargs, **variant_config})

    per_query_metrics = []
    per_query_metrics_by_type = defaultdict(list)

    for q in eval_questions:
        qid = str(q['question_id'])
        if qid not in candidates_pool:
            continue
        
        candidates = candidates_pool[qid]['candidates']
        time_sensitivity = candidates_pool[qid].get('time_sensitivity')

        try:
            ranked, _ = ranker.rank(
                q['question'], candidates,
                time_sensitivity=time_sensitivity, topk=10
            )
        except Exception as e:
            print(f"  [WARN] qid={qid} rank failed: {e}")
            continue

        # 收集 embeddings 用于计算文本 ILD（若已启用 MMR，可以复用；否则按需 embed）
        embeddings = None
        if variant_config.get('enable_mmr'):
            # MMR 路径下 embedding 已被计算过，但没缓存到 result；这里重新做一次以保证可比
            embeddings = ranker._get_embeddings(ranked)

        labels = relevance_labels.get(qid)
        intent_dir = q.get('metadata', {}).get('intent_direction')

        metrics = evaluate_single(
            ranked,
            relevance_labels=labels,
            embeddings=embeddings,
            intent_directions=[intent_dir] if intent_dir else None,
        )
        per_query_metrics.append(metrics)
        per_query_metrics_by_type[q.get('question_type', 'unknown')].append(metrics)

    return per_query_metrics, per_query_metrics_by_type


def evaluate_all_variants(rag_system):
    """主评估流程：跑所有变体并生成 markdown 报告"""
    eval_questions, candidates_pool, relevance_labels = load_eval_data()

    all_results = {}
    all_results_by_type = defaultdict(dict)

    for variant_name, config in RANKER_VARIANTS.items():
        per_q, per_q_by_type = run_variant(
            variant_name, config, eval_questions, candidates_pool,
            relevance_labels, rag_system
        )
        all_results[variant_name] = aggregate_metrics(per_q)
        for qtype, metrics_list in per_q_by_type.items():
            all_results_by_type[qtype][variant_name] = aggregate_metrics(metrics_list)

    generate_report(all_results, all_results_by_type)


def generate_report(all_results, all_results_by_type):
    """生成 markdown 评估报告"""
    lines = ["# Rerank 多样性优化评估报告\n"]
    lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"评估问题数: {EVAL_QUESTION_LIMIT}\n")
    lines.append("\n---\n")

    # 1. 全局对比表
    lines.append("## 1. 全局指标对比\n")
    all_metric_keys = set()
    for v in all_results.values():
        all_metric_keys.update(v.keys())
    metric_keys = sorted(all_metric_keys)

    header = "| 变体 | " + " | ".join(metric_keys) + " |"
    sep = "|------|" + "|".join(["------"] * len(metric_keys)) + "|"
    lines.append(header)
    lines.append(sep)
    for variant_name, metrics in all_results.items():
        row = f"| {variant_name} | "
        row += " | ".join(f"{metrics.get(k, 0):.4f}" for k in metric_keys)
        row += " |"
        lines.append(row)
    lines.append("")

    # 2. 分场景对比表
    for qtype, results_map in all_results_by_type.items():
        lines.append(f"\n## 2.{qtype} 类型问题对比\n")
        lines.append(header)
        lines.append(sep)
        for variant_name, metrics in results_map.items():
            row = f"| {variant_name} | "
            row += " | ".join(f"{metrics.get(k, 0):.4f}" for k in metric_keys)
            row += " |"
            lines.append(row)
        lines.append("")

    # 3. 关键发现（简单文字总结）
    lines.append("\n## 3. 关键发现\n")
    baseline = all_results.get('baseline', {})
    for variant_name in ['mmr_lambda0.7', 'mmr_0.7+cat_2']:
        v = all_results.get(variant_name, {})
        if not v or not baseline:
            continue
        lines.append(f"### {variant_name} vs baseline\n")
        for k in ['ndcg@10', 'cat_coverage', 'cat_top3_concentration', 'ild_text']:
            if k in v and k in baseline:
                delta = v[k] - baseline[k]
                pct = (delta / baseline[k] * 100) if baseline[k] != 0 else 0
                sign = "+" if delta >= 0 else ""
                lines.append(f"- **{k}**: {baseline[k]:.4f} → {v[k]:.4f} ({sign}{delta:.4f}, {sign}{pct:.1f}%)")
        lines.append("")

    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print(f"\n[Eval] 报告已生成: {REPORT_PATH}")
    print("\n".join(lines[:50]))  # 打印前 50 行预览


# ──────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────

def _init_rag_system():
    """懒加载 RAG 系统（避免无需 RAG 时的环境检查开销）"""
    from modules.rag_system import HotelReviewRAG

    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("DASHSCOPE_INTL_API_KEY")
    intl_key = os.getenv("DASHSCOPE_INTL_API_KEY")
    dv_key = os.getenv("DASHVECTOR_API_KEY")
    dv_endpoint = os.getenv("DASHVECTOR_HOTEL_ENDPOINT")
    if not all([api_key, dv_key, dv_endpoint]):
        raise RuntimeError("请配置环境变量 DASHSCOPE_API_KEY / DASHVECTOR_API_KEY / DASHVECTOR_HOTEL_ENDPOINT")

    data_dir = REPO_ROOT / "rag-service" / "data"
    return HotelReviewRAG(
        api_key=api_key,
        intl_api_key=intl_key,
        dashvector_api_key=dv_key,
        dashvector_endpoint=dv_endpoint,
        data_dir=data_dir,
    )


def main():
    parser = argparse.ArgumentParser(description="Rerank 多样性优化评估")
    parser.add_argument("--build-pool", action="store_true", help="构建候选池（首次需要）")
    parser.add_argument("--eval", action="store_true", help="跑评估")
    args = parser.parse_args()

    if not args.build_pool and not args.eval:
        parser.print_help()
        return

    rag = _init_rag_system()

    if args.build_pool:
        with open(EVAL_SET_PATH, 'r', encoding='utf-8') as f:
            eval_questions = json.load(f)
        build_candidates_pool(rag, eval_questions)

    if args.eval:
        evaluate_all_variants(rag)


if __name__ == "__main__":
    main()
