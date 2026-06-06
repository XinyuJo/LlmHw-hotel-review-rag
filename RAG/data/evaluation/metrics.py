"""
Rerank 评估指标库

包含两类指标：
1. 相关性指标（需要 ground truth 标注）：NDCG@K / MRR@K / MAP@K
2. 多样性指标（无需标注）：ILD / Category Coverage / Category Gini / Top-N Concentration

所有指标输入约定：
- ranked_results: list[dict]，每项至少含 'comment_id'（用于查 label）
- relevance_labels: dict[comment_id -> int]，0=irrelevant, 1=marginal, 2=relevant, 3=highly relevant
- embeddings: np.ndarray, shape=[N, D]
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable

import numpy as np


# ──────────────────────────────────────────────
# 相关性指标
# ──────────────────────────────────────────────

def dcg_at_k(gains: list[float], k: int) -> float:
    """Discounted Cumulative Gain @ K"""
    gains = gains[:k]
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked_results: list[dict], relevance_labels: dict, k: int = 10) -> float:
    """
    NDCG@K
    
    relevance_labels: { comment_id: int in [0, 3] }
    缺失 label 的视为 0
    """
    if not ranked_results:
        return 0.0
    gains = [relevance_labels.get(r['comment_id'], 0) for r in ranked_results]
    dcg = dcg_at_k(gains, k)
    ideal_gains = sorted(relevance_labels.values(), reverse=True)
    idcg = dcg_at_k(ideal_gains, k)
    return dcg / idcg if idcg > 0 else 0.0


def mrr_at_k(ranked_results: list[dict], relevance_labels: dict,
             k: int = 10, threshold: int = 2) -> float:
    """
    MRR@K：首条相关性 >= threshold 的位置倒数
    """
    for rank, r in enumerate(ranked_results[:k], 1):
        if relevance_labels.get(r['comment_id'], 0) >= threshold:
            return 1.0 / rank
    return 0.0


def precision_at_k(ranked_results: list[dict], relevance_labels: dict,
                   k: int = 10, threshold: int = 2) -> float:
    """Precision@K"""
    hits = sum(1 for r in ranked_results[:k]
               if relevance_labels.get(r['comment_id'], 0) >= threshold)
    return hits / k


def map_at_k(ranked_results: list[dict], relevance_labels: dict,
             k: int = 10, threshold: int = 2) -> float:
    """Mean Average Precision @ K（单 query 版本）"""
    score = 0.0
    hits = 0
    for rank, r in enumerate(ranked_results[:k], 1):
        if relevance_labels.get(r['comment_id'], 0) >= threshold:
            hits += 1
            score += hits / rank
    total_relevant = sum(1 for v in relevance_labels.values() if v >= threshold)
    return score / min(total_relevant, k) if total_relevant > 0 else 0.0


# ──────────────────────────────────────────────
# 多样性指标
# ──────────────────────────────────────────────

def intra_list_diversity(embeddings: np.ndarray) -> float:
    """
    ILD: 列表内 embedding 平均两两余弦距离 (1 - cosine_similarity)
    
    embeddings: shape=[K, D]，应对应 Top-K 选中的评论
    返回值 in [0, 1]，越高越多样
    """
    if len(embeddings) < 2:
        return 0.0
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    emb_norm = embeddings / (norms + 1e-8)
    sim_matrix = emb_norm @ emb_norm.T
    # 取上三角（不含对角线）
    K = len(embeddings)
    iu = np.triu_indices(K, k=1)
    avg_sim = sim_matrix[iu].mean()
    return float(1 - avg_sim)


def category_coverage(ranked_results: list[dict], total_categories: int = 14) -> float:
    """
    类别覆盖度: 选中评论涉及的小类数 / 总小类数
    """
    cats = set()
    for r in ranked_results:
        meta_cats = r.get('metadata', {}).get('categories', [])
        if isinstance(meta_cats, list):
            cats.update(meta_cats)
    return len(cats) / total_categories


def category_gini(ranked_results: list[dict]) -> float:
    """
    类别分布的 Gini 系数（基于主类别）
    返回 [0, 1]，0 表示完全均衡，1 表示完全集中
    """
    primary_cats = []
    for r in ranked_results:
        meta_cats = r.get('metadata', {}).get('categories', [])
        if isinstance(meta_cats, list) and meta_cats:
            primary_cats.append(meta_cats[0])
        else:
            primary_cats.append('unknown')
    
    if not primary_cats:
        return 0.0
    
    counts = np.array(sorted(Counter(primary_cats).values()))
    n = len(counts)
    if n <= 1 or counts.sum() == 0:
        return 0.0
    cum = np.cumsum(counts)
    gini = (n + 1 - 2 * np.sum(cum) / cum[-1]) / n
    return float(gini)


def top_n_category_concentration(ranked_results: list[dict], n: int = 3) -> float:
    """
    Top-N 类别集中度: 前 n 多类别占总评论的比例
    返回 [0, 1]，越低越分散
    """
    primary_cats = []
    for r in ranked_results:
        meta_cats = r.get('metadata', {}).get('categories', [])
        if isinstance(meta_cats, list) and meta_cats:
            primary_cats.append(meta_cats[0])
    
    if not primary_cats:
        return 0.0
    
    counts = Counter(primary_cats)
    top_n_count = sum(c for _, c in counts.most_common(n))
    return top_n_count / len(primary_cats)


def subtopic_recall(ranked_results: list[dict], intent_directions: list[str]) -> float:
    """
    子主题召回率: 选中评论的类别覆盖了多少个意图方向
    
    intent_directions: 该问题对应的意图方向列表（小类名称）
    """
    if not intent_directions:
        return 0.0
    covered = set()
    for r in ranked_results:
        meta_cats = r.get('metadata', {}).get('categories', [])
        if isinstance(meta_cats, list):
            for cat in meta_cats:
                if cat in intent_directions:
                    covered.add(cat)
    return len(covered) / len(intent_directions)


def alpha_ndcg_at_k(ranked_results: list[dict], subtopic_labels: dict,
                    k: int = 10, alpha: float = 0.5) -> float:
    """
    α-NDCG@K: 带新颖性折扣的 NDCG
    
    subtopic_labels: { comment_id: list[subtopic_id] }
                     每条评论命中的子主题列表
    alpha: 重复子主题的折扣率（0=不折扣，1=完全折扣）
    """
    if not ranked_results:
        return 0.0
    
    seen_subtopics = Counter()
    dcg = 0.0
    for rank, r in enumerate(ranked_results[:k], 1):
        subs = subtopic_labels.get(r['comment_id'], [])
        gain = sum((1 - alpha) ** seen_subtopics[s] for s in subs)
        dcg += gain / math.log2(rank + 1)
        for s in subs:
            seen_subtopics[s] += 1
    
    # 理想情况（简化版）：所有不同子主题各占一条
    all_subs = set()
    for subs in subtopic_labels.values():
        all_subs.update(subs)
    if not all_subs:
        return 0.0
    ideal_gains = [1.0] * min(len(all_subs), k)
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal_gains))
    return dcg / idcg if idcg > 0 else 0.0


# ──────────────────────────────────────────────
# 综合评估
# ──────────────────────────────────────────────

def evaluate_single(ranked_results: list[dict],
                    relevance_labels: dict | None = None,
                    embeddings: np.ndarray | None = None,
                    intent_directions: list[str] | None = None,
                    k: int = 10) -> dict:
    """
    对单个 query 的排序结果计算所有指标
    
    relevance_labels: 若 None 则跳过相关性指标
    embeddings: 若 None 则跳过文本 ILD
    intent_directions: 若 None 则跳过 subtopic recall
    """
    metrics = {}
    
    # 相关性
    if relevance_labels is not None:
        metrics['ndcg@10'] = ndcg_at_k(ranked_results, relevance_labels, k)
        metrics['mrr@10'] = mrr_at_k(ranked_results, relevance_labels, k)
        metrics['map@10'] = map_at_k(ranked_results, relevance_labels, k)
        metrics['precision@10'] = precision_at_k(ranked_results, relevance_labels, k)
    
    # 多样性 - 类别
    metrics['cat_coverage'] = category_coverage(ranked_results)
    metrics['cat_gini'] = category_gini(ranked_results)
    metrics['cat_top3_concentration'] = top_n_category_concentration(ranked_results, n=3)
    
    # 多样性 - 文本
    if embeddings is not None and len(embeddings) >= 2:
        metrics['ild_text'] = intra_list_diversity(embeddings)
    
    # 子主题
    if intent_directions is not None:
        metrics['subtopic_recall'] = subtopic_recall(ranked_results, intent_directions)
    
    # 平均相关性分数（无需 ground truth）
    rerank_scores = [r.get('rerank_score', 0) for r in ranked_results[:k]]
    metrics['avg_rerank_score'] = float(np.mean(rerank_scores)) if rerank_scores else 0.0
    metrics['min_rerank_score'] = float(np.min(rerank_scores)) if rerank_scores else 0.0
    
    return metrics


def aggregate_metrics(per_query_metrics: list[dict]) -> dict:
    """对多 query 的指标做平均聚合"""
    if not per_query_metrics:
        return {}
    keys = per_query_metrics[0].keys()
    return {k: float(np.mean([m.get(k, 0) for m in per_query_metrics])) for k in keys}
