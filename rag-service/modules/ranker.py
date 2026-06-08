"""排序模块：Reranker + 线性加权 + MMR / 类别多样性"""

from __future__ import annotations

import time
from collections import defaultdict
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from dashscope import TextReRank


class Reranker:
    """Reranker：使用 Qwen3-Rerank 模型计算相关性得分"""

    def __init__(self, api_key: str, model: str = "qwen3-rerank"):
        self.api_key = api_key
        self.model = model

    def rerank(self, query: str, documents: list[str], topk: int = None) -> dict:
        """
        对文档进行重排序

        返回:
            {index: relevance_score}
        """
        if topk is None:
            topk = len(documents)

        response = TextReRank.call(
            api_key=self.api_key,
            model=self.model,
            query=query,
            documents=documents,
            top_n=topk,
            return_documents=False
        )

        if response.status_code == 200:
            return {item.index: item.relevance_score for item in response.output.results}
        else:
            raise RuntimeError(f"Rerank 调用失败: {response.message}")


def _safe_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """基于 P95 的鲁棒归一化，避免极端值主导分布"""
    if len(x) == 0:
        return x
    p95 = np.percentile(x, 95)
    return np.clip(x / (p95 + eps), 0, 1)


class MultiFactorRanker:
    """多因子排序器：融合相关性、内容质量、时效性，可选 MMR / 类别多样性"""

    def __init__(self, reranker,
                 embedding_client=None,
                 # 既有特征权重
                 w_relevance: float = 0.40,
                 w_quality: float = 0.25,
                 w_length: float = 0.05,
                 w_review: float = 0.05,
                 w_useful: float = 0.05,
                 w_recency: float = 0.20,
                 # 既有时效性参数
                 base_decay: float = 0.5,
                 implied_boost: float = 0.5,
                 clear_boost: float = 0.5,
                 half_life_days: int = 180,
                 # —— 新增：P0 修复 ——
                 dataset_latest_date: datetime | None = None,
                 relevance_floor: float = 0.0,
                 use_p95_normalization: bool = False,
                 # —— 新增：MMR 多样性 ——
                 enable_mmr: bool = False,
                 mmr_lambda: float = 0.7,
                 # —— 新增：类别配额兜底 ——
                 enable_category_diversity: bool = False,
                 max_per_category: int = 2,
                 # —— 新增：MMR + 类别配额组合时的过选比例 ——
                 mmr_overselect_ratio: float = 1.5):
        self.reranker = reranker
        self.embedding_client = embedding_client

        self.w_relevance = w_relevance
        self.w_quality = w_quality
        self.w_length = w_length
        self.w_review = w_review
        self.w_useful = w_useful
        self.w_recency = w_recency

        self.base_decay = base_decay
        self.implied_boost = implied_boost
        self.clear_boost = clear_boost
        self.half_life_days = half_life_days

        # 新增参数
        self.dataset_latest_date = dataset_latest_date
        self.relevance_floor = relevance_floor
        self.use_p95_normalization = use_p95_normalization

        self.enable_mmr = enable_mmr
        self.mmr_lambda = mmr_lambda

        self.enable_category_diversity = enable_category_diversity
        self.max_per_category = max_per_category

        self.mmr_overselect_ratio = mmr_overselect_ratio

        # 启用 MMR 时必须提供 embedding_client
        if self.enable_mmr and self.embedding_client is None:
            raise ValueError("启用 MMR 时必须提供 embedding_client")

    # ──────────────────────────────────────────────
    # 主入口
    # ──────────────────────────────────────────────

    def rank(self, query: str, candidates: list[dict], time_sensitivity: str = None,
             topk: int = 10, today: datetime | None = None) -> tuple[list[dict], dict]:
        """
        多因子排序

        返回:
            (ranked_results, timing_info)
        """
        ranking_start = time.time()

        if not candidates:
            return [], {'total': 0, 'rerank': 0, 'scoring': 0, 'mmr': 0}

        # 1. Rerank 打分
        rerank_start = time.time()
        documents = [c['comment'] for c in candidates]
        relevance_map = self.reranker.rerank(query, documents)
        rerank_time = time.time() - rerank_start

        # 2. 计算各特征
        scoring_start = time.time()
        feature_arrays = self._compute_features(
            candidates, relevance_map, time_sensitivity, today
        )
        relevance_score = feature_arrays['relevance']

        # 3. 综合得分（线性加权）
        final_score = (
            self.w_relevance * feature_arrays['relevance'] +
            self.w_quality * feature_arrays['quality'] +
            self.w_length * feature_arrays['length'] +
            self.w_review * feature_arrays['review'] +
            self.w_useful * feature_arrays['useful'] +
            self.w_recency * feature_arrays['recency']
        )
        scoring_time = time.time() - scoring_start

        # 4. 相关性阈值截断
        keep_mask = self._apply_relevance_floor(relevance_score, topk)

        # 5. 多样性选择（MMR + 类别配额）
        mmr_start = time.time()
        sorted_index, selection_method = self._select_topk(
            candidates, final_score, keep_mask, topk
        )
        mmr_time = time.time() - mmr_start

        # 6. 构建结果（含完整 rerank rank 信息）
        rerank_sorted_index = np.argsort(relevance_score)[::-1]
        rerank_rank = np.empty_like(rerank_sorted_index)
        rerank_rank[rerank_sorted_index] = np.arange(1, len(relevance_score) + 1)

        ranked_results = []
        for rank, idx in enumerate(sorted_index, 1):
            c = candidates[idx]
            result = {
                **c,
                'rerank_score': float(relevance_score[idx]),
                'rerank_rank': int(rerank_rank[idx]),
                'final_score': float(final_score[idx]),
                'final_rank': rank,
                'selection_method': selection_method,
                'feature_scores': {
                    'relevance': float(feature_arrays['relevance'][idx]),
                    'quality': float(feature_arrays['quality'][idx]),
                    'log_comment_len': float(feature_arrays['length'][idx]),
                    'log_review_count': float(feature_arrays['review'][idx]),
                    'log_useful_count': float(feature_arrays['useful'][idx]),
                    'recency': float(feature_arrays['recency'][idx])
                }
            }
            ranked_results.append(result)

        timing_info = {
            'total': time.time() - ranking_start,
            'rerank': rerank_time,
            'scoring': scoring_time,
            'mmr': mmr_time
        }
        return ranked_results, timing_info

    # ──────────────────────────────────────────────
    # 特征计算
    # ──────────────────────────────────────────────

    def _compute_features(self, candidates, relevance_map, time_sensitivity, today):
        """统一计算 6 个特征维度，返回 dict[name -> ndarray]"""
        n = len(candidates)

        # 相关性
        relevance = np.array([relevance_map.get(i, 0) for i in range(n)])

        # 质量分（已经是 0-10 整数）
        quality_raw = np.array([c['metadata']['quality_score'] for c in candidates])
        quality = quality_raw / 10.0

        # 长度 / 评论数 / 点赞数
        comment_len = np.array([len(c['comment']) for c in candidates])
        log_comment_len = np.log(comment_len + 1)

        review_count = np.array([c['metadata']['review_count'] for c in candidates])
        log_review_count = np.log(review_count + 1)

        useful_count = np.array([c['metadata']['useful_count'] for c in candidates])
        log_useful_count = np.log(useful_count + 1)

        if self.use_p95_normalization:
            length = _safe_normalize(log_comment_len)
            review = _safe_normalize(log_review_count)
            useful = _safe_normalize(log_useful_count)
        else:
            # 保留原始硬编码归一化（向后兼容）
            length = log_comment_len / 7.51
            review = log_review_count / 6.32
            useful = log_useful_count / 3.64

        # 时效性（修复 today 锚定）
        decay = self.base_decay
        if time_sensitivity == "implied":
            decay += self.implied_boost
        elif time_sensitivity == "clear":
            decay += self.implied_boost + self.clear_boost

        if today is None:
            # P0 修复：优先用数据集最新日期锚定，避免随系统时间无意义衰减
            today = self.dataset_latest_date or datetime.today()

        publish_date = pd.to_datetime([c['metadata']['publish_date'] for c in candidates])
        days_ago = (today - publish_date).days.values
        days_ago = np.maximum(days_ago, 0)
        recency = np.exp(-decay * days_ago / self.half_life_days)

        return {
            'relevance': relevance,
            'quality': quality,
            'length': length,
            'review': review,
            'useful': useful,
            'recency': recency
        }

    # ──────────────────────────────────────────────
    # 相关性阈值截断
    # ──────────────────────────────────────────────

    def _apply_relevance_floor(self, relevance_score, topk):
        """返回布尔 mask，True 表示通过阈值"""
        n = len(relevance_score)
        if self.relevance_floor <= 0:
            return np.ones(n, dtype=bool)

        keep_mask = relevance_score >= self.relevance_floor
        # 兜底：至少保留 max(topk//3, 3) 条
        min_keep = min(n, max(topk // 3, 3))
        if keep_mask.sum() < min_keep:
            keep_idx = np.argsort(relevance_score)[::-1][:min_keep]
            keep_mask = np.zeros(n, dtype=bool)
            keep_mask[keep_idx] = True
        return keep_mask

    # ──────────────────────────────────────────────
    # 选择 Top-K（核心：根据配置决定走纯排序 / MMR / 类别配额 / 组合）
    # ──────────────────────────────────────────────

    def _select_topk(self, candidates, final_score, keep_mask, topk):
        """根据配置选择 Top-K，返回 (sorted_indices_list, method_name)"""
        valid_indices = np.where(keep_mask)[0]
        if len(valid_indices) == 0:
            return [], 'empty'

        # 分场景路由
        if not self.enable_mmr and not self.enable_category_diversity:
            # 纯线性加权（保持原有行为）
            sub_score = final_score[valid_indices]
            order = np.argsort(sub_score)[::-1][:topk]
            return [int(valid_indices[i]) for i in order], 'pure'

        if self.enable_mmr and not self.enable_category_diversity:
            # 纯 MMR
            selected_local = self._mmr_select(
                final_score[valid_indices],
                self._get_embeddings([candidates[i] for i in valid_indices]),
                topk, self.mmr_lambda
            )
            return [int(valid_indices[i]) for i in selected_local], 'mmr'

        if not self.enable_mmr and self.enable_category_diversity:
            # 纯类别配额
            sub_score = final_score[valid_indices]
            sub_order = np.argsort(sub_score)[::-1]
            selected_local = self._category_diverse_filter(
                [candidates[valid_indices[i]] for i in sub_order],
                topk, self.max_per_category
            )
            # 注意：selected_local 是 sub_order 上的下标
            return [int(valid_indices[sub_order[i]]) for i in selected_local], 'cat'

        # MMR + 类别配额组合
        overselect_k = max(topk, int(topk * self.mmr_overselect_ratio))
        mmr_selected_local = self._mmr_select(
            final_score[valid_indices],
            self._get_embeddings([candidates[i] for i in valid_indices]),
            overselect_k, self.mmr_lambda
        )
        # 在 MMR 结果中按类别配额裁剪
        cat_selected_local = self._category_diverse_filter(
            [candidates[valid_indices[i]] for i in mmr_selected_local],
            topk, self.max_per_category
        )
        return [int(valid_indices[mmr_selected_local[i]]) for i in cat_selected_local], 'mmr+cat'

    # ──────────────────────────────────────────────
    # MMR 选择
    # ──────────────────────────────────────────────

    def _mmr_select(self, final_score: np.ndarray, embeddings: np.ndarray,
                    topk: int, lambda_: float) -> list[int]:
        """
        MMR 贪心选择

        参数:
            final_score: shape=[N] 的综合分数
            embeddings:  shape=[N, D] 的候选评论 embedding
            topk: 目标数量
            lambda_: 相关性 vs 多样性权衡参数

        返回:
            选中的下标列表（长度 <= topk）
        """
        n = len(final_score)
        if n <= topk:
            return list(np.argsort(final_score)[::-1])

        # min-max 归一化 final_score 到 [0,1]，与相似度量纲对齐
        s_min, s_max = final_score.min(), final_score.max()
        s = (final_score - s_min) / (s_max - s_min + 1e-8)

        # L2 归一化 + 余弦相似度矩阵
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        emb_norm = embeddings / (norms + 1e-8)
        sim_matrix = emb_norm @ emb_norm.T

        # 起始：相关性最高的一条
        first = int(np.argmax(s))
        selected = [first]
        remaining = set(range(n)) - {first}

        while len(selected) < topk and remaining:
            rem_arr = np.array(sorted(remaining))
            # 每个 remaining 候选与已选集合的最大相似度
            max_sim_to_selected = sim_matrix[rem_arr][:, selected].max(axis=1)
            mmr_score = lambda_ * s[rem_arr] - (1 - lambda_) * max_sim_to_selected
            chosen = int(rem_arr[np.argmax(mmr_score)])
            selected.append(chosen)
            remaining.remove(chosen)

        return selected

    # ──────────────────────────────────────────────
    # 类别配额过滤
    # ──────────────────────────────────────────────

    def _category_diverse_filter(self, sorted_candidates: list[dict], topk: int,
                                 max_per_cat: int) -> list[int]:
        """
        按主类别配额裁剪有序候选

        参数:
            sorted_candidates: 已按分数降序的候选列表
            topk: 目标数量
            max_per_cat: 每个类别最大数量

        返回:
            选中的下标（在 sorted_candidates 中的下标）
        """
        cat_count = defaultdict(int)
        selected_indices = []
        deferred = []  # 配额已满但可能用于补足

        for i, c in enumerate(sorted_candidates):
            cats = c.get('metadata', {}).get('categories', [])
            # categories 可能存为字符串（JSON 格式）或 list
            if isinstance(cats, str):
                try:
                    import json
                    cats = json.loads(cats.replace("'", '"'))
                except Exception:
                    cats = []
            primary_cat = cats[0] if cats else 'unknown'

            if cat_count[primary_cat] < max_per_cat:
                selected_indices.append(i)
                cat_count[primary_cat] += 1
                if len(selected_indices) >= topk:
                    break
            else:
                deferred.append(i)

        # 不够 topk 时，用 deferred 补足（按原始顺序）
        if len(selected_indices) < topk:
            need = topk - len(selected_indices)
            selected_indices.extend(deferred[:need])

        return selected_indices

    # ──────────────────────────────────────────────
    # Embedding 获取
    # ──────────────────────────────────────────────

    def _get_embeddings(self, candidates: list[dict]) -> np.ndarray:
        """对候选评论做 batch embedding（启用 MMR 时调用）"""
        if self.embedding_client is None:
            raise RuntimeError("embedding_client 未初始化，无法启用 MMR")
        texts = [c['comment'] for c in candidates]
        # DashScope text-embedding-v4 单次最多 25 条，做 batch
        BATCH = 25
        all_embs = []
        for i in range(0, len(texts), BATCH):
            batch = texts[i:i + BATCH]
            embs = self.embedding_client.embed_batch(batch)
            all_embs.extend(embs)
        return np.array(all_embs, dtype=np.float32)
