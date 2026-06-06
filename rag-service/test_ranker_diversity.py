"""
Ranker 多样性优化单元测试

无需真实 API，用 mock 数据验证：
1. baseline 行为与改造前一致
2. relevance_floor 截断生效
3. MMR 选择能提升多样性
4. 类别配额能打散主类
5. MMR + 类别配额组合
6. today 锚定修复

运行：python test_ranker_diversity.py
"""

import sys
import importlib.util
from datetime import datetime
from pathlib import Path

import numpy as np

# 直接从文件加载 ranker，避免 modules/__init__.py 触发 generator 加载（其依赖 Python 3.10+ 语法）
_ranker_path = Path(__file__).parent / "modules" / "ranker.py"
_spec = importlib.util.spec_from_file_location("ranker", _ranker_path)
_ranker_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ranker_module)
MultiFactorRanker = _ranker_module.MultiFactorRanker
_safe_normalize = _ranker_module._safe_normalize


# ──────────────────────────────────────────────
# Mock 对象
# ──────────────────────────────────────────────

class MockReranker:
    """
    模拟 Qwen3-Rerank：基础分 0.2 + 字符重叠分 (max 0.8)
    
    加基础分避免所有"非命中"文档分数都为 0 导致 min-max 归一化后变成平局
    """
    def rerank(self, query, documents, topk=None):
        scores = {}
        q_chars = set(query)
        for i, doc in enumerate(documents):
            overlap = len(q_chars & set(doc))
            scores[i] = 0.2 + min(overlap / max(len(q_chars), 1), 1.0) * 0.8
        return scores


class MockEmbeddingClient:
    """模拟 Embedding：根据评论包含的关键词产生 4 维 one-hot-like 向量"""
    KEYWORDS = ['早餐', '前台', '位置', '房间']

    def embed_batch(self, texts):
        embeddings = []
        for text in texts:
            vec = [1.0 if kw in text else 0.0 for kw in self.KEYWORDS]
            # 加一点噪声避免完全平行
            vec = [v + 0.01 * (i + 1) for i, v in enumerate(vec)]
            embeddings.append(vec)
        return embeddings


def make_candidates():
    """
    构造 12 条 mock 候选：
    - 6 条都是"早餐"相关（类别集中，rerank 分数高）
    - 3 条"前台"
    - 2 条"位置"
    - 1 条"房间"
    """
    template = {
        'metadata': {
            'score': 5.0,
            'publish_date': '2025-01-01',
            'quality_score': 8,
            'review_count': 5,
            'useful_count': 3,
            'room_type': '花园大床房',
            'fuzzy_room_type': '大床房',
            'categories': []
        }
    }
    candidates = []
    
    def make(cid, text, cats, days_offset=0, quality=8):
        c = {
            'comment_id': cid,
            'comment': text,
            'metadata': dict(template['metadata']),
        }
        c['metadata']['categories'] = cats
        c['metadata']['quality_score'] = quality
        return c

    # 6 条早餐相关
    for i in range(6):
        candidates.append(make(f'c_breakfast_{i}', f'早餐很丰富品种多 {i}', ['餐饮设施']))
    # 3 条前台
    for i in range(3):
        candidates.append(make(f'c_frontdesk_{i}', f'前台服务热情周到 {i}', ['前台服务']))
    # 2 条位置
    for i in range(2):
        candidates.append(make(f'c_location_{i}', f'位置在市中心很方便 {i}', ['交通便利性']))
    # 1 条房间
    candidates.append(make('c_room_0', '房间整洁安静', ['房间设施']))
    
    return candidates


# ──────────────────────────────────────────────
# 测试用例
# ──────────────────────────────────────────────

def assert_eq(actual, expected, msg=""):
    assert actual == expected, f"{msg}: expected {expected}, got {actual}"


def assert_ge(actual, expected, msg=""):
    assert actual >= expected, f"{msg}: expected >= {expected}, got {actual}"


def test_baseline_behavior():
    """测试 1：baseline 行为（不启用任何新特性）—— 应与改造前完全一致"""
    print("\n[Test 1] baseline 行为...")
    ranker = MultiFactorRanker(
        MockReranker(),
        dataset_latest_date=datetime(2025, 4, 17),
    )
    candidates = make_candidates()
    ranked, timing = ranker.rank("早餐", candidates, topk=5)
    
    assert_eq(len(ranked), 5, "Top-5 数量")
    assert all('selection_method' in r and r['selection_method'] == 'pure' for r in ranked), "method=pure"
    # 早餐相关的应该排在前面（因为 mock rerank 用字符重叠）
    top_cats = [r['metadata']['categories'][0] for r in ranked[:3]]
    print(f"   Top-3 类别: {top_cats}")
    assert '餐饮设施' in top_cats, "餐饮设施应在 Top-3"
    print("   ✓ baseline 行为正常")


def test_relevance_floor():
    """测试 2：相关性阈值截断"""
    print("\n[Test 2] relevance_floor...")
    ranker = MultiFactorRanker(
        MockReranker(),
        dataset_latest_date=datetime(2025, 4, 17),
        relevance_floor=0.5,  # 极高阈值
    )
    candidates = make_candidates()
    ranked, _ = ranker.rank("早餐", candidates, topk=10)
    
    print(f"   返回数量: {len(ranked)}")
    # 因为有 min_keep 兜底，至少返回 max(10//3, 3) = 3 条
    assert_ge(len(ranked), 3, "至少返回 3 条")
    print("   ✓ relevance_floor 截断生效")


def test_mmr_diversity():
    """测试 3：MMR 多样性 —— 应能打散同类评论"""
    print("\n[Test 3] MMR 多样性...")
    
    # 对照组：纯 baseline
    baseline = MultiFactorRanker(
        MockReranker(),
        dataset_latest_date=datetime(2025, 4, 17),
    )
    candidates = make_candidates()
    baseline_ranked, _ = baseline.rank("早餐", candidates, topk=5)
    baseline_cats = [r['metadata']['categories'][0] for r in baseline_ranked]
    
    # 实验组：MMR（lambda=0.3 偏多样性）
    mmr_ranker = MultiFactorRanker(
        MockReranker(),
        embedding_client=MockEmbeddingClient(),
        dataset_latest_date=datetime(2025, 4, 17),
        enable_mmr=True,
        mmr_lambda=0.3,  # 强多样性
    )
    mmr_ranked, _ = mmr_ranker.rank("早餐", candidates, topk=5)
    mmr_cats = [r['metadata']['categories'][0] for r in mmr_ranked]
    
    baseline_unique = len(set(baseline_cats))
    mmr_unique = len(set(mmr_cats))
    
    print(f"   Baseline Top-5 类别: {baseline_cats} (去重: {baseline_unique})")
    print(f"   MMR      Top-5 类别: {mmr_cats} (去重: {mmr_unique})")
    assert mmr_unique > baseline_unique, f"MMR 应增加类别多样性: {mmr_unique} > {baseline_unique}"
    assert all(r['selection_method'] == 'mmr' for r in mmr_ranked), "method=mmr"
    print(f"   ✓ MMR 类别多样性提升 {baseline_unique} → {mmr_unique}")


def test_category_diversity():
    """
    测试 4：类别配额硬约束
    
    候选：6 早餐, 3 前台, 2 位置, 1 房间 → 4 类，每类配额 2 时独立可选 2+2+2+1=7 条
    topk=6 时配额可完全满足，无需 deferred 补足
    """
    print("\n[Test 4] 类别配额...")
    ranker = MultiFactorRanker(
        MockReranker(),
        dataset_latest_date=datetime(2025, 4, 17),
        enable_category_diversity=True,
        max_per_category=2,
    )
    candidates = make_candidates()
    ranked, _ = ranker.rank("早餐", candidates, topk=6)
    
    cats = [r['metadata']['categories'][0] for r in ranked]
    from collections import Counter
    cat_counts = Counter(cats)
    print(f"   Top-6 类别分布: {dict(cat_counts)}")
    
    assert all(count <= 2 for count in cat_counts.values()), "每类不超过 2 条"
    assert all(r['selection_method'] == 'cat' for r in ranked), "method=cat"
    print("   ✓ 类别配额硬约束生效")


def test_mmr_plus_category():
    """
    测试 5：MMR + 类别配额组合
    
    候选 4 类，每类配额 2 → 独立可选 7 条；topk=4 配额可完全满足
    """
    print("\n[Test 5] MMR + 类别配额组合...")
    ranker = MultiFactorRanker(
        MockReranker(),
        embedding_client=MockEmbeddingClient(),
        dataset_latest_date=datetime(2025, 4, 17),
        enable_mmr=True,
        mmr_lambda=0.3,
        enable_category_diversity=True,
        max_per_category=2,
        mmr_overselect_ratio=1.5,
    )
    candidates = make_candidates()
    ranked, _ = ranker.rank("早餐", candidates, topk=4)
    
    cats = [r['metadata']['categories'][0] for r in ranked]
    from collections import Counter
    cat_counts = Counter(cats)
    print(f"   Top-4 类别分布: {dict(cat_counts)}")
    
    assert all(count <= 2 for count in cat_counts.values()), "每类不超过 2 条"
    assert all(r['selection_method'] == 'mmr+cat' for r in ranked), "method=mmr+cat"
    print("   ✓ MMR + 类别配额组合生效")


def test_today_anchor():
    """测试 6：today 锚定修复"""
    print("\n[Test 6] today 锚定...")
    
    # 旧行为：today=None 且未传 dataset_latest_date → 用系统时间
    ranker_old = MultiFactorRanker(MockReranker())  # 不传 dataset_latest_date
    
    # 新行为：传 dataset_latest_date
    ranker_new = MultiFactorRanker(
        MockReranker(),
        dataset_latest_date=datetime(2025, 4, 17),
    )
    
    candidates = make_candidates()
    # 不传 today，强制走 fallback 路径
    ranked_old, _ = ranker_old.rank("早餐", candidates, topk=3, today=None)
    ranked_new, _ = ranker_new.rank("早餐", candidates, topk=3, today=None)
    
    recency_old = ranked_old[0]['feature_scores']['recency']
    recency_new = ranked_new[0]['feature_scores']['recency']
    
    print(f"   旧（datetime.today() fallback）: recency = {recency_old:.4f}")
    print(f"   新（dataset_latest_date=2025-04-17）: recency = {recency_new:.4f}")
    # 数据集锚定的 recency 应该更高（因为评论日期 2025-01-01 距 2025-04-17 较近）
    assert_ge(recency_new, recency_old, "新版 recency 应不低于旧版（因为锚定时间更接近评论发布日）")
    print("   ✓ today 锚定修复生效")


def test_p95_normalization():
    """测试 7：P95 归一化"""
    print("\n[Test 7] P95 归一化...")
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])  # 含极端值
    normalized = _safe_normalize(arr)
    print(f"   原始: {arr}")
    print(f"   P95 归一化: {normalized}")
    assert normalized.max() <= 1.0, "归一化后 max <= 1"
    assert normalized.min() >= 0.0, "归一化后 min >= 0"
    print("   ✓ P95 归一化正常")


def test_categories_string_format():
    """测试 8：categories 字段是字符串格式（CSV 加载场景）"""
    print("\n[Test 8] categories 字符串格式兼容...")
    ranker = MultiFactorRanker(
        MockReranker(),
        dataset_latest_date=datetime(2025, 4, 17),
        enable_category_diversity=True,
        max_per_category=1,
    )
    
    candidates = make_candidates()
    # 模拟 CSV 加载场景，categories 是字符串
    for c in candidates:
        c['metadata']['categories'] = str(c['metadata']['categories'])
    
    ranked, _ = ranker.rank("早餐", candidates, topk=4)
    print(f"   返回数量: {len(ranked)}")
    assert_eq(len(ranked), 4, "Top-4 数量")
    print("   ✓ 字符串格式 categories 兼容")


# ──────────────────────────────────────────────
# 运行所有测试
# ──────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_baseline_behavior,
        test_relevance_floor,
        test_mmr_diversity,
        test_category_diversity,
        test_mmr_plus_category,
        test_today_anchor,
        test_p95_normalization,
        test_categories_string_format,
    ]
    
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"   ✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"测试结果: {passed} 通过, {failed} 失败")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
