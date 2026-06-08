"""
评估指标库 demo：用 test_ranker_diversity 的 mock 数据，跑出真实的多样性 vs 相关性指标

这个脚本演示了 metrics.py 的所有指标计算方式，并在 mock 数据上对比 baseline / MMR / 类别配额 三种策略。

运行：python3 test_metrics_demo.py
"""

import sys
import importlib.util
from datetime import datetime
from pathlib import Path

import numpy as np

# 加载 ranker
_ranker_path = Path(__file__).parent / "modules" / "ranker.py"
_spec = importlib.util.spec_from_file_location("ranker", _ranker_path)
_ranker_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ranker_module)
MultiFactorRanker = _ranker_module.MultiFactorRanker

# 加载 metrics（在 RAG/data/evaluation/ 下）
_metrics_path = Path(__file__).parent.parent / "RAG" / "data" / "evaluation" / "metrics.py"
_spec2 = importlib.util.spec_from_file_location("metrics", _metrics_path)
_metrics_module = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(_metrics_module)
evaluate_single = _metrics_module.evaluate_single
aggregate_metrics = _metrics_module.aggregate_metrics

# 复用 test_ranker_diversity 的 mock
from test_ranker_diversity import MockReranker, MockEmbeddingClient, make_candidates


def main():
    candidates = make_candidates()
    print(f"候选池：{len(candidates)} 条评论")
    cats_dist = {}
    for c in candidates:
        cat = c['metadata']['categories'][0]
        cats_dist[cat] = cats_dist.get(cat, 0) + 1
    print(f"类别分布：{cats_dist}")
    
    # 构造 ground truth 相关性标注（mock）
    relevance_labels = {}
    for c in candidates:
        cid = c['comment_id']
        cat = c['metadata']['categories'][0]
        # 假设 query="酒店有什么特色"，4 个类别都有一定相关性
        if cat == '餐饮设施':
            relevance_labels[cid] = 3
        elif cat == '前台服务':
            relevance_labels[cid] = 2
        elif cat == '交通便利性':
            relevance_labels[cid] = 2
        elif cat == '房间设施':
            relevance_labels[cid] = 1
        else:
            relevance_labels[cid] = 0
    
    # 三种策略对比
    variants = {
        "baseline": {
            "enable_mmr": False,
            "enable_category_diversity": False,
        },
        "mmr_lambda0.5": {
            "enable_mmr": True, "mmr_lambda": 0.5,
            "enable_category_diversity": False,
        },
        "mmr_lambda0.3": {
            "enable_mmr": True, "mmr_lambda": 0.3,
            "enable_category_diversity": False,
        },
        "cat_max2": {
            "enable_mmr": False,
            "enable_category_diversity": True, "max_per_category": 2,
        },
        "mmr_0.5+cat_2": {
            "enable_mmr": True, "mmr_lambda": 0.5,
            "enable_category_diversity": True, "max_per_category": 2,
        },
    }
    
    results = {}
    for variant_name, config in variants.items():
        ranker_kwargs = {
            'dataset_latest_date': datetime(2025, 4, 17),
            **config,
        }
        if config.get('enable_mmr'):
            ranker_kwargs['embedding_client'] = MockEmbeddingClient()
        
        ranker = MultiFactorRanker(MockReranker(), **ranker_kwargs)
        ranked, _ = ranker.rank("酒店有什么特色", candidates, topk=5)
        
        # 计算 embeddings（用于 ILD）
        emb_client = MockEmbeddingClient()
        embeddings = np.array(emb_client.embed_batch([r['comment'] for r in ranked]))
        
        # 模拟意图方向（用户问"酒店特色"，期望覆盖餐饮/前台/位置/房间）
        intent_directions = ['餐饮设施', '前台服务', '交通便利性', '房间设施']
        
        metrics = evaluate_single(
            ranked,
            relevance_labels=relevance_labels,
            embeddings=embeddings,
            intent_directions=intent_directions,
        )
        results[variant_name] = metrics
    
    # 打印对比表
    print("\n" + "="*100)
    print("评估指标对比（mock 数据，仅演示流程）")
    print("="*100)
    metric_keys = ['ndcg@10', 'mrr@10', 'precision@10', 'cat_coverage', 
                   'cat_gini', 'cat_top3_concentration', 'ild_text', 
                   'subtopic_recall', 'avg_rerank_score']
    
    header = f"{'变体':<20}" + "".join(f"{k:<15}" for k in metric_keys)
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        row = f"{name:<20}"
        for k in metric_keys:
            v = m.get(k, 0)
            row += f"{v:<15.4f}"
        print(row)
    
    # 关键发现
    baseline = results['baseline']
    print("\n" + "="*100)
    print("关键发现（vs baseline）")
    print("="*100)
    for variant in ['mmr_lambda0.5', 'mmr_lambda0.3', 'cat_max2', 'mmr_0.5+cat_2']:
        v = results[variant]
        print(f"\n[{variant}]")
        for k in metric_keys:
            delta = v[k] - baseline[k]
            sign = "+" if delta >= 0 else ""
            pct = (delta / (baseline[k] + 1e-8) * 100)
            arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
            print(f"  {k:<25} {baseline[k]:.4f} → {v[k]:.4f}  ({sign}{delta:.4f}, {sign}{pct:.1f}%) {arrow}")


if __name__ == "__main__":
    main()
