# 离线验证报告（Mock 数据）

> 本报告基于 [`test_metrics_demo.py`](../../rag-service/test_metrics_demo.py) 在 mock 数据上的实际运行结果，用于验证新功能的可用性。
> 真实评估需配置 DashScope/DashVector 凭据后运行 [`eval_rerank.py`](../../RAG/data/evaluation/eval_rerank.py)。

## 1. Mock 数据构造

| 类别 | 数量 | 期望相关性 label |
|------|------|------------------|
| 餐饮设施 | 6 | 3（highly relevant）|
| 前台服务 | 3 | 2（relevant）|
| 交通便利性 | 2 | 2（relevant）|
| 房间设施 | 1 | 1（marginal）|

Query: `"酒店有什么特色"` (典型模糊性问题，期望覆盖多个类别)

## 2. 五种 Ranker 配置对比（Top-5）

| 指标 | baseline | MMR λ=0.5 | MMR λ=0.3 | cat_max2 | mmr+cat |
|------|---------:|----------:|----------:|---------:|--------:|
| **NDCG@10** | 0.4759 | 0.4956 | 0.4920 | **0.5071** | 0.4956 |
| **Subtopic Recall** | 0.5000 | **1.0000** | **1.0000** | 0.7500 | **1.0000** |
| **Cat Coverage** | 0.1429 | **0.2857** | **0.2857** | 0.2143 | **0.2857** |
| **Cat Top-3 集中度** | 1.0000 | 0.8000 | 0.8000 | 1.0000 | 0.8000 |
| **ILD (文本)** | 0.5698 | **0.8539** | **0.8539** | 0.7652 | **0.8539** |

> 💡 **粗体表示该指标下的最优值**

## 3. 核心结论

### 3.1 baseline 的问题暴露

baseline Top-5 全部是"餐饮设施"类别（cat_coverage=14.29%，subtopic_recall=50%）—— **验证了"模糊性问题易出现单一类别集中"** 的预期问题。

### 3.2 MMR 显著提升多样性

| 维度 | 改进幅度 |
|------|---------|
| Subtopic Recall | **+100%**（覆盖全部 4 个意图方向）|
| Cat Coverage | **+100%**（14% → 29%）|
| ILD (文本) | **+50%** |
| **NDCG@10** | **+4.1%**（不降反升）|

### 3.3 类别配额是更平衡的选择

| 维度 | cat_max2 vs baseline |
|------|--------------------|
| Subtopic Recall | +50% |
| **NDCG@10** | **+6.6%**（最高）|
| Precision@10 | 持平（无下降）|

### 3.4 MMR + 类别配额组合

与纯 MMR(λ=0.5) 效果几乎一致，未带来额外多样性收益，但在真实数据上预计能起到兜底作用（防止 MMR 漏掉某些应有的类别）。

## 4. 与方案预测的对比

| 指标 | [plan.md](plan.md) 预期 | 实测（mock） | 结论 |
|------|------------------------|------------|------|
| ILD 提升 | +30% 以上 | **+50%** | ✓ 超额完成 |
| Category Coverage（模糊问题）| +20% 以上 | **+100%** | ✓ 超额完成 |
| Top-3 Category 占比 | -20% 以上 | **-20%** | ✓ 达标 |
| NDCG@10 不低于 baseline -2% | - | **+4.1%** | ✓ 不仅未掉，反而上升 |

## 5. 单元测试结果

`python3 rag-service/test_ranker_diversity.py` 全部通过：

```
[Test 1] baseline 行为...                       ✓
[Test 2] relevance_floor...                     ✓
[Test 3] MMR 多样性... (1 → 4 unique cats)      ✓
[Test 4] 类别配额...                            ✓
[Test 5] MMR + 类别配额组合...                  ✓
[Test 6] today 锚定... (0.2352 → 0.7449)        ✓
[Test 7] P95 归一化...                          ✓
[Test 8] categories 字符串格式兼容...           ✓

测试结果: 8 通过, 0 失败
```

## 6. 下一步

在配置 `DASHSCOPE_API_KEY` 和 `DASHVECTOR_API_KEY` 之后，按以下步骤跑真实评估：

```bash
cd RAG/data/evaluation

# 1) 构建 90 个 query 的 Top-100 候选池（一次性，约 5-10 分钟）
python3 eval_rerank.py --build-pool

# 2) 跑 7 个 ranker 变体的对比评估
python3 eval_rerank.py --eval

# 报告生成在 eval_rerank_report.md
```

可选：通过 LLM 标注 ground truth 后，NDCG/MRR/MAP 指标即可生效（待补充 `relevance_labels.json`）。
