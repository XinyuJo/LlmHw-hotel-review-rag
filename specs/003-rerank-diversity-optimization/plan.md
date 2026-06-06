# Rerank 多样性优化技术方案

> **版本**: v1.0  
> **作者**: 孔韫知  
> **日期**: 2026-06-06  
> **状态**: 实施中

---

## 1. 背景与动机

### 1.1 当前 Rerank 链路

[`MultiFactorRanker.rank()`](../../rag-service/modules/ranker.py:70-168) 当前的流程：

```
召回 Top-100 候选
    ↓
Qwen3-Rerank 打分（relevance_score）
    ↓
6 维特征归一化（relevance / quality / length / review / useful / recency）
    ↓
线性加权融合（0.40/0.25/0.05/0.05/0.05/0.20）
    ↓
按 final_score 降序排序 → 截断 Top-10
```

### 1.2 现状问题（按严重程度排序）

| # | 问题 | 严重度 | 位置 |
|---|------|--------|------|
| **P0-1** | **零多样性约束**：Top-10 纯按分数截断，模糊性问题（如"酒店有什么特色"）易出现 10 条全是"早餐"或"前台" | 高 | [`ranker.py:132`](../../rag-service/modules/ranker.py:132) |
| **P0-2** | **`today` 锚定错误**：fallback 用 `datetime.today()`，但数据集截止 2025/4/18，导致时效性分数随真实时间无意义衰减 | 高 | [`ranker.py:117-118`](../../rag-service/modules/ranker.py:117-118) |
| **P1-1** | **无相关性截断**：即便候选相关性都 <0.1 仍会塞 10 条进 LLM 上下文，污染回答 | 中 | [`ranker.py:141`](../../rag-service/modules/ranker.py:141) |
| **P1-2** | **归一化分母硬编码**（7.51/6.32/3.64），新数据分布漂移导致 norm > 1 | 中 | [`ranker.py:99/103/107`](../../rag-service/modules/ranker.py:99-107) |
| **P2-1** | **权重手工调参**，未基于评估集做学习 | 低 | [`ranker.py:46-51`](../../rag-service/modules/ranker.py:46-51) |
| **P2-2** | **缺少 Rerank 失败降级**，单点故障影响整条链路 | 低 | [`ranker.py:36-39`](../../rag-service/modules/ranker.py:36-39) |

### 1.3 评估能力缺口

现有评估（[`RAG/模型评估.md`](../../RAG/模型评估.md)）是**端到端 LLM-as-Judge 排名制消融**，无法：

- 单独评价 Rerank 模块的相关性质量（缺少 NDCG / MRR / MAP）
- 量化多样性维度（缺少 ILD / Category Coverage / Subtopic Recall）
- 支持快速迭代 ranker 参数（端到端评估每跑一轮成本约 720 次 LLM 调用）

---

## 2. 优化目标

### 2.1 业务目标

| 目标 | 指标 | 期望提升 |
|------|------|---------|
| 提升模糊性问题回答的全面性 | LLM-Judge 内容覆盖度排名 | 从 3.17 → ≤2.8 |
| 减少 Top-10 评论冗余 | ILD（类别 + 文本两类） | 类别 ILD: +30% 以上 |
| 降低无关评论污染 LLM 上下文 | 上下文中 relevance < 0.15 的比例 | ≤5% |

### 2.2 工程目标

- **零破坏性**：所有改造默认关闭，老调用方完全不受影响
- **可观测**：新引入特征/分数都进入 `feature_scores`，便于评估可视化
- **可评估**：搭建离线评估脚手架，**单次评估全量 <5 分钟**（不调 LLM 生成）

---

## 3. 详细方案

### 3.1 P0 修复：3 个一行 bug

#### 3.1.1 `today` 锚定

```python
# ranker.py 修改前
if not today:
    today = datetime.today()

# ranker.py 修改后
if today is None:
    # 优先使用构造时传入的数据集最新日期；否则降级到系统时间
    today = self.dataset_latest_date or datetime.today()
```

在 `MultiFactorRanker.__init__` 增加参数 `dataset_latest_date: datetime | None = None`，由 [`rag_system.py`](../../rag-service/modules/rag_system.py) 在初始化时传入 `TODAY = datetime(2025, 4, 17)`（来自 [`config.py:6`](../../rag-service/config.py:6)）。

#### 3.1.2 相关性阈值截断

新增参数 `relevance_floor: float = 0.0`，在按 `final_score` 排序前过滤：

```python
if self.relevance_floor > 0:
    keep_mask = relevance_score >= self.relevance_floor
    # 兜底：至少保留 max(topk//3, 3) 条
    min_keep = max(topk // 3, 3)
    if keep_mask.sum() < min_keep:
        keep_idx = np.argsort(relevance_score)[::-1][:min_keep]
        keep_mask = np.zeros_like(keep_mask, dtype=bool)
        keep_mask[keep_idx] = True
```

推荐默认值 `relevance_floor = 0.15`（生产灰度后再调）。

#### 3.1.3 归一化分母自适应

把硬编码常数替换为 P95：

```python
def _safe_normalize(x, eps=1e-8):
    """基于 P95 的鲁棒归一化"""
    p95 = np.percentile(x, 95)
    return np.clip(x / (p95 + eps), 0, 1)

norm_length = _safe_normalize(log_comment_len)
norm_review = _safe_normalize(log_review_count)
norm_useful = _safe_normalize(log_useful_count)
```

### 3.2 核心改造：MMR 多样性选择

#### 3.2.1 算法

$$d^* = \arg\max_{d \in R \setminus S} \left[ \lambda \cdot \widehat{\text{score}}(d) - (1-\lambda) \cdot \max_{d_j \in S} \text{sim}(d, d_j) \right]$$

其中：
- $\widehat{\text{score}}(d)$：`final_score` 的 min-max 归一化（避免与相似度量纲不一致）
- $\text{sim}(d, d_j)$：cosine 相似度（embedding 已 L2 归一化）
- $\lambda \in [0, 1]$：相关性 vs 多样性权衡，默认 0.7

#### 3.2.2 实现

新增 `MultiFactorRanker._mmr_select()` 方法：

```python
def _mmr_select(self, final_score, embeddings, topk, lambda_=0.7):
    """MMR 贪心选择"""
    n = len(final_score)
    if n <= topk:
        return list(np.argsort(final_score)[::-1])
    
    # min-max 归一化 final_score
    s_min, s_max = final_score.min(), final_score.max()
    s = (final_score - s_min) / (s_max - s_min + 1e-8)
    
    # L2 归一化 + 预计算相似度矩阵
    emb = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
    sim_matrix = emb @ emb.T
    np.fill_diagonal(sim_matrix, -np.inf)  # 避免自身参与
    
    selected = [int(np.argmax(s))]
    remaining = set(range(n)) - {selected[0]}
    
    while len(selected) < topk and remaining:
        rem_arr = np.array(list(remaining))
        max_sim_to_selected = sim_matrix[rem_arr][:, selected].max(axis=1)
        mmr_score = lambda_ * s[rem_arr] - (1 - lambda_) * max_sim_to_selected
        chosen = int(rem_arr[np.argmax(mmr_score)])
        selected.append(chosen)
        remaining.remove(chosen)
    
    return selected
```

#### 3.2.3 Embedding 来源

为避免 ranker 单独调一次 Embedding API（增加 ~300ms 延迟），有三种方案：

| 方案 | 改动量 | 延迟 | 选择 |
|------|--------|------|------|
| A: retriever 透传 embedding | 大（要修 4 路召回都收集） | 0 | ❌ |
| B: ranker 按需补 embed（候选评论文本一次 batch 调用） | 小 | +200~400ms | ✅ |
| C: 从 DashVector 反查 | 中 | +100ms | 备选 |

**选择方案 B**：`MultiFactorRanker` 接收可选的 `embedding_client`，在 `enable_mmr=True` 时统一对 100 条候选评论做一次 batch embed。

### 3.3 辅助改造：类别配额硬约束（兜底）

利用 [`enriched_comments.csv`](../../RAG/data/processed/enriched_comments.csv) 中的 `categories` 字段（每条评论 1-3 个二级小类标签），新增：

```python
def _category_diverse_filter(self, candidates, sorted_indices, topk, max_per_cat=2):
    """从有序候选中按类别配额选 Top-K"""
    cat_count = defaultdict(int)
    selected = []
    for idx in sorted_indices:
        cats = candidates[idx]['metadata'].get('categories', [])
        primary_cat = cats[0] if cats else 'unknown'
        if cat_count[primary_cat] >= max_per_cat:
            continue
        selected.append(idx)
        cat_count[primary_cat] += 1
        if len(selected) >= topk:
            break
    return selected
```

**两个模式可叠加**：MMR 选 15 条 → 类别配额裁剪到 10 条。

### 3.4 接口设计

`MultiFactorRanker.__init__` 新增参数（**全部可选 + 默认关闭**）：

```python
def __init__(self, reranker,
             embedding_client=None,                  # ← 新增（启用 MMR 时必需）
             # 既有特征权重 ...
             # 既有时效性参数 ...
             # 新增 P0 修复
             dataset_latest_date: datetime | None = None,
             relevance_floor: float = 0.0,
             use_p95_normalization: bool = False,
             # 新增 MMR
             enable_mmr: bool = False,
             mmr_lambda: float = 0.7,
             # 新增类别配额
             enable_category_diversity: bool = False,
             max_per_category: int = 2,
             # 控制顺序：先 MMR 选 K' 条，再按类别配额裁剪到 K 条
             mmr_overselect_ratio: float = 1.5):
```

`rank()` 返回结果新增字段：

```python
result['feature_scores']['mmr_score'] = ...      # 仅当 enable_mmr 时
result['selection_method'] = 'pure' | 'mmr' | 'mmr+cat' | 'cat'
```

### 3.5 调用链改动

`rag_system.py` 的 `query()` 和 `query_stream()` 新增参数透传：

```python
def query(..., 
          enable_mmr: bool = False, mmr_lambda: float = 0.7,
          enable_category_diversity: bool = False, max_per_category: int = 2,
          relevance_floor: float = 0.0):
    ...
    ranker = MultiFactorRanker(
        self.reranker,
        embedding_client=self.embedding_client if enable_mmr else None,
        ...,
        dataset_latest_date=TODAY,
        relevance_floor=relevance_floor,
        enable_mmr=enable_mmr, mmr_lambda=mmr_lambda,
        enable_category_diversity=enable_category_diversity,
        max_per_category=max_per_category,
    )
```

---

## 4. 评估体系

### 4.1 评估指标库

新建 [`RAG/data/evaluation/metrics.py`](../../RAG/data/evaluation/metrics.py)，实现以下指标：

#### 4.1.1 相关性类（需要 ground truth 标注）

| 指标 | 公式 | 范围 | 说明 |
|------|------|------|------|
| **NDCG@K** | $\frac{DCG@K}{IDCG@K}$ | [0, 1] | 标准相关性指标 |
| **MRR@K** | $\frac{1}{\text{rank of first relevant}}$ | [0, 1] | 首条相关位置 |
| **MAP@K** | $\frac{1}{R}\sum \text{Precision@k}$ | [0, 1] | 平均精确度 |

**ground truth 获取**：用 Qwen3-Max 对 90 个评估问题的召回结果做 0-3 四级相关性弱标注（一次性，~9000 次 LLM 调用，约 ¥50）。结果缓存到 `evaluation/relevance_labels.json`。

#### 4.1.2 多样性类（无需标注）

| 指标 | 公式 | 范围 | 说明 |
|------|------|------|------|
| **ILD (文本)** | $\frac{2}{K(K-1)} \sum_{i<j} (1 - \cos(e_i, e_j))$ | [0, 1] | 列表内 embedding 平均差异 |
| **Category Coverage** | $\|\bigcup_i \text{cats}(d_i)\| / 14$ | [0, 1] | 覆盖了几个小类 |
| **Category Gini** | 类别分布的 Gini 系数 | [0, 1] | 类别分布均衡度（越低越均衡）|
| **Top-3 Category 占比** | $\sum_{c \in top3} \text{count}(c) / K$ | [0, 1] | 前 3 多类别的集中度（越低越好）|

#### 4.1.3 端到端类（可选，调 LLM）

| 指标 | 说明 |
|------|------|
| **Answer Coverage** | LLM 判断回答覆盖了用户问题的几个子意图 |
| **Hallucination Rate** | LLM 判断回答中是否有未在 reference 中出现的信息 |

### 4.2 评估脚本

新建 [`RAG/data/evaluation/eval_rerank.py`](../../RAG/data/evaluation/eval_rerank.py)：

```python
def evaluate_ranker_variants(eval_set, candidates_pool, variants_config):
    """
    对多个 ranker 配置变体跑离线评估
    
    eval_set: 评估问题集 (含 question, intent_direction)
    candidates_pool: 召回阶段输出的 Top-100 候选（一次跑好缓存）
    variants_config: { 'baseline': {...}, 'mmr_0.7': {...}, ... }
    """
    results = defaultdict(lambda: defaultdict(list))
    
    for variant_name, ranker_kwargs in variants_config.items():
        ranker = MultiFactorRanker(**ranker_kwargs)
        for sample in eval_set:
            candidates = candidates_pool[sample['question_id']]
            ranked, _ = ranker.rank(sample['question'], candidates, topk=10)
            
            results[variant_name]['ndcg@10'].append(
                ndcg_at_k(ranked, relevance_labels[sample['question_id']], k=10)
            )
            results[variant_name]['ild'].append(intra_list_diversity(ranked))
            results[variant_name]['cat_coverage'].append(category_coverage(ranked))
            # ... 其他指标
    
    return results
```

### 4.3 评估实验设计

#### 实验 A：MMR Lambda 扫描

| 配置 | `enable_mmr` | `mmr_lambda` | 目标 |
|------|--------------|--------------|------|
| baseline | False | - | 当前线上版本 |
| mmr_0.9 | True | 0.9 | 弱多样性 |
| mmr_0.7 | True | 0.7 | 推荐起点 |
| mmr_0.5 | True | 0.5 | 中等多样性 |
| mmr_0.3 | True | 0.3 | 强多样性 |

**画图**：横轴 ILD，纵轴 NDCG@10，找拐点。

#### 实验 B：类别配额扫描

| 配置 | `max_per_category` |
|------|---------------------|
| baseline | ∞ |
| cat_3 | 3 |
| cat_2 | 2 |
| cat_1 | 1 |

#### 实验 C：组合策略

| 配置 | 说明 |
|------|------|
| baseline | 纯线性加权 |
| mmr_only | MMR (λ=0.7) |
| cat_only | 类别配额 (max=2) |
| mmr+cat | MMR 选 15 → 类别配额裁到 10 |

#### 实验 D：分场景验证

将评估集按 `question_type` 切分：
- 结构化问题（n=70）：期望多样性收益小，相关性应基本不掉
- 模糊性问题（n=10）：期望多样性收益大
- 时效性问题（n=10）：验证 today 修复后 recency 是否合理

---

## 5. 实施计划

### 5.1 任务拆解

| 阶段 | 任务 | 文件 | 工作量 |
|------|------|------|--------|
| **P0** | 修复 `today` 锚定 | [`ranker.py`](../../rag-service/modules/ranker.py) | 0.5h |
| **P0** | 加相关性阈值 + 自适应归一化 | [`ranker.py`](../../rag-service/modules/ranker.py) | 1h |
| **P1** | 实现 `_mmr_select` | [`ranker.py`](../../rag-service/modules/ranker.py) | 2h |
| **P1** | 实现 `_category_diverse_filter` | [`ranker.py`](../../rag-service/modules/ranker.py) | 1h |
| **P1** | `rag_system.py` 透传参数 | [`rag_system.py`](../../rag-service/modules/rag_system.py) | 0.5h |
| **P2** | 评估指标库 | `RAG/data/evaluation/metrics.py` | 2h |
| **P2** | 评估脚本框架 | `RAG/data/evaluation/eval_rerank.py` | 2h |
| **P2** | Demo / 单元测试 | `tests/test_ranker_diversity.py` | 1h |

**总计**：约 10 工时

### 5.2 灰度策略

1. **Day 1**：合入 P0 修复，默认参数全部 0（行为不变），观察一周
2. **Day 8**：开启 `relevance_floor=0.15`，对比上下文质量
3. **Day 15**：开启 `enable_mmr=True, mmr_lambda=0.7`，跑 A/B 评估
4. **Day 22**：根据实验 A 结果固化最佳 lambda

### 5.3 回滚预案

所有新参数默认为关闭/0，回滚只需在 `rag_system.py` 调用时去除对应参数即可。

---

## 6. 风险与权衡

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| MMR 引入 +300ms 延迟（Embedding 调用）| 用户感知 | 异步并行 / 启用缓存 / 仅模糊性问题启用 |
| 多样性过强导致相关性掉点 | 回答质量 | 通过实验 A 找最佳 λ，灰度发布 |
| LLM 标注 ground truth 有偏 | 评估失真 | 多模型交叉标注 / 抽样人工复核 |
| 类别配额对小类别评论压制过强 | 长尾类缺失 | `max_per_category=2` 而非 1，且仅作 MMR 之后的兜底 |

---

## 7. 验收标准

### 7.1 功能验收

- [ ] `MultiFactorRanker` 新参数全部可独立开关
- [ ] `enable_mmr=True` 时输出包含 `feature_scores['mmr_score']`
- [ ] 单元测试覆盖：纯 baseline、纯 MMR、纯类别配额、MMR+类别配额 4 种模式

### 7.2 指标验收（基于评估实验）

| 指标 | baseline | 目标 |
|------|---------|------|
| NDCG@10（全局） | TBD | 不低于 baseline -2% |
| ILD（全局） | TBD | +30% 以上 |
| Category Coverage（模糊性问题）| TBD | +20% 以上 |
| Top-3 Category 占比 | TBD | -20% 以上 |

### 7.3 评估验收

- [ ] `eval_rerank.py` 可一键跑出 baseline vs 各 variants 的全指标对比表
- [ ] 输出 markdown 格式报告 + 多样性-相关性 trade-off 曲线图

---

## 8. 后续工作（不在本次范围）

- **LightGBM-LambdaRank**：用评估集的 ranking 标签训练 LTR 模型替代线性加权
- **DPP**：如果 MMR 仍不满足多样性需求，考虑 DPP（更精确但更慢）
- **意图感知动态权重**：根据 question_type 动态调整 `mmr_lambda`（结构化问题→0.9，模糊性问题→0.5）
- **Cross-encoder fine-tuning**：基于本场景标注数据微调 Qwen3-Rerank 或换更小的本地模型降本
