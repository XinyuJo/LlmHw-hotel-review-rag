# HyDE + SFT 实验报告

## 1. 实验目标

本实验评估 HyDE 生成模型对酒店评论 RAG 的影响。由于原 `full` 配置使用的 HyDE 生成模型是 DashScope 闭源模型，本次额外补充两类开源对照：

| 配置 | HyDE 生成器 | 目的 |
|---|---|---|
| `no_hyde` | 关闭 HyDE | 无假设文档增强基线 |
| `full` | DashScope 通用 LLM HyDE | 原完整系统基线 |
| `qwen3_base_hyde` | 本地 vLLM 部署 Qwen3-4B-Instruct | 开源通用模型替换 |
| `qwen3_sft_hyde_r8/r16/r32` | Qwen3-4B-Instruct + LoRA | 验证酒店评论领域 SFT 的收益 |

## 2. HyDE 技术原理

HyDE（Hypothetical Document Embeddings）的核心思想是：先让 LLM 根据用户查询生成若干条“假设相关文档”，再对这些假设文档做 embedding 检索。相比直接对短查询向量化，假设文档通常包含更完整的评价对象、体验细节和情感倾向，可以拉近用户问题与真实评论之间的语义距离。

本项目中，`HyDEGenerator.generate()` 会为每个改写后的 query 生成 3 条假设酒店评论，随后系统对这些假设评论做向量检索，并与 BM25、原始向量、反向 Query、摘要召回结果做 RRF 融合。

## 3. SFT 数据构造

数据来源为 `RAG/data/processed/reverse_queries.csv`。每条样本包含反向 Query、对应评论 ID、原始评论和房型信息。构造过程如下：

1. 按 `(query, comment_id)` 去重，并跳过空 query、空评论和过短目标。
2. 调用 API 从原始评论中抽取仅与 query 相关的短片段，限制在 20-160 字。
3. 保存为对话式 SFT JSONL，并自动切分 train/valid。
4. 支持 `--resume` 和多 worker 并发，长任务中断后可继续跑。

本次实际产物：

| 指标 | 数值 |
|---|---:|
| 有效样本数 | 5528 |
| 训练集 | 4976 |
| 验证集 | 552 |
| 覆盖唯一评论 ID | 2126 |
| 平均目标片段长度 | 40.8 字 |
| 目标片段长度范围 | 20-160 字 |

房型分布：

| 房型 | 样本数 |
|---|---:|
| 大床房 | 2082 |
| 双床房 | 2087 |
| 套房 | 1322 |
| 主题房 | 37 |

数据构造命令：

```bash
python RAG/sft_hyde/build_hyde_sft_dataset.py \
  --input RAG/data/processed/reverse_queries.csv \
  --output RAG/data/evaluation/hyde_sft_full.jsonl \
  --use-api \
  --max-chars 160 \
  --min-chars 20 \
  --resume \
  --shuffle \
  --seed 42 \
  --valid-ratio 0.1 \
  --workers 8
```

## 4. SFT 训练设置

训练使用远程机器 `/root/Exp4/Qwen3-4B-Instruct-2507` 作为基座模型，并基于 LoRA 做三组 rank 对比。

| 项 | 设置 |
|---|---|
| Base model | Qwen3-4B-Instruct |
| 微调方式 | LoRA |
| LoRA rank | 8 / 16 / 32 |
| max steps | 400 |
| train / valid | 4976 / 552 |
| 权重目录 | `/root/sxy/hotel-review-rag/RAG/data/evaluation/hyde_sft_lora_r{8,16,32}` |

训练结果：

| Rank | Train loss | Valid loss | 训练时长 |
|---:|---:|---:|---:|
| 8 | 2.7738 | 2.5111 | 785.5s |
| 16 | 2.7242 | 2.4854 | 779.9s |
| 32 | 2.6820 | 2.4599 | 801.3s |

从验证集 loss 看，rank 越大越好，`r32` 最低；但最终 RAG ranking 中 `r8` 表现最好，说明低验证 loss 不一定直接等价于 RAG 端到端效果提升。

## 5. vLLM 部署与接入

本次 vLLM 服务部署在远程机器上，因此 `http://127.0.0.1:8000/v1` 只能被远程机器本机访问。RAG 响应生成脚本也必须在远程机器上运行。

Qwen3 的默认上下文长度较大，在 24GB 显卡上需要限制 KV cache：

```bash
MODEL_PATH=/root/Exp4/Qwen3-4B-Instruct-2507 \
SERVED_MODEL_NAME=Qwen3-4B-Instruct \
MAX_MODEL_LEN=4096 \
GPU_MEMORY_UTILIZATION=0.85 \
bash RAG/sft_hyde/serve_vllm_base.sh
```

部署 LoRA：

```bash
MODEL_PATH=/root/Exp4/Qwen3-4B-Instruct-2507 \
LORA_PATH=RAG/data/evaluation/hyde_sft_lora_r8 \
SERVED_MODEL_NAME=hyde_sft_r8 \
MAX_MODEL_LEN=4096 \
MAX_LORA_RANK=32 \
GPU_MEMORY_UTILIZATION=0.85 \
bash RAG/sft_hyde/serve_vllm_lora.sh
```

RAG 服务接入方式：

```bash
HYDE_BACKEND=vllm
HYDE_VLLM_BASE_URL=http://127.0.0.1:8000/v1
HYDE_VLLM_MODEL=hyde_sft_r8
HYDE_VLLM_API_KEY=EMPTY
```

代码上，`HotelReviewRAG` 根据 `hyde_backend` 为 HyDE 单独注入 `OpenAICompatibleLLMClient`；意图扩展、embedding、rerank 和最终回答仍使用原 DashScope 链路。

## 6. 消融评估

评估集共 90 个问题。每个配置均成功生成 90 条回答，错误数为 0。评估方式为匿名多系统 ranking，平均名次越低越好。

### Overall

| 配置 | 平均综合排名 |
|---|---:|
| `qwen3_sft_hyde_r8` | 3.2000 |
| `qwen3_sft_hyde_r16` | 3.2444 |
| `qwen3_base_hyde` | 3.3000 |
| `qwen3_sft_hyde_r32` | 3.3667 |
| `no_hyde` | 3.8556 |
| `full` | 4.0333 |

### 分维度结果

| 维度 | 最优配置 | 平均名次 |
|---|---|---:|
| 意图理解 | `qwen3_sft_hyde_r8` | 3.2000 |
| 内容覆盖 | `qwen3_sft_hyde_r8` | 3.2000 |
| 观点平衡 | `qwen3_sft_hyde_r8` | 3.2889 |
| 引用溯源 | `qwen3_sft_hyde_r8` | 3.2111 |
| 时效合理 | `qwen3_sft_hyde_r8` | 3.2333 |
| 表达质量 | `qwen3_sft_hyde_r8` | 3.2222 |

完整结果见：

```text
RAG/data/evaluation/hyde_ranking_summary.json
RAG/data/evaluation/hyde_ranking_results.json
```

## 7. 结论

1. 开源 Qwen3-4B-Instruct 作为 HyDE 生成器可以跑通本地 vLLM 推理，并在本次匿名 ranking 中优于无 HyDE 基线。
2. SFT 后的 `r8` 和 `r16` 均优于 Qwen3 base，说明基于真实酒店评论构造的假设文档训练数据能提升端到端 RAG 效果。
3. `r32` 的验证 loss 最低，但 ranking 不如 `r8/r16`，可能是更高 rank 使生成评论更贴近训练片段但多样性下降，或输出长度/风格对检索融合不一定最优。
4. 最终推荐采用 `qwen3_sft_hyde_r8` 作为当前 SFT HyDE 配置：端到端 ranking 最好，参数量更小，部署成本也低于 r16/r32。

## 8. 复现命令

生成 Qwen3 base 响应：

```bash
python RAG/sft_hyde/run_hyde_responses.py \
  --name qwen3_base_hyde \
  --hyde-backend vllm \
  --vllm-base-url http://127.0.0.1:8000/v1 \
  --vllm-model Qwen3-4B-Instruct \
  --workers 2 \
  --retries 3
```

生成 SFT 响应：

```bash
python RAG/sft_hyde/run_hyde_responses.py \
  --name qwen3_sft_hyde_r8 \
  --hyde-backend vllm \
  --vllm-base-url http://127.0.0.1:8000/v1 \
  --vllm-model hyde_sft_r8 \
  --workers 2 \
  --retries 3
```

运行 ranking：

```bash
python RAG/sft_hyde/evaluate_hyde_rankings.py \
  --configs no_hyde full qwen3_base_hyde qwen3_sft_hyde_r8 qwen3_sft_hyde_r16 qwen3_sft_hyde_r32 \
  --output RAG/data/evaluation/hyde_ranking_results.json \
  --summary-output RAG/data/evaluation/hyde_ranking_summary.json \
  --model qwen-max
```
