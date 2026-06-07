"""对 HyDE 相关配置进行匿名排名评估。"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]
SERVICE_DIR = ROOT / "rag-service"
sys.path.insert(0, str(SERVICE_DIR))

from modules.clients import LLMClient  # noqa: E402


DIMENSIONS = [
    "intent_understanding", "content_coverage", "opinion_balance",
    "traceability", "temporal_awareness", "expression_quality", "overall",
]


def normalize_result(raw: dict) -> dict:
    """兼容 notebook 旧格式和 run_hyde_responses.py 新格式。"""
    if "cited_comments" in raw:
        return raw

    refs = raw.get("references", {}) or {}
    comments = []
    for c in refs.get("comments", [])[:8]:
        meta = c.get("metadata", {})
        comments.append({
            "comment": c.get("comment", ""),
            "publish_date": meta.get("publish_date", ""),
            "score": meta.get("score", ""),
            "room_type": meta.get("room_type", ""),
        })
    summaries = []
    for s in refs.get("summaries", [])[:3]:
        meta = s.get("metadata", {})
        summaries.append({
            "category": meta.get("category", ""),
            "keywords": meta.get("keywords", ""),
            "summary": s.get("summary", s.get("content", "")),
        })
    return {
        **raw,
        "cited_comments": comments,
        "cited_summaries": summaries,
    }


def load_results(configs: list[str], response_dir: Path) -> dict[str, list[dict]]:
    all_results = {}
    for config in configs:
        path = response_dir / f"responses_{config}.json"
        rows = json.loads(path.read_text(encoding="utf-8"))
        all_results[config] = [normalize_result(row) for row in rows]
        print(f"{config}: {len(rows)} rows from {path}")
    return all_results


def build_ranking_prompt(question: str, responses: dict[str, dict], config_order: list[str]) -> str:
    labels = [chr(65 + i) for i in range(len(config_order))]
    responses_block = ""

    for label, config_name in zip(labels, config_order):
        r = responses[config_name]
        if r.get("error"):
            responses_block += f"\n系统{label}\n[该系统未能生成回复: {r.get('error')}]\n"
            continue

        cited_info = ""
        if r.get("cited_comments"):
            cited_info += "\n[该系统检索到的用户评论]\n"
            for ci, c in enumerate(r["cited_comments"], 1):
                cited_info += (
                    f"评论{ci}（发布于{c.get('publish_date', '')}，"
                    f"评分: {c.get('score', '')}分，房型: {c.get('room_type', '')}）：\n"
                    f"{c.get('comment', '')}\n\n"
                )
        if r.get("cited_summaries"):
            cited_info += "[该系统检索到的评论摘要]\n"
            for s in r["cited_summaries"]:
                cited_info += (
                    f"[{s.get('category', '')}类别摘要]\n"
                    f"关键词: {s.get('keywords', '')}\n"
                    f"内容: {s.get('summary', '')}\n\n"
                )

        responses_block += f"\n系统{label}\n{r.get('response', '')}\n{cited_info}"

    n = len(config_order)
    return f"""
你是一位严格的酒店问答 RAG 系统评估专家。
下面给出了同一个用户问题在 {n} 个不同系统配置下生成的回复。请在每个评估维度上，对这 {n} 个回复从优到劣排序。

【评估背景】
- 基于真实住客评论的酒店智能客服系统（广州花园酒店）
- 用户提问时间为 2025年4月18日
- 各系统从同一评论知识库中检索信息并生成回复，但 HyDE 生成器或是否启用 HyDE 不同
- 评论是住客主观体验，没有唯一标准答案；好的回答应综合多条评论给出全面客观的参考意见
- 每个系统的回复后附有其检索到的原始评论和摘要，供你核查引用是否准确、有无幻觉

【用户问题】
{question}

【各系统的回复】
{responses_block}

【评估维度说明】
1. 意图理解度：是否真正理解用户深层需求，回答是否有针对性。
2. 内容覆盖度：是否充分覆盖问题涉及的各方面，信息是否具体。
3. 观点平衡性：是否客观呈现正反评价，避免只说好话或坏话。
4. 引用溯源性：关键信息是否能从检索到的评论或摘要中找到依据，是否存在幻觉。
5. 时效合理性：是否合理利用评论时间信息，时效敏感问题是否优先近期评论。
6. 表达专业度：结构是否清晰，语言是否精炼专业，有无冗余模板话术。

【排名规则】
- 每个维度独立排名，1 = 最优，{n} = 最差
- 不允许并列
- 未能生成回复的系统自动排末位
- 综合排名 overall 基于六个维度整体权衡

【输出格式】
严格以 JSON 格式输出，值为系统标签按排名排列的数组（第一个最优，最后一个最差）。不要有任何额外说明：
{{
    "intent_understanding": ["{labels[0]}", "{labels[1]}", "..."],
    "content_coverage": ["{labels[0]}", "{labels[1]}", "..."],
    "opinion_balance": ["{labels[0]}", "{labels[1]}", "..."],
    "traceability": ["{labels[0]}", "{labels[1]}", "..."],
    "temporal_awareness": ["{labels[0]}", "{labels[1]}", "..."],
    "expression_quality": ["{labels[0]}", "{labels[1]}", "..."],
    "overall": ["{labels[0]}", "{labels[1]}", "..."]
}}
"""


def rank_one(eval_llm: LLMClient, question: dict, all_results: dict[str, list[dict]]) -> dict:
    config_names = list(all_results.keys())
    responses = {}
    for config_name in config_names:
        result = next((r for r in all_results[config_name] if r["question_id"] == question["question_id"]), None)
        responses[config_name] = result or {"question_id": question["question_id"], "error": "未找到结果", "response": ""}

    config_order = config_names.copy()
    random.shuffle(config_order)
    labels = [chr(65 + i) for i in range(len(config_order))]
    label_to_config = {label: config for label, config in zip(labels, config_order)}
    prompt = build_ranking_prompt(question["question"], responses, config_order)

    for attempt in range(3):
        try:
            response = eval_llm.generate(prompt, temperature=0.1)
            response = response.replace("```json", "").replace("```", "").strip()
            rankings_raw = json.loads(response)

            rankings = {}
            for dim in DIMENSIONS:
                ranked_labels = rankings_raw[dim]
                if len(ranked_labels) != len(config_order):
                    raise ValueError(f"{dim}: 排名数量错误")
                rankings[dim] = {
                    label_to_config[label]: rank
                    for rank, label in enumerate(ranked_labels, 1)
                    if label in label_to_config
                }
                if len(rankings[dim]) != len(config_order):
                    raise ValueError(f"{dim}: 非法标签")

            return {
                "question_id": question["question_id"],
                "config_order": config_order,
                "label_to_config": label_to_config,
                "rankings": rankings,
                "error": None,
            }
        except Exception as exc:
            if attempt < 2:
                time.sleep(2)
                continue
            return {
                "question_id": question["question_id"],
                "config_order": config_order,
                "label_to_config": label_to_config,
                "rankings": None,
                "error": str(exc),
            }


def summarize(rankings: list[dict], configs: list[str]) -> dict:
    summary = {}
    for dim in DIMENSIONS:
        dim_summary = {}
        for config in configs:
            vals = [
                row["rankings"][dim][config]
                for row in rankings
                if row.get("rankings") and config in row["rankings"][dim]
            ]
            dim_summary[config] = round(sum(vals) / len(vals), 4) if vals else None
        summary[dim] = dict(sorted(dim_summary.items(), key=lambda kv: kv[1] if kv[1] is not None else 999))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--response-dir", default="RAG/data/evaluation")
    parser.add_argument("--eval-set", default="RAG/data/evaluation/eval_set.json")
    parser.add_argument("--output", default="RAG/data/evaluation/hyde_ranking_results.json")
    parser.add_argument("--summary-output", default="RAG/data/evaluation/hyde_ranking_summary.json")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--model", default="qwen-max")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    key = os.getenv("DASHSCOPE_INTL_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
    if not key:
        raise RuntimeError("缺少 DASHSCOPE_API_KEY")

    questions = json.loads((ROOT / args.eval_set).read_text(encoding="utf-8"))[:90]
    if args.limit:
        questions = questions[:args.limit]
    all_results = load_results(args.configs, ROOT / args.response_dir)
    eval_llm = LLMClient(key, model=args.model, json=True)

    rankings = []
    for question in tqdm(questions, desc="ranking"):
        rankings.append(rank_one(eval_llm, question, all_results))

    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rankings, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = summarize(rankings, args.configs)
    (ROOT / args.summary_output).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary["overall"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
