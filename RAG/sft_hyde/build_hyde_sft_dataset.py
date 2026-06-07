"""构造 HyDE SFT 数据集。

输入来自 reverse_queries.csv: query -> 原始评论。
输出 JSONL: {"conversations": [{"role": "user", ...}, {"role": "assistant", ...}]}
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


SYSTEM_PREFIX = "请根据用户查询生成一段简短、真实、和查询高度相关的酒店评论假设文档。"


def split_sentences(text: str) -> list[str]:
    """按中文标点切分评论，保留足够短的片段用于 HyDE。"""
    text = re.sub(r"\s+", " ", str(text)).strip()
    parts = re.split(r"(?<=[。！？!?；;])", text)
    return [p.strip() for p in parts if p.strip()]


def keyword_fallback_extract(query: str, comment: str, max_chars: int) -> str:
    """无 API 时的保底抽取：选取和 query 字符重叠最多的短句。"""
    sentences = split_sentences(comment)
    if not sentences:
        return str(comment)[:max_chars]

    query_chars = set(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", query))
    scored = []
    for sentence in sentences:
        chars = set(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", sentence))
        overlap = len(query_chars & chars)
        scored.append((overlap, len(sentence), sentence))

    selected = []
    total = 0
    for _, _, sentence in sorted(scored, key=lambda x: (-x[0], x[1])):
        if total + len(sentence) > max_chars and selected:
            continue
        selected.append(sentence)
        total += len(sentence)
        if total >= max_chars * 0.6:
            break

    return "".join(selected)[:max_chars] or str(comment)[:max_chars]


def dashscope_extract(query: str, comment: str, model: str, max_chars: int) -> str:
    """调用 DashScope 从原评论中抽取与 query 相关的片段。"""
    from dashscope import Generation

    prompt = f"""
你是酒店评论信息抽取助手。请只从【原始评论】中抽取与【用户查询】直接相关的内容，可以做少量压缩改写，但不要编造原评论没有的信息。

【用户查询】
{query}

【原始评论】
{comment}

【输出要求】
- 输出一段酒店评论风格文本
- 长度不超过 {max_chars} 个中文字符
- 只输出文本，不要解释
"""
    response = Generation.call(
        api_key=os.environ["DASHSCOPE_API_KEY"],
        model=model,
        prompt=prompt,
        temperature=0.1,
        result_format="message",
    )
    if response.status_code != 200:
        raise RuntimeError(response.message)
    return response.output.choices[0].message.content.strip()[:max_chars]


def stable_id(query: str, comment_id: str, comment: str) -> str:
    raw = f"{query}\t{comment_id}\t{comment[:80]}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def load_done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                item_id = item.get("metadata", {}).get("sample_id")
                if item_id:
                    done.add(item_id)
            except json.JSONDecodeError:
                continue
    return done


def clean_target(text: str, min_chars: int, max_chars: int) -> str | None:
    text = re.sub(r"\s+", " ", str(text)).strip()
    text = text.replace("```", "").strip()
    if not text or text.lower() == "nan":
        return None
    if len(text) < min_chars:
        return None
    return text[:max_chars]


def load_source_rows(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_csv(args.input)
    original_rows = len(df)
    df = df.dropna(subset=[args.query_col, args.comment_col]).copy()
    df[args.query_col] = df[args.query_col].astype(str).str.strip()
    df[args.comment_col] = df[args.comment_col].astype(str).str.strip()
    df = df[(df[args.query_col] != "") & (df[args.comment_col] != "")]
    df = df.drop_duplicates(subset=[args.query_col, args.id_col])
    if args.shuffle:
        df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)
    print(f"原始行数: {original_rows}, 清洗去重后: {len(df)}")
    return df


def write_splits(output: Path, examples: list[dict], args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    shuffled = examples[:]
    rng.shuffle(shuffled)

    valid_size = max(1, int(len(shuffled) * args.valid_ratio)) if shuffled else 0
    valid = shuffled[:valid_size]
    train = shuffled[valid_size:]

    train_path = output.with_name(output.stem + ".train.jsonl")
    valid_path = output.with_name(output.stem + ".valid.jsonl")
    for path, rows in [(train_path, train), (valid_path, valid)]:
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    lengths = [len(row["conversations"][1]["content"]) for row in examples]
    query_lengths = [len(row["conversations"][0]["content"]) for row in examples]
    comments = [row["metadata"].get("comment_id", "") for row in examples]
    room_types = [row["metadata"].get("fuzzy_room_type", "") for row in examples]
    stats = {
        "total": len(examples),
        "train": len(train),
        "valid": len(valid),
        "unique_comment_id": len(set(comments)),
        "avg_target_chars": round(sum(lengths) / len(lengths), 2) if lengths else 0,
        "min_target_chars": min(lengths) if lengths else 0,
        "max_target_chars": max(lengths) if lengths else 0,
        "avg_prompt_chars": round(sum(query_lengths) / len(query_lengths), 2) if query_lengths else 0,
        "fuzzy_room_type_counts": Counter(room_types),
        "train_path": str(train_path),
        "valid_path": str(valid_path),
    }
    stats_path = output.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"训练集: {train_path} ({len(train)} 条)")
    print(f"验证集: {valid_path} ({len(valid)} 条)")
    print(f"统计文件: {stats_path}")


def build_examples(args: argparse.Namespace) -> list[dict]:
    df = load_source_rows(args)
    output = Path(args.output)
    done_ids = load_done_ids(output) if args.resume else set()
    if done_ids:
        print(f"断点续跑: 已有 {len(done_ids)} 条样本，将跳过")

    examples = []
    if args.resume and output.exists():
        with output.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    examples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    use_api = args.use_api and bool(os.getenv("DASHSCOPE_API_KEY"))
    mode = "api" if use_api else "fallback"
    print(f"抽取模式: {mode}")

    output.parent.mkdir(parents=True, exist_ok=True)

    def make_example(i: int, row: pd.Series) -> dict | None:
        query = str(row[args.query_col]).strip()
        comment = str(row[args.comment_col]).strip()
        if not query or not comment or query == "nan" or comment == "nan":
            return None
        comment_id = str(row.get(args.id_col, ""))
        item_id = stable_id(query, comment_id, comment)
        if item_id in done_ids:
            return None

        if use_api:
            try:
                target = dashscope_extract(query, comment, args.model, args.max_chars)
                time.sleep(args.sleep)
            except Exception as exc:
                print(f"[WARN] API 抽取失败，第 {i} 行改用 fallback: {exc}")
                target = keyword_fallback_extract(query, comment, args.max_chars)
        else:
            target = keyword_fallback_extract(query, comment, args.max_chars)
        target = clean_target(target, args.min_chars, args.max_chars)
        if not target:
            return None

        return {
            "conversations": [
                {"role": "user", "content": f"{SYSTEM_PREFIX}\n用户查询：{query}"},
                {"role": "assistant", "content": target},
            ],
            "metadata": {
                "sample_id": item_id,
                "source_row": int(i),
                "comment_id": comment_id,
                "room_type": str(row.get("room_type", "")),
                "fuzzy_room_type": str(row.get("fuzzy_room_type", "")),
            },
        }

    pending_rows = [(int(i), row) for i, row in df.iterrows()]
    out_f = output.open("a" if args.resume else "w", encoding="utf-8")
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(make_example, i, row) for i, row in pending_rows]
        for future in as_completed(futures):
            example = future.result()
            if not example:
                continue
            examples.append(example)
            out_f.write(json.dumps(example, ensure_ascii=False) + "\n")
            if len(examples) % args.flush_every == 0:
                out_f.flush()
                print(f"已生成 {len(examples)} 条")

    out_f.close()

    return examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="RAG/data/processed/reverse_queries.csv")
    parser.add_argument("--output", default="RAG/data/evaluation/hyde_sft_train.jsonl")
    parser.add_argument("--query-col", default="query")
    parser.add_argument("--comment-col", default="comment")
    parser.add_argument("--id-col", default="comment_id")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-chars", type=int, default=160)
    parser.add_argument("--min-chars", type=int, default=20)
    parser.add_argument("--use-api", action="store_true")
    parser.add_argument("--model", default="qwen-plus")
    parser.add_argument("--sleep", type=float, default=0.05)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--flush-every", type=int, default=50)
    args = parser.parse_args()

    examples = build_examples(args)
    output = Path(args.output)
    print(f"已保存 {len(examples)} 条 HyDE SFT 样本到 {output}")
    write_splits(output, examples, args)
    if examples:
        print(json.dumps(examples[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
