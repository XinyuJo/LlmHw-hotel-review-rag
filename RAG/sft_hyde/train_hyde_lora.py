"""使用 Unsloth 对 Qwen3-4B-Instruct 做 HyDE LoRA SFT。

默认参数以“作业跑通”为目标，正式实验可增大 max_steps/数据量。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def load_jsonl(path: str) -> list[dict]:
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                examples.append({"conversations": item["conversations"]})
    return examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/root/sxy/Qwen3-4B-Instruct")
    parser.add_argument("--data", default="RAG/data/evaluation/hyde_sft_train.train.jsonl")
    parser.add_argument("--eval-data", default="")
    parser.add_argument("--output", default="hyde_sft_lora")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"

    from datasets import Dataset
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template, train_on_responses_only
    import torch
    from trl import SFTConfig, SFTTrainer

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        load_in_8bit=False,
        full_finetuning=False,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    tokenizer = get_chat_template(tokenizer, chat_template="qwen3-instruct")
    train_dataset = Dataset.from_list(load_jsonl(args.data))
    eval_dataset = Dataset.from_list(load_jsonl(args.eval_data)) if args.eval_data else None

    def formatting_prompts_func(examples):
        texts = [
            tokenizer.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=False
            )
            for convo in examples["conversations"]
        ]
        return {"text": texts}

    train_dataset = train_dataset.map(formatting_prompts_func, batched=True)
    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(formatting_prompts_func, batched=True)
    print(f"训练样本数: {len(train_dataset)}")
    print(f"验证样本数: {len(eval_dataset) if eval_dataset is not None else 0}")
    print(train_dataset[0]["text"][:500])

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    logging_dir = output / "logs"
    run_name = args.run_name or f"hyde-lora-r{args.lora_r}"
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            logging_steps=1,
            eval_strategy="steps" if eval_dataset is not None else "no",
            eval_steps=args.eval_steps,
            save_strategy="steps",
            save_steps=args.save_steps,
            save_total_limit=2,
            optim="adamw_8bit",
            output_dir=str(output / "checkpoints"),
            logging_dir=str(logging_dir),
            report_to="none",
            run_name=run_name,
        ),
    )
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )
    train_result = trainer.train()
    metrics = dict(train_result.metrics)
    if eval_dataset is not None:
        metrics.update({f"final_{k}": v for k, v in trainer.evaluate().items()})
    metrics.update({
        "run_name": run_name,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "max_steps": args.max_steps,
        "train_examples": len(train_dataset),
        "eval_examples": len(eval_dataset) if eval_dataset is not None else 0,
    })
    (output / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    model.save_pretrained(str(output))
    tokenizer.save_pretrained(str(output))
    print(f"LoRA 权重已保存到: {output.resolve()}")

    FastLanguageModel.for_inference(model)
    sample_queries = [
        "这家酒店最近早餐怎么样？",
        "带三岁小孩住这里安全吗，房间有没有尖角或落地窗风险？",
        "这家酒店交通方便吗，周边吃饭选择多不多？",
    ]
    samples = []
    for query in sample_queries:
        messages = [{
            "role": "user",
            "content": f"请根据用户查询生成一段简短、真实、和查询高度相关的酒店评论假设文档。\n用户查询：{query}",
        }]
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs,
                max_new_tokens=160,
                temperature=0.7,
                do_sample=True,
            )
        text = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
        samples.append({"query": query, "hyde": text.strip()})
        print(f"\n[样例] {query}\n{text.strip()}")
    (output / "inference_samples.json").write_text(
        json.dumps(samples, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
