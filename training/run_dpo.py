# Copyright (c) 2026
#
# DPO 训练脚本（作为 off-policy 基线，对应 docs/流程.md 5.5节 / docs/方案计划.md Week5）
#
# 使用 `trl` 库的 DPOTrainer 实现（veRL 本身专注 on-policy RL，不内置标准 DPO trainer）。
# 偏好对数据来自 SFT 模型的 rejection sampling（见 data/scripts/build_dpo_pairs.py），
# 数据格式为 jsonl，每行包含 {"prompt": ..., "chosen": ..., "rejected": ...}
#
# 用法（单机多卡，accelerate 或 torchrun 均可）：
#   accelerate launch --num_processes 4 run_dpo.py \
#       --model_path ../models/sft_coldstart/qwen3-0.6b \
#       --data_path ../data/processed/gsm8k_dpo_pairs.jsonl \
#       --output_dir ../models/dpo_ckpt/qwen3-0.6b

from __future__ import annotations

import argparse
import os

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="SFT 冷启动模型路径（policy 与 reference 的初始化权重）")
    parser.add_argument("--data_path", required=True, help="DPO 偏好对数据 jsonl 路径")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--eval_data_path", default=None, help="可选的验证集偏好对数据")
    parser.add_argument("--learning_rate", type=float, default=5e-7)
    parser.add_argument("--beta", type=float, default=0.1, help="DPO 温度系数")
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=1536)
    parser.add_argument("--max_prompt_length", type=int, default=512)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--report_to", default="wandb")
    parser.add_argument("--project_name", default="llm-rl-reasoning")
    parser.add_argument("--run_name", default="dpo_gsm8k")
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("WANDB_PROJECT", args.project_name)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
    )
    # DPO 需要一个固定的 reference 模型；trl 支持传 None 让内部自动 deepcopy，
    # 但对大模型显存更友好的做法是显式加载一份只读的 ref 模型。
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
    )

    data_files = {"train": args.data_path}
    if args.eval_data_path:
        data_files["validation"] = args.eval_data_path
    dataset = load_dataset("json", data_files=data_files)

    train_dataset = dataset["train"]
    eval_dataset = dataset.get("validation")

    training_args = DPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        beta=args.beta,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        bf16=args.bf16,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_dataset is not None else "no",
        report_to=[args.report_to] if args.report_to else [],
        run_name=args.run_name,
        remove_unused_columns=False,
        gradient_checkpointing=True,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
