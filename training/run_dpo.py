# Copyright (c) 2026
#
# DPO training script (off-policy baseline).
#
# Uses the DPOTrainer from the `trl` library (veRL focuses on on-policy RL
# and does not include a standard DPO trainer).
# Preference pairs are built by rejection-sampling from the SFT model
# (see data/scripts/build_dpo_pairs.py); each line is a JSON object:
#   {"prompt": ..., "chosen": ..., "rejected": ...}
#
# Usage (single-node multi-GPU, accelerate or torchrun both work):
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
    parser.add_argument("--model_path", required=True, help="SFT cold-start checkpoint (init weights for policy and reference)")
    parser.add_argument("--data_path", required=True, help="path to DPO preference-pair jsonl")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--eval_data_path", default=None, help="optional validation preference-pair data")
    parser.add_argument("--learning_rate", type=float, default=5e-7)
    parser.add_argument("--beta", type=float, default=0.1, help="DPO temperature coefficient")
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
    # DPO requires a frozen reference model. trl supports passing None to let
    # it deepcopy internally, but explicitly loading a separate read-only copy
    # is more memory-efficient for large models.
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
