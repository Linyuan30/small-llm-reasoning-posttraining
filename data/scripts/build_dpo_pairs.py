# Copyright (c) 2026
#
# Rejection-sampling from the SFT cold-start model to build DPO preference pairs.
#
# Pipeline:
#   1. Load the SFT model with vLLM; sample N responses per training prompt.
#   2. Score each response with the rule reward (reward/rule_reward.py).
#   3. Take the highest-scoring response as "chosen" and the lowest as "rejected".
#      If best_score == worst_score (all samples tied), skip the prompt — no signal.
#
# Usage:
#   python build_dpo_pairs.py \
#       --model_path ../../models/sft_coldstart/qwen3-0.6b \
#       --input_parquet ../processed/gsm8k/train.parquet \
#       --output_path ../processed/gsm8k_dpo_pairs.jsonl \
#       --num_samples 8 \
#       --max_prompts 4000

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "reward"))
from rule_reward import RuleReward  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--input_parquet", required=True, help="RL training parquet produced by prepare_gsm8k.py / prepare_math.py")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--num_samples", type=int, default=8, help="number of samples per prompt")
    parser.add_argument("--max_prompts", type=int, default=4000, help="max number of prompts to process")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    from vllm import LLM, SamplingParams  # lazy import to avoid hard vllm dep when not needed

    df = pd.read_parquet(args.input_parquet)
    if args.max_prompts is not None:
        df = df.sample(n=min(args.max_prompts, len(df)), random_state=args.seed).reset_index(drop=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    chat_prompts = []
    ground_truths = []
    for _, row in df.iterrows():
        messages = list(row["prompt"])
        chat_prompts.append(
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        )
        ground_truths.append(row["reward_model"]["ground_truth"])

    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype="bfloat16",
    )
    sampling_params = SamplingParams(
        n=args.num_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
    )

    print(f"Running rejection sampling on {len(chat_prompts)} prompts x {args.num_samples} samples ...")
    outputs = llm.generate(chat_prompts, sampling_params)

    reward_fn = RuleReward()
    pairs = []
    skipped_no_signal = 0

    for prompt_text, output, gt in zip(chat_prompts, outputs, ground_truths):
        candidates = [o.text for o in output.outputs]
        scored = [(reward_fn.score(text, gt).score, text) for text in candidates]
        scored.sort(key=lambda x: x[0])

        worst_score, worst_text = scored[0]
        best_score, best_text = scored[-1]

        if best_score <= worst_score:
            skipped_no_signal += 1
            continue

        pairs.append(
            {
                "prompt": prompt_text,
                "chosen": best_text,
                "rejected": worst_text,
                "chosen_score": best_score,
                "rejected_score": worst_score,
            }
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    print(f"Built {len(pairs)} DPO pairs, skipped {skipped_no_signal} prompts with no preference signal.")
    print(f"Saved to {args.output_path}")


if __name__ == "__main__":
    main()
