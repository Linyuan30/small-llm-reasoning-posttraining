# Copyright (c) 2026
#
# Download and format GSM8K into:
#   1. data/processed/gsm8k/{train,test}.parquet  — RL training data for veRL GRPO/PPO
#   2. data/processed/gsm8k_sft_coldstart.jsonl   — SFT cold-start data (<think>/<answer> format)
#
# Usage:
#   python prepare_gsm8k.py --local_save_dir ../processed/gsm8k \
#       --sft_output ../processed/gsm8k_sft_coldstart.jsonl \
#       --sft_sample_size 2000

from __future__ import annotations

import argparse
import json
import os
import sys

import datasets

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from format_cot import (  # noqa: E402
    SYSTEM_PROMPT,
    build_prompt_messages,
    clean_gsm8k_rationale,
    format_think_answer,
)

DATA_SOURCE = "openai/gsm8k"


def build_rl_dataset(dataset: datasets.Dataset, split: str) -> datasets.Dataset:
    """Build RL (GRPO/PPO) training data in the veRL parquet schema."""

    def process_fn(example, idx):
        question_raw = example.pop("question")
        answer_raw = example.pop("answer")
        _, final_answer = clean_gsm8k_rationale(answer_raw)

        return {
            "data_source": "gsm8k",
            "prompt": build_prompt_messages(question_raw),
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": final_answer},
            "extra_info": {
                "split": split,
                "index": idx,
                "question": question_raw,
                "answer_raw": answer_raw,
            },
        }

    return dataset.map(function=process_fn, with_indices=True, remove_columns=dataset.column_names)


def build_sft_records(dataset: datasets.Dataset, sample_size: int, seed: int = 42) -> list[dict]:
    """Sample from the GSM8K train split and reformat into <think>/<answer> for SFT."""
    n = min(sample_size, len(dataset))
    shuffled = dataset.shuffle(seed=seed).select(range(n))

    records = []
    for idx, example in enumerate(shuffled):
        question = example["question"]
        rationale, final_answer = clean_gsm8k_rationale(example["answer"])
        assistant_reply = format_think_answer(rationale, final_answer)

        records.append(
            {
                "data_source": "gsm8k",
                "index": idx,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"{question}\n\n"
                        "Please reason step by step, and put your final answer within "
                        "<answer></answer>. Respond in the following format strictly:\n"
                        "<think>\n(your reasoning process)\n</think>\n<answer>\n(final answer only)\n</answer>",
                    },
                    {"role": "assistant", "content": assistant_reply},
                ],
            }
        )
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dataset_path", default=None, help="local HF cache path (skip download if provided)")
    parser.add_argument("--local_save_dir", default="../processed/gsm8k", help="output directory for RL parquet files")
    parser.add_argument("--sft_output", default="../processed/gsm8k_sft_coldstart.jsonl", help="output path for SFT cold-start jsonl")
    parser.add_argument("--sft_sample_size", type=int, default=2000, help="number of SFT records to sample")
    args = parser.parse_args()

    print(f"Loading {DATA_SOURCE} dataset ...", flush=True)
    if args.local_dataset_path is not None:
        dataset = datasets.load_dataset(args.local_dataset_path, "main")
    else:
        dataset = datasets.load_dataset(DATA_SOURCE, "main")

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]
    print(f"train size = {len(train_dataset)}, test size = {len(test_dataset)}")

    rl_train = build_rl_dataset(train_dataset, "train")
    rl_test = build_rl_dataset(test_dataset, "test")

    save_dir = os.path.abspath(args.local_save_dir)
    os.makedirs(save_dir, exist_ok=True)
    rl_train.to_parquet(os.path.join(save_dir, "train.parquet"))
    rl_test.to_parquet(os.path.join(save_dir, "test.parquet"))
    print(f"Saved RL parquet to {save_dir}")

    sft_records = build_sft_records(train_dataset, args.sft_sample_size)
    sft_output_path = os.path.abspath(args.sft_output)
    os.makedirs(os.path.dirname(sft_output_path), exist_ok=True)
    with open(sft_output_path, "w", encoding="utf-8") as f:
        for record in sft_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Saved {len(sft_records)} SFT coldstart records to {sft_output_path}")

    # Save one example for quick format sanity-check
    example_path = os.path.join(save_dir, "train_example.json")
    with open(example_path, "w", encoding="utf-8") as f:
        json.dump(rl_train[0], f, ensure_ascii=False, indent=2)
    print(f"Saved one RL example to {example_path}")

    sft_example_path = os.path.join(os.path.dirname(sft_output_path), "gsm8k_sft_example.json")
    with open(sft_example_path, "w", encoding="utf-8") as f:
        json.dump(sft_records[0], f, ensure_ascii=False, indent=2)
    print(f"Saved one SFT example to {sft_example_path}")


if __name__ == "__main__":
    main()
