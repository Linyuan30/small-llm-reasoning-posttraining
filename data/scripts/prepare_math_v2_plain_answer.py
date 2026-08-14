# Copyright (c) 2026
#
# Ablation variant (v2): MATH formatted with plain-answer output — no <answer> XML tags;
# uses <think>...</think> followed by the answer as plain text.
# Output is written to separate files and does not overwrite the main-line data:
#   1. data/processed/math_v2_plain_answer/{train,test}.parquet
#   2. data/processed/math_sft_coldstart_v2_plain_answer.jsonl
#
# Usage:
#   python prepare_math_v2_plain_answer.py \
#       --local_dataset_path ../processed/math_hf_cache \
#       --local_save_dir ../processed/math_v2_plain_answer \
#       --sft_output ../processed/math_sft_coldstart_v2_plain_answer.jsonl \
#       --sft_sample_size 2000

from __future__ import annotations

import argparse
import json
import os
import sys

import datasets

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from format_cot_v2_plain_answer import (  # noqa: E402
    SYSTEM_PROMPT,
    build_prompt_messages,
    extract_boxed_answer,
    format_think_answer,
)

DATA_SOURCE = "DigitalLearningGmbH/MATH-lighteval"


def build_rl_dataset(dataset: datasets.Dataset, split: str) -> datasets.Dataset:
    def process_fn(example, idx):
        problem = example.pop("problem")
        solution = example.pop("solution")
        level = example.get("level", None)
        subject = example.get("type", None)
        final_answer = extract_boxed_answer(solution)

        return {
            "data_source": "math",
            "prompt": build_prompt_messages(problem),
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": final_answer or ""},
            "extra_info": {
                "split": split,
                "index": idx,
                "question": problem,
                "solution_raw": solution,
                "level": level,
                "subject": subject,
            },
        }

    return dataset.map(function=process_fn, with_indices=True, remove_columns=dataset.column_names)


def build_sft_records(dataset: datasets.Dataset, sample_size: int, seed: int = 42) -> list[dict]:
    shuffled = dataset.shuffle(seed=seed)

    records = []
    for example in shuffled:
        if len(records) >= sample_size:
            break

        problem = example["problem"]
        solution = example["solution"]
        final_answer = extract_boxed_answer(solution)
        if not final_answer:
            continue

        assistant_reply = format_think_answer(solution, final_answer)
        records.append(
            {
                "data_source": "math",
                "index": len(records),
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"{problem}\n\n"
                        "Please reason step by step. Respond in the following format strictly:\n"
                        "<think>\n(your reasoning process)\n</think>\n(final answer only, no tags)",
                    },
                    {"role": "assistant", "content": assistant_reply},
                ],
            }
        )
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dataset_path", default=None, help="local HF cache path (skip download if provided)")
    parser.add_argument("--local_save_dir", default="../processed/math_v2_plain_answer", help="output directory for RL parquet files")
    parser.add_argument(
        "--sft_output", default="../processed/math_sft_coldstart_v2_plain_answer.jsonl", help="output path for SFT cold-start jsonl"
    )
    parser.add_argument("--sft_sample_size", type=int, default=2000, help="number of SFT records to sample")
    args = parser.parse_args()

    print(f"Loading {DATA_SOURCE} dataset ...", flush=True)
    if args.local_dataset_path is not None:
        dataset = datasets.load_dataset(args.local_dataset_path)
    else:
        dataset = datasets.load_dataset(DATA_SOURCE)

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

    example_path = os.path.join(save_dir, "train_example.json")
    with open(example_path, "w", encoding="utf-8") as f:
        json.dump(rl_train[0], f, ensure_ascii=False, indent=2)
    print(f"Saved one RL example to {example_path}")

    sft_example_path = os.path.join(os.path.dirname(sft_output_path), "math_sft_example_v2_plain_answer.json")
    with open(sft_example_path, "w", encoding="utf-8") as f:
        json.dump(sft_records[0], f, ensure_ascii=False, indent=2)
    print(f"Saved one SFT example to {sft_example_path}")


if __name__ == "__main__":
    main()
