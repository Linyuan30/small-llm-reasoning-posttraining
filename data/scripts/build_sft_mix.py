# Copyright (c) 2026
#
# Merge gsm8k_sft_coldstart.jsonl and math_sft_coldstart.jsonl (~2000 records each),
# split into train/val parquets for run_sft.sh.
#
# Each source contributes val_size_per_source records to the val set; the rest form
# the train set (default: 1900 train + 100 val per source → 3800 train + 200 val total).
#
# Usage (main-line format):
#   python build_sft_mix.py \
#       --inputs ../processed/gsm8k_sft_coldstart.jsonl ../processed/math_sft_coldstart.jsonl \
#       --train_output ../processed/sft_coldstart_mix_train.parquet \
#       --val_output ../processed/sft_coldstart_mix_val.parquet
#
# Usage (v2 plain-answer variant):
#   python build_sft_mix.py \
#       --inputs ../processed/gsm8k_sft_coldstart_v2_plain_answer.jsonl \
#                ../processed/math_sft_coldstart_v2_plain_answer.jsonl \
#       --train_output ../processed/sft_coldstart_mix_train_v2_plain_answer.parquet \
#       --val_output ../processed/sft_coldstart_mix_val_v2_plain_answer.parquet

from __future__ import annotations

import argparse
import os

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, help="one or more SFT jsonl file paths")
    parser.add_argument("--train_output", required=True)
    parser.add_argument("--val_output", required=True)
    parser.add_argument("--val_size_per_source", type=int, default=100, help="val records per source")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_parts, val_parts = [], []
    for path in args.inputs:
        df = pd.read_json(path, lines=True)
        df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
        val_df = df.iloc[: args.val_size_per_source]
        train_df = df.iloc[args.val_size_per_source :]
        print(f"{path}: total={len(df)}, train={len(train_df)}, val={len(val_df)}")
        train_parts.append(train_df)
        val_parts.append(val_df)

    train_all = pd.concat(train_parts, ignore_index=True).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    val_all = pd.concat(val_parts, ignore_index=True).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.train_output)), exist_ok=True)
    train_all.to_parquet(args.train_output)
    val_all.to_parquet(args.val_output)
    print(f"Saved train ({len(train_all)} rows) to {args.train_output}")
    print(f"Saved val ({len(val_all)} rows) to {args.val_output}")


if __name__ == "__main__":
    main()
