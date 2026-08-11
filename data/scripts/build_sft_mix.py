# Copyright (c) 2026
#
# 把 gsm8k_sft_coldstart.jsonl 与 math_sft_coldstart.jsonl（各约2000条）
# 混合、切分成 train/val parquet，供 run_sft.sh 使用。
#
# 每个来源各抽 val_size_per_source 条作为验证集，剩余合并作为训练集，
# 与已有的 sft_coldstart_mix_train/val.parquet 构造逻辑保持一致
# （各2000条 -> 各1900训练+100验证 -> 合并3800训练+200验证）。
#
# 用法（默认，主线格式）：
#   python build_sft_mix.py \
#       --inputs ../processed/gsm8k_sft_coldstart.jsonl ../processed/math_sft_coldstart.jsonl \
#       --train_output ../processed/sft_coldstart_mix_train.parquet \
#       --val_output ../processed/sft_coldstart_mix_val.parquet
#
# 用法（v2 变体，无 answer 标签格式）：
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
    parser.add_argument("--inputs", nargs="+", required=True, help="一个或多个 SFT jsonl 文件路径")
    parser.add_argument("--train_output", required=True)
    parser.add_argument("--val_output", required=True)
    parser.add_argument("--val_size_per_source", type=int, default=100, help="每个来源抽取多少条作为验证集")
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
