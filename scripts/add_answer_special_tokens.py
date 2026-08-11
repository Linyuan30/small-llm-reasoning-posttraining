# Copyright (c) 2026
#
# 给 base 模型的 tokenizer 新增 <answer>/</answer> special token，并 resize
# model 的 embedding / lm_head，另存为一份新的 base 模型权重。
#
# 背景（详见 docs/问题记录.md）：
#   Qwen3 tokenizer 原生把 <think>/</think> 作为单独的 added special token
#   （分别编码为 1 个 token），但 <answer>/</answer> 并不是 special token，
#   会被 BPE 拆成 3 个普通子词（如 '<' 'answer' '>'）。这导致模型学习
#   "<answer>...</answer>" 这个格式边界比学 "<think>...</think>" 难得多，
#   是 SFT 格式合规率低的一个重要根因。
#
#   本脚本给 tokenizer 新增这两个 special token 并同步 resize 模型的
#   embedding（新 token 的向量取「原始子词组合」embedding 的均值做初始化，
#   而不是随机初始化，这样新 token 一开始就带有一定语义，收敛更快）。
#
# 用法：
#   python add_answer_special_tokens.py \
#       --base_model_path ../../model/Qwen3-0.6B-Base \
#       --output_path ../models/base_with_answer_token/qwen3-0.6b
#
#   python add_answer_special_tokens.py \
#       --base_model_path ../../model/Qwen3-1.7B-Base \
#       --output_path ../models/base_with_answer_token/qwen3-1.7b

from __future__ import annotations

import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

NEW_SPECIAL_TOKENS = ["<answer>", "</answer>"]


def mean_pool_init_embedding(model, tokenizer, new_token: str, old_ids: list[int]) -> None:
    """用旧 token 序列 embedding 的均值初始化新 special token 的 embedding。

    对 输入 embedding（embed_tokens）和 输出 embedding（lm_head，若未 tie）
    都做同样处理。
    """
    new_id = tokenizer.convert_tokens_to_ids(new_token)

    input_embeddings = model.get_input_embeddings().weight.data
    old_vecs = input_embeddings[old_ids]
    input_embeddings[new_id] = old_vecs.mean(dim=0)

    output_embeddings = model.get_output_embeddings()
    if output_embeddings is not None and not model.config.tie_word_embeddings:
        output_weight = output_embeddings.weight.data
        old_out_vecs = output_weight[old_ids]
        output_weight[new_id] = old_out_vecs.mean(dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", required=True, help="原始 base 模型路径")
    parser.add_argument("--output_path", required=True, help="新模型（含新增 special token）保存路径")
    args = parser.parse_args()

    print(f"Loading tokenizer & model from {args.base_model_path} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path, trust_remote_code=True, torch_dtype=torch.float32
    )

    # 记录新 token 在“旧 tokenizer”下会被拆成哪些 token id，用于均值初始化
    old_ids_map = {}
    for tok in NEW_SPECIAL_TOKENS:
        ids = tokenizer.encode(tok, add_special_tokens=False)
        old_ids_map[tok] = ids
        print(f"Before: {tok!r} -> {ids} -> {tokenizer.convert_ids_to_tokens(ids)}")

    num_added = tokenizer.add_special_tokens({"additional_special_tokens": NEW_SPECIAL_TOKENS})
    print(f"Added {num_added} new special tokens: {NEW_SPECIAL_TOKENS}")

    old_vocab_size = model.get_input_embeddings().weight.shape[0]
    model.resize_token_embeddings(len(tokenizer))
    new_vocab_size = model.get_input_embeddings().weight.shape[0]
    print(f"Resized embedding: {old_vocab_size} -> {new_vocab_size}")

    with torch.no_grad():
        for tok in NEW_SPECIAL_TOKENS:
            mean_pool_init_embedding(model, tokenizer, tok, old_ids_map[tok])

    # 验证编码结果
    for tok in NEW_SPECIAL_TOKENS:
        ids = tokenizer.encode(tok, add_special_tokens=False)
        print(f"After:  {tok!r} -> {ids} -> {tokenizer.convert_ids_to_tokens(ids)}")

    os.makedirs(args.output_path, exist_ok=True)
    model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)
    print(f"Saved new base model (with <answer>/</answer> special tokens) to {args.output_path}")


if __name__ == "__main__":
    main()
