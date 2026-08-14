# Copyright (c) 2026
#
# Add <answer>/</answer> as special tokens to the base model tokenizer, resize
# the model embeddings, and save a new copy of the base model weights.
#
# Background: Qwen3 natively treats <think>/</think> as single special tokens,
# but <answer>/</answer> are not special — the tokenizer splits them into ~3
# sub-word tokens each ('<', 'answer', '>'). This makes the <answer>...</answer>
# format boundary harder to learn during SFT compared to <think>...</think>, and
# is a likely contributor to the low strict_format_rate on the Base route.
#
# New token embeddings are initialised as the mean of the original sub-word
# embeddings (rather than random), giving the new tokens a sensible starting
# point and faster convergence.
#
# Usage:
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
    """Initialise a new special-token embedding as the mean of the original sub-word embeddings.

    Applied to both the input embedding (embed_tokens) and the output embedding
    (lm_head), if the weights are not tied.
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
    parser.add_argument("--base_model_path", required=True, help="path to the original base model")
    parser.add_argument("--output_path", required=True, help="path to save the modified model")
    args = parser.parse_args()

    print(f"Loading tokenizer & model from {args.base_model_path} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path, trust_remote_code=True, torch_dtype=torch.float32
    )

    # Record how each new token is tokenized *before* adding it as a special token,
    # so we can use those sub-word ids for mean-pool initialisation.
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

    # Verify encoding after adding the special tokens
    for tok in NEW_SPECIAL_TOKENS:
        ids = tokenizer.encode(tok, add_special_tokens=False)
        print(f"After:  {tok!r} -> {ids} -> {tokenizer.convert_ids_to_tokens(ids)}")

    os.makedirs(args.output_path, exist_ok=True)
    model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)
    print(f"Saved new base model (with <answer>/</answer> special tokens) to {args.output_path}")


if __name__ == "__main__":
    main()
