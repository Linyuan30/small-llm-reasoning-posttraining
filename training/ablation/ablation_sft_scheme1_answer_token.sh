#!/usr/bin/env bash
# Ablation [scheme 1]: SFT re-train after adding <answer>/</answer> as special tokens
# to the tokenizer.
# Pre-requisite: run scripts/add_answer_special_tokens.py first to produce the
# base_with_answer_token model pointed to by MODEL_PATH.
set -euo pipefail

# This script lives in training/ablation/; the repo root is two levels up
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

CONDA_ENV_BIN=${CONDA_ENV_BIN:-/home/sankuai/conda/envs/rl/bin}
export PATH="${CONDA_ENV_BIN}:${PATH}"
cd "${REPO_ROOT}/training"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,2,3,4}
export MODEL_PATH=${MODEL_PATH:-"${REPO_ROOT}/../models/base_with_answer_token/qwen3-0.6b"}
export TRAIN_FILE=${TRAIN_FILE:-"${REPO_ROOT}/data/processed/sft_coldstart_mix_train.parquet"}
export VAL_FILE=${VAL_FILE:-"${REPO_ROOT}/data/processed/sft_coldstart_mix_val.parquet"}
export TOTAL_EPOCHS=7
export PROJECT_NAME=llm-rl-reasoning
export EXPERIMENT_NAME=sft_answer_special_token_qwen3_0.6b_fixed
export LOGGER='["console"]'

SAVE_PATH=${SAVE_PATH:-"${REPO_ROOT}/../models/sft_coldstart/qwen3-0.6b-answer-token"}

bash run_sft.sh 4 "${SAVE_PATH}"
