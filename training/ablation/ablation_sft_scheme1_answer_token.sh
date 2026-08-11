#!/usr/bin/env bash
# 消融实验【方案1】：给 tokenizer 新增 <answer>/</answer> special token 后的 SFT 重训练
# 前置条件：需先用 scripts/add_answer_special_tokens.py 生成 MODEL_PATH 指向的 base_with_answer_token 模型
set -euo pipefail

# 本脚本位于 training/ablation/，仓库根目录在其上两级
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
