#!/usr/bin/env bash
# 消融实验【方案2】：plain answer格式（去掉 <answer> XML 标签，直接用纯文本答案）的 SFT 重训练
# 前置条件：需先用 data/scripts/prepare_*_v2_plain_answer.py 生成 TRAIN_FILE/VAL_FILE 对应的数据
set -euo pipefail

# 本脚本位于 training/ablation/，仓库根目录在其上两级
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

CONDA_ENV_BIN=${CONDA_ENV_BIN:-/home/sankuai/conda/envs/rl/bin}
export PATH="${CONDA_ENV_BIN}:${PATH}"
cd "${REPO_ROOT}/training"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-5,6,7,1}
export MODEL_PATH=${MODEL_PATH:-"${REPO_ROOT}/../model/Qwen3-0.6B-Base"}
export TRAIN_FILE=${TRAIN_FILE:-"${REPO_ROOT}/data/processed/sft_coldstart_mix_train_v2_plain_answer.parquet"}
export VAL_FILE=${VAL_FILE:-"${REPO_ROOT}/data/processed/sft_coldstart_mix_val_v2_plain_answer.parquet"}
export TOTAL_EPOCHS=7
export PROJECT_NAME=llm-rl-reasoning
export EXPERIMENT_NAME=sft_v2_plain_answer_qwen3_0.6b_fixed
export LOGGER='["console"]'

SAVE_PATH=${SAVE_PATH:-"${REPO_ROOT}/../models/sft_coldstart/qwen3-0.6b-v2-plain-answer"}

bash run_sft.sh 4 "${SAVE_PATH}"
