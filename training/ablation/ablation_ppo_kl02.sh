#!/usr/bin/env bash
# P1 消融实验：PPO KL_LOSS_COEF = 0.02（其余超参与已修复的 v3 主实验保持一致）
# 对照组：KL=0.01 已有主实验结果 docs/实验结果.md 实验④ v3，KL=0.005 见 _ablation_ppo_kl005.sh
set -o pipefail
# 本脚本位于 training/ablation/，仓库根目录在其上两级
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

CONDA_ENV=${CONDA_ENV:-rl}
CONDA_SH=${CONDA_SH:-/home/sankuai/conda/etc/profile.d/conda.sh}

set +u
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
set -u

mkdir -p logs/ablation

env CUDA_VISIBLE_DEVICES=6,7 \
    MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/../models/sft_coldstart/qwen3-0.6b/global_step_42}" \
    GSM8K_TRAIN_FILE="${GSM8K_TRAIN_FILE:-${REPO_ROOT}/data/processed/gsm8k/train.parquet}" \
    GSM8K_TEST_FILE="${GSM8K_TEST_FILE:-${REPO_ROOT}/data/processed/gsm8k/test.parquet}" \
    TRAIN_BATCH_SIZE=256 \
    PPO_MINI_BATCH_SIZE=64 \
    KL_LOSS_COEF=0.02 \
    TOTAL_EPOCHS=4 \
    SAVE_FREQ=10 \
    TEST_FREQ=2 \
    EXPERIMENT_NAME=ppo_qwen3_0.6b_kl0.02 \
    RAY_DISABLE_DASHBOARD=1 \
    bash "${REPO_ROOT}/training/run_ppo.sh" 2 \
    trainer.logger='["console"]' \
    "actor_rollout_ref.actor.checkpoint.contents=[model,optimizer,extra,hf_model]"
