#!/usr/bin/env bash
# Ablation: PPO KL_LOSS_COEF = 0.005 (all other hyperparameters match the v3 main run)
# Baseline: KL=0.01 main run in docs/experiments.md; KL=0.02 in ablation_ppo_kl02.sh
set -o pipefail
# This script lives in training/ablation/; the repo root is two levels up
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

CONDA_ENV=${CONDA_ENV:-rl}
CONDA_SH=${CONDA_SH:-/home/sankuai/conda/etc/profile.d/conda.sh}

set +u
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
set -u

mkdir -p logs/ablation

env CUDA_VISIBLE_DEVICES=4,5 \
    MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/../models/sft_coldstart/qwen3-0.6b/global_step_42}" \
    GSM8K_TRAIN_FILE="${GSM8K_TRAIN_FILE:-${REPO_ROOT}/data/processed/gsm8k/train.parquet}" \
    GSM8K_TEST_FILE="${GSM8K_TEST_FILE:-${REPO_ROOT}/data/processed/gsm8k/test.parquet}" \
    TRAIN_BATCH_SIZE=256 \
    PPO_MINI_BATCH_SIZE=64 \
    KL_LOSS_COEF=0.005 \
    TOTAL_EPOCHS=4 \
    SAVE_FREQ=10 \
    TEST_FREQ=2 \
    EXPERIMENT_NAME=ppo_qwen3_0.6b_kl0.005 \
    RAY_DISABLE_DASHBOARD=1 \
    bash "${REPO_ROOT}/training/run_ppo.sh" 2 \
    trainer.logger='["console"]' \
    "actor_rollout_ref.actor.checkpoint.contents=[model,optimizer,extra,hf_model]"
