#!/usr/bin/env bash
# P1 消融实验：GRPO group size (rollout.n) = 16
# 对照组：n=8 已有主实验结果 docs/实验结果.md 实验③，n=4 见 _ablation_grpo_n4.sh
set -o pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

set +u
source /home/sankuai/conda/etc/profile.d/conda.sh
conda activate rl
set -u

mkdir -p logs/ablation

env CUDA_VISIBLE_DEVICES=2,3 \
    MODEL_PATH="${PWD}/../models/sft_coldstart/qwen3-0.6b/global_step_42" \
    GSM8K_TRAIN_FILE="${PWD}/data/processed/gsm8k/train.parquet" \
    GSM8K_TEST_FILE="${PWD}/data/processed/gsm8k/test.parquet" \
    TRAIN_BATCH_SIZE=256 \
    PPO_MINI_BATCH_SIZE=64 \
    ROLLOUT_N=16 \
    TOTAL_EPOCHS=4 \
    SAVE_FREQ=10 \
    TEST_FREQ=5 \
    EXPERIMENT_NAME=grpo_qwen3_0.6b_n16 \
    RAY_DISABLE_DASHBOARD=1 \
    bash training/run_grpo.sh 2 \
    trainer.logger='["console"]' \
    "actor_rollout_ref.actor.checkpoint.contents=[model,optimizer,extra,hf_model]"
