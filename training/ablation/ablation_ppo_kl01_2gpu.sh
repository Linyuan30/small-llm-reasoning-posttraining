#!/usr/bin/env bash
# P1 消融实验（受控对照）：PPO KL_LOSS_COEF = 0.01，但改用 2 卡（与 kl0.005/kl0.02 消融组卡数对齐）
# 目的：排除"2卡 vs 4卡"这一未受控变量，验证 v3 主实验（4卡）中 step21-28 length 冲高
#       到底是 KL 系数本身导致，还是训练随机性 + 卡数差异共同作用的结果。
# 对照组：
#   - KL=0.01, 4卡（v3主实验，未受控） -> docs/实验结果.md 实验④ v3
#   - KL=0.005, 2卡 -> _ablation_ppo_kl005.sh
#   - KL=0.02,  2卡 -> _ablation_ppo_kl02.sh
#   - KL=0.01, 2卡（本脚本，受控对照）
set -o pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

set +u
source /home/sankuai/conda/etc/profile.d/conda.sh
conda activate rl
set -u

mkdir -p logs/ablation

env CUDA_VISIBLE_DEVICES=0,1 \
    MODEL_PATH="${PWD}/../models/sft_coldstart/qwen3-0.6b/global_step_42" \
    GSM8K_TRAIN_FILE="${PWD}/data/processed/gsm8k/train.parquet" \
    GSM8K_TEST_FILE="${PWD}/data/processed/gsm8k/test.parquet" \
    TRAIN_BATCH_SIZE=256 \
    PPO_MINI_BATCH_SIZE=64 \
    KL_LOSS_COEF=0.01 \
    TOTAL_EPOCHS=4 \
    SAVE_FREQ=10 \
    TEST_FREQ=2 \
    EXPERIMENT_NAME=ppo_qwen3_0.6b_kl0.01_2gpu \
    RAY_DISABLE_DASHBOARD=1 \
    bash training/run_ppo.sh 2 \
    trainer.logger='["console"]' \
    "actor_rollout_ref.actor.checkpoint.contents=[model,optimizer,extra,hf_model]"
