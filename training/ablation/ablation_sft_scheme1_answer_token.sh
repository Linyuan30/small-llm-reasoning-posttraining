#!/usr/bin/env bash
set -euo pipefail
export PATH="/home/sankuai/conda/envs/rl/bin:${PATH}"
cd /home/hadoop-risk-control-algo/dolphinfs_ssd_hadoop-risk-control-algo/oyyx/train/training

export CUDA_VISIBLE_DEVICES=0,2,3,4
export MODEL_PATH="/home/hadoop-risk-control-algo/dolphinfs_ssd_hadoop-risk-control-algo/oyyx/train/models/base_with_answer_token/qwen3-0.6b"
export TRAIN_FILE="/home/hadoop-risk-control-algo/dolphinfs_ssd_hadoop-risk-control-algo/oyyx/train/data/processed/sft_coldstart_mix_train.parquet"
export VAL_FILE="/home/hadoop-risk-control-algo/dolphinfs_ssd_hadoop-risk-control-algo/oyyx/train/data/processed/sft_coldstart_mix_val.parquet"
export TOTAL_EPOCHS=7
export PROJECT_NAME=llm-rl-reasoning
export EXPERIMENT_NAME=sft_answer_special_token_qwen3_0.6b_fixed
export LOGGER='["console"]'

bash run_sft.sh 4 \
    /home/hadoop-risk-control-algo/dolphinfs_ssd_hadoop-risk-control-algo/oyyx/train/models/sft_coldstart/qwen3-0.6b-answer-token
