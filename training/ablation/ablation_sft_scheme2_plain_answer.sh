#!/usr/bin/env bash
set -euo pipefail
export PATH="/home/sankuai/conda/envs/rl/bin:${PATH}"
cd /home/hadoop-risk-control-algo/dolphinfs_ssd_hadoop-risk-control-algo/oyyx/train/training

export CUDA_VISIBLE_DEVICES=5,6,7,1
export MODEL_PATH="/home/hadoop-risk-control-algo/dolphinfs_ssd_hadoop-risk-control-algo/oyyx/model/Qwen3-0.6B-Base"
export TRAIN_FILE="/home/hadoop-risk-control-algo/dolphinfs_ssd_hadoop-risk-control-algo/oyyx/train/data/processed/sft_coldstart_mix_train_v2_plain_answer.parquet"
export VAL_FILE="/home/hadoop-risk-control-algo/dolphinfs_ssd_hadoop-risk-control-algo/oyyx/train/data/processed/sft_coldstart_mix_val_v2_plain_answer.parquet"
export TOTAL_EPOCHS=7
export PROJECT_NAME=llm-rl-reasoning
export EXPERIMENT_NAME=sft_v2_plain_answer_qwen3_0.6b_fixed
export LOGGER='["console"]'

bash run_sft.sh 4 \
    /home/hadoop-risk-control-algo/dolphinfs_ssd_hadoop-risk-control-algo/oyyx/train/models/sft_coldstart/qwen3-0.6b-v2-plain-answer
