#!/usr/bin/env bash
# SFT cold-start | Qwen3-0.6B/1.7B-Base | FSDP engine
#
# Usage:
#   bash run_sft.sh <nproc_per_node> <save_path> [extra hydra overrides...]
#
# Example — scale comparison (0.6B vs 1.7B) in parallel on disjoint GPUs:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 MODEL_PATH=/path/to/model/Qwen3-0.6B-Base \
#     bash run_sft.sh 4 ../models/sft_coldstart/qwen3-0.6b
#   CUDA_VISIBLE_DEVICES=4,5,6,7 MODEL_PATH=/path/to/model/Qwen3-1.7B-Base \
#     bash run_sft.sh 4 ../models/sft_coldstart/qwen3-1.7b

set -xeuo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: run_sft.sh <nproc_per_node> <save_path> [extra hydra overrides...]"
    exit 1
fi

nproc_per_node=$1
save_path=$2
shift 2

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

########################### user-adjustable ###########################
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-0.6B-Base}
TRAIN_FILE=${TRAIN_FILE:-${REPO_ROOT}/data/processed/sft_coldstart_mix_train.parquet}
VAL_FILE=${VAL_FILE:-${REPO_ROOT}/data/processed/sft_coldstart_mix_val.parquet}
MICRO_BATCH_SIZE_PER_GPU=${MICRO_BATCH_SIZE_PER_GPU:-4}
MAX_LENGTH=${MAX_LENGTH:-1536}
LR=${LR:-1e-5}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-15}
SP_SIZE=${SP_SIZE:-1}
PROJECT_NAME=${PROJECT_NAME:-llm-rl-reasoning}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-sft_coldstart_$(basename "${MODEL_PATH}")}
# LOGGER can be '["console"]' or '["console","wandb"]'
LOGGER=${LOGGER:-'["console","wandb"]'}
# wandb needs a proxy to reach the internet; scoped to the trainer so it
# does not pollute the global http_proxy/https_proxy env vars.
WANDB_PROXY=${WANDB_PROXY:-http://10.176.253.182:8080}
########################### end user-adjustable ###########################

torchrun --standalone --nnodes=1 --nproc_per_node=${nproc_per_node} \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.multiturn.enable=true \
    data.multiturn.messages_key=messages \
    data.multiturn.tools_key=null \
    data.max_length=${MAX_LENGTH} \
    data.truncation=right \
    data.micro_batch_size_per_gpu=${MICRO_BATCH_SIZE_PER_GPU} \
    optim.lr=${LR} \
    ulysses_sequence_parallel_size=${SP_SIZE} \
    model.partial_pretrain="${MODEL_PATH}" \
    model.trust_remote_code=true \
    use_remove_padding=true \
    trainer.default_local_dir="${save_path}" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.logger="${LOGGER}" \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.wandb_proxy="${WANDB_PROXY}" \
    "$@"
