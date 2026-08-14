#!/usr/bin/env bash
# GRPO | Qwen3-0.6B/1.7B | FSDP + vLLM rollout | 8x A100 80G
#
# Usage:
#   bash run_grpo.sh <nproc_per_node> [extra hydra overrides...]
#
# Key env-var overrides:
#   MODEL_PATH          base/SFT checkpoint path (default: Qwen3-0.6B-Base SFT cold-start ckpt)
#   ROLLOUT_N           GRPO group size (samples per prompt); main ablation variable (default: 8)
#   KL_LOSS_COEF        KL penalty coefficient (default: 0.001)
#   EXPERIMENT_NAME     wandb run name, useful for distinguishing ablation groups
#
# Example — group-size ablation (4/8/16) in parallel on disjoint GPUs:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 ROLLOUT_N=4  EXPERIMENT_NAME=grpo_gsize4  bash run_grpo.sh 4
#   CUDA_VISIBLE_DEVICES=4,5     ROLLOUT_N=8  EXPERIMENT_NAME=grpo_gsize8  bash run_grpo.sh 2
#   CUDA_VISIBLE_DEVICES=6,7     ROLLOUT_N=16 EXPERIMENT_NAME=grpo_gsize16 bash run_grpo.sh 2

set -xeuo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: run_grpo.sh <nproc_per_node> [extra hydra overrides...]"
    exit 1
fi

nproc_per_node=$1
shift 1

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

########################### user-adjustable ###########################
MODEL_PATH=${MODEL_PATH:-${REPO_ROOT}/models/sft_coldstart/qwen3-0.6b}
GSM8K_TRAIN_FILE=${GSM8K_TRAIN_FILE:-${REPO_ROOT}/data/processed/gsm8k/train.parquet}
GSM8K_TEST_FILE=${GSM8K_TEST_FILE:-${REPO_ROOT}/data/processed/gsm8k/test.parquet}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-256}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-64}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-8}
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-16}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1024}

ACTOR_LR=${ACTOR_LR:-1e-6}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.001}
ENTROPY_COEFF=${ENTROPY_COEFF:-0}

ROLLOUT_TP=${ROLLOUT_TP:-1}
# In hybrid_engine mode, FSDP (actor/ref) and vLLM (rollout) share the same
# GPU. gpu_memory_utilization is the fraction of *currently free* VRAM that
# vLLM claims at init time — setting it too high causes "No available memory
# for the cache blocks" errors when the number of parallel workers changes
# (e.g. 4-GPU runs). Conservative default: 0.35.
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.35}
ROLLOUT_N=${ROLLOUT_N:-8}  # GRPO group size — main ablation variable

PROJECT_NAME=${PROJECT_NAME:-llm-rl-reasoning}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-grpo_gsm8k_qwen3_0.6b_n${ROLLOUT_N}}
SAVE_FREQ=${SAVE_FREQ:-20}
TEST_FREQ=${TEST_FREQ:-5}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-15}
# wandb needs a proxy to reach the internet; scoped here so it does not
# pollute the global http(s)_proxy (which would affect vLLM and other
# HTTP clients during the rollout phase)
WANDB_PROXY=${WANDB_PROXY:-http://10.176.253.182:8080}

CUSTOM_REWARD_PATH=${CUSTOM_REWARD_PATH:-${REPO_ROOT}/reward/rule_reward.py}
########################### end user-adjustable ###########################

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="${GSM8K_TRAIN_FILE}"
    data.val_files="${GSM8K_TEST_FILE}"
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=True
    data.truncation='error'
)

MODEL=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF}
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=${ENTROPY_COEFF}
    actor_rollout_ref.actor.fsdp_config.param_offload=False
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
)

REF=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}
    actor_rollout_ref.ref.fsdp_config.param_offload=True
)

REWARD=(
    reward_model.reward_manager=naive
    custom_reward_function.path="${CUSTOM_REWARD_PATH}"
    custom_reward_function.name=compute_score
)

TRAINER=(
    trainer.critic_warmup=0
    trainer.logger='["console","wandb"]'
    trainer.wandb_proxy="${WANDB_PROXY}"
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.n_gpus_per_node=${nproc_per_node}
    trainer.nnodes=1
    trainer.save_freq=${SAVE_FREQ}
    trainer.test_freq=${TEST_FREQ}
    trainer.total_epochs=${TOTAL_EPOCHS}
    trainer.default_local_dir="${REPO_ROOT}/models/grpo_ckpt/${EXPERIMENT_NAME}"
)

python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${REWARD[@]}" \
    "${TRAINER[@]}" \
    "$@"
