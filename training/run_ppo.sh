#!/usr/bin/env bash
# PPO (full RLHF: Actor + Critic + Reward + Reference) | Qwen3-0.6B/1.7B
# FSDP training + vLLM rollout | 8x A100 80G
#
# Usage:
#   bash run_ppo.sh <nproc_per_node> [extra hydra overrides...]
#
# Key env-var overrides:
#   MODEL_PATH        actor/critic init checkpoint (typically the SFT cold-start ckpt)
#   KL_LOSS_COEF      KL penalty coefficient — main variable in ablation runs (default: 0.01)
#   EXPERIMENT_NAME   wandb run name, useful for distinguishing ablation groups
#
# Example — KL ablation (3 groups in parallel on disjoint GPUs):
#   CUDA_VISIBLE_DEVICES=0,1,2   KL_LOSS_COEF=0.0005 EXPERIMENT_NAME=ppo_kl0.0005 bash run_ppo.sh 3
#   CUDA_VISIBLE_DEVICES=3,4,5   KL_LOSS_COEF=0.001  EXPERIMENT_NAME=ppo_kl0.001  bash run_ppo.sh 3
#   CUDA_VISIBLE_DEVICES=6,7     KL_LOSS_COEF=0.005  EXPERIMENT_NAME=ppo_kl0.005  bash run_ppo.sh 2
#
# ============================================================================
# Collapse fix history (see docs/ppo_analysis.md for full postmortem)
#
# v1 (collapsed): KL_LOSS_COEF=0.001, CRITIC_WARMUP=0, temperature=1.0
#   response_length exploded to the 1024-token cap around step 5-20;
#   reward collapsed to -1 and never recovered within 116 steps.
#
# v2 (collapsed): CRITIC_WARMUP=5, KL_LOSS_COEF=0.005, CRITIC_LR=5e-6,
#   temperature=0.8.  Metrics looked stable for the first 11 steps
#   (vf_explained_var improved from -2.674 toward 0, response_length
#   held at 90-125), but the same length explosion re-occurred at step 12
#   once warmup ended and normal actor+critic updates resumed — warmup
#   alone was not enough to stabilise the critic.
#
# Root cause: critic cold-starts from a random value head, so early GAE
#   advantages are mostly noise.  If one noisy update assigns positive
#   advantage to "produce longer output", policy gradient reinforces that
#   direction (longer -> harder to value -> noisier advantage -> further
#   reinforcement — a positive feedback loop until max_response_length).
#   GRPO's group-relative baseline sidesteps this by not needing a learned
#   value function at all.
#
# v3 (current defaults — multiple guards active simultaneously):
#   1. CRITIC_WARMUP=20          4x v2; covers the actual collapse point
#                                (step 12) and gives the critic enough time
#                                to bring vf_explained_var to a stable level
#   2. PPO_TRUNCATED_EXTRA_PENALTY=2.0  passed via
#                                custom_reward_function.reward_kwargs to
#                                reward/rule_reward.py::compute_score;
#                                applies up to -2.0 extra penalty on top of
#                                the base no_answer_score=-1.0 (total -3.0)
#                                for rollouts that look truncated (no answer
#                                extracted, length near the cap), giving
#                                critic/actor an earlier and sharper negative
#                                signal against length runaway
#                                (requires response_length_ratio forwarding
#                                in verl/workers/reward_manager/naive.py,
#                                already patched)
#   3. CRITIC_CLIPRANGE_VALUE=0.2  tighter than the default 0.5; limits
#                                per-step value prediction jumps and reduces
#                                value-function oscillation
#   4. KL_LOSS_COEF=0.01         double v2's 0.005; stronger constraint on
#                                how fast actor drifts from the SFT init,
#                                buying the critic more time to catch up
#   5. CRITIC_LR=5e-6            same as v2; prevents the critic from
#                                over-updating
#   6. ROLLOUT_TEMPERATURE=0.7   lower than v2's 0.8; reduces rollout
#                                variance
#   7. TEST_FREQ=2               frequent validation to catch anomalies early
#
# If the defaults above still produce a collapse, consider:
#   - Reducing PPO_MINI_BATCH_SIZE (smaller but more frequent actor updates)
#   - Switching to algorithm.adv_estimator=grpo or rloo
#   - Lowering max_response_length (e.g. 512) to cap the damage
# ============================================================================

set -xeuo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: run_ppo.sh <nproc_per_node> [extra hydra overrides...]"
    exit 1
fi

nproc_per_node=$1
shift 1

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

########################### user-adjustable ###########################
MODEL_PATH=${MODEL_PATH:-${REPO_ROOT}/models/sft_coldstart/qwen3-0.6b}
CRITIC_MODEL_PATH=${CRITIC_MODEL_PATH:-${MODEL_PATH}}
GSM8K_TRAIN_FILE=${GSM8K_TRAIN_FILE:-${REPO_ROOT}/data/processed/gsm8k/train.parquet}
GSM8K_TEST_FILE=${GSM8K_TEST_FILE:-${REPO_ROOT}/data/processed/gsm8k/test.parquet}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-256}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-64}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-1024}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-16384}

ACTOR_LR=${ACTOR_LR:-1e-6}
CRITIC_LR=${CRITIC_LR:-5e-6}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.01}
ENTROPY_COEFF=${ENTROPY_COEFF:-0}
# GAE parameters (ablation point: advantage estimation)
GAMMA=${GAMMA:-1.0}
LAM=${LAM:-0.95}
# Critic warmup: update only the critic for the first N steps before
# allowing actor updates, giving the value function time to converge
# before its advantages are used to update the policy (see collapse fix
# notes above). v2 used 5 steps (failed at step 12); 20 covers that.
CRITIC_WARMUP=${CRITIC_WARMUP:-20}
# Critic value-clip range: tighter than the default 0.5 to reduce
# per-step value prediction oscillation
CRITIC_CLIPRANGE_VALUE=${CRITIC_CLIPRANGE_VALUE:-0.2}
# Rollout sampling temperature: lower than the default 1.0 to reduce
# rollout variance while the critic is still warming up
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-0.7}
# Extra penalty for suspected truncated rollouts (no answer + near length
# cap). Passed via custom_reward_function.reward_kwargs — more reliable
# than env vars in a distributed Ray setup. Set to 0 to disable.
PPO_TRUNCATED_EXTRA_PENALTY=${PPO_TRUNCATED_EXTRA_PENALTY:-2.0}
PPO_LENGTH_PENALTY_START_RATIO=${PPO_LENGTH_PENALTY_START_RATIO:-0.9}

ROLLOUT_TP=${ROLLOUT_TP:-1}
# In hybrid_engine mode, FSDP (actor/critic/ref) and vLLM (rollout)
# share the same GPU; PPO also keeps a resident critic, so memory is
# tighter than GRPO — conservative at 0.3.
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.3}
ROLLOUT_N=${ROLLOUT_N:-1}

PROJECT_NAME=${PROJECT_NAME:-llm-rl-reasoning}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-ppo_gsm8k_qwen3_0.6b_kl${KL_LOSS_COEF}}
SAVE_FREQ=${SAVE_FREQ:-20}
TEST_FREQ=${TEST_FREQ:-2}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-15}
# wandb needs a proxy to reach the internet; scoped here so it does not
# pollute the global http(s)_proxy
WANDB_PROXY=${WANDB_PROXY:-http://10.176.253.182:8080}

CUSTOM_REWARD_PATH=${CUSTOM_REWARD_PATH:-${REPO_ROOT}/reward/rule_reward.py}
########################### end user-adjustable ###########################

DATA=(
    algorithm.adv_estimator=gae
    algorithm.gamma=${GAMMA}
    algorithm.lam=${LAM}
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
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
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
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.temperature=${ROLLOUT_TEMPERATURE}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
)

REF=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    actor_rollout_ref.ref.fsdp_config.param_offload=True
)

CRITIC=(
    critic.model.path="${CRITIC_MODEL_PATH}"
    critic.model.use_remove_padding=True
    critic.model.enable_gradient_checkpointing=True
    critic.optim.lr=${CRITIC_LR}
    critic.use_dynamic_bsz=True
    critic.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}
    critic.cliprange_value=${CRITIC_CLIPRANGE_VALUE}
    critic.model.fsdp_config.param_offload=False
    critic.model.fsdp_config.optimizer_offload=False
)

REWARD=(
    reward_model.reward_manager=naive
    custom_reward_function.path="${CUSTOM_REWARD_PATH}"
    custom_reward_function.name=compute_score
    +custom_reward_function.reward_kwargs.truncated_extra_penalty=${PPO_TRUNCATED_EXTRA_PENALTY}
    +custom_reward_function.reward_kwargs.length_penalty_start_ratio=${PPO_LENGTH_PENALTY_START_RATIO}
)

TRAINER=(
    trainer.balance_batch=True
    trainer.critic_warmup=${CRITIC_WARMUP}
    trainer.logger='["console","wandb"]'
    trainer.wandb_proxy="${WANDB_PROXY}"
    trainer.project_name=${PROJECT_NAME}
    trainer.experiment_name=${EXPERIMENT_NAME}
    trainer.n_gpus_per_node=${nproc_per_node}
    trainer.nnodes=1
    trainer.save_freq=${SAVE_FREQ}
    trainer.test_freq=${TEST_FREQ}
    trainer.total_epochs=${TOTAL_EPOCHS}
    trainer.default_local_dir="${REPO_ROOT}/models/ppo_ckpt/${EXPERIMENT_NAME}"
)

python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${CRITIC[@]}" \
    "${REWARD[@]}" \
    "${TRAINER[@]}" \
    "$@"
