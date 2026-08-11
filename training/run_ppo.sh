#!/usr/bin/env bash
# PPO (完整 RLHF: Actor + Critic + Reward + Reference) | Qwen3-0.6B/1.7B
# FSDP training + vLLM rollout | 8x A100 80G
#
# 对应 docs/方案计划.md Week3-4：PPO 跑通（四模型协同），KL系数消融实验
#
# 用法：
#   bash run_ppo.sh <nproc_per_node> [其他 hydra 覆盖参数...]
#
# 常用环境变量覆盖：
#   MODEL_PATH        Actor/Critic 初始化模型路径（一般为 SFT 冷启动 checkpoint）
#   KL_LOSS_COEF      KL 惩罚系数，消融实验的核心变量 (默认: 0.01，见下方"崩溃修复"说明)
#   EXPERIMENT_NAME   wandb 实验名，便于区分消融实验组
#
# 示例：KL系数消融实验（3组），分别在不同 GPU 组上并行跑：
#   CUDA_VISIBLE_DEVICES=0,1,2   KL_LOSS_COEF=0.0005 EXPERIMENT_NAME=ppo_kl0.0005 bash run_ppo.sh 3
#   CUDA_VISIBLE_DEVICES=3,4,5   KL_LOSS_COEF=0.001  EXPERIMENT_NAME=ppo_kl0.001  bash run_ppo.sh 3
#   CUDA_VISIBLE_DEVICES=6,7     KL_LOSS_COEF=0.005  EXPERIMENT_NAME=ppo_kl0.005  bash run_ppo.sh 2
#
# ============================================================================
# 崩溃修复说明 v2（2026-08-07，对应 docs/实验结果.md 实验④ PPO 的训练崩溃分析）：
#
# 【第一轮修复（已失败）】KL_LOSS_COEF=0.001, CRITIC_WARMUP=0, temperature=1.0
# 在 step5~20 附近发生 response_length 爆炸，reward 塌陷到 -1，116步内未恢复。
#
# 【第二轮修复（已失败）】CRITIC_WARMUP=5, KL_LOSS_COEF=0.005, CRITIC_LR=5e-6,
# temperature=0.8。前11步确实明显更稳定（vf_explained_var 从 -2.674 改善到接近
# 0，response_length 稳定在 90-125），但 warmup 结束恢复正常 actor+critic 交替
# 更新后，同样的长度爆炸在 step12 又发生了（只是从 step5 推迟到了 step12），
# 说明仅靠短 warmup 不足以根治，critic 仍然不够稳健。
#
# 【根因进一步确认】critic 从随机 value head 冷启动，早期 GAE advantage 估计
# 不准，又没有 GRPO 那种组内相对基线兜底，一旦某次更新给"变长输出"了错误的
# 正向 advantage，策略梯度会持续强化这个方向（输出越长→critic越难估值→advantage
# 越不准→进一步强化变长，正反馈循环），直至打满 max_response_length。
#
# 【第三轮修复（当前默认值，多道防线同时生效）】：
#   1. CRITIC_WARMUP=20         相比第二轮的4倍，覆盖之前实际崩溃发生的 step12，
#                               给 critic 足够时间把 vf_explained_var 拉到真正稳定
#   2. PPO_TRUNCATED_EXTRA_PENALTY=2.0（通过 hydra 的
#                               custom_reward_function.reward_kwargs 传给
#                               reward/rule_reward.py::compute_score）
#                               新增机制：对"提取不到answer且长度接近上限"的
#                               疑似截断样本，在 no_answer_score=-1.0 基础上叠加
#                               最高 -2.0 的额外惩罚（总计最低 -3.0），让"截断"比
#                               "普通答错"惩罚更陡岭，给 critic/actor 更早、更明确
#                               的负反馈，从 reward 层面直接抑制长度失控的正反馈循环
#                               （需搭配 verl/workers/reward_manager/naive.py 的
#                               response_length_ratio 传递支持，已完成）
#   3. CRITIC_CLIPRANGE_VALUE=0.2  相比默认 0.5 收紧，限制 critic 单步价值预测
#                               跳动幅度，减少价值函数震荡
#   4. KL_LOSS_COEF=0.01        相比第二轮的 0.005 再提高一倍，更强约束 actor
#                               偏离 SFT 初始分布的速度，给 critic 争取更多追赶时间
#   5. CRITIC_LR=5e-6           保持第二轮的降低值，避免 critic 更新过猛
#   6. ROLLOUT_TEMPERATURE=0.7  相比第二轮的 0.8 进一步降低，减小 rollout 方差
#   7. TEST_FREQ=2              保持加密验证，便于尽早发现异常
#
# 如果上述默认值仍然复现崩溃，说明问题可能不仅仅是 critic 冷启动，建议考虑：
#   - 降低 PPO_MINI_BATCH_SIZE（增加 actor 更新频率但每次幅度更小）
#   - 尝试 algorithm.adv_estimator=grpo 或 rloo 作为 PPO 的替代方案
#   - 直接降低 max_response_length（如512）降低长度爆炸的上限与代价
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
# GAE 参数（消融点：advantage estimation）
GAMMA=${GAMMA:-1.0}
LAM=${LAM:-0.95}
# critic warmup：训练开始的前 N 步只更新 critic、不更新 actor，让价值函数先
# 追上真实 reward 分布，避免早期不准的 advantage 误导 actor（见上方崩溃修复说明）。
# 第二轮用 5 步仅能支撑到 step12，因此提高到 20 以覆盖实际崩溃点。
CRITIC_WARMUP=${CRITIC_WARMUP:-20}
# critic 价值函数裁剪范围：相比默认 0.5 收紧，限制单步价值预测跳动幅度
CRITIC_CLIPRANGE_VALUE=${CRITIC_CLIPRANGE_VALUE:-0.2}
# rollout 采样温度：默认 1.0 方差较大，容易在 critic 尚不稳定时采出极端样本
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-0.7}
# 截断样本额外惩罚（通过 hydra 的 custom_reward_function.reward_kwargs 传给
# reward/rule_reward.py::compute_score，比环境变量更可靠，不依赖 Ray 分布式
# 进程的环境继承行为）：默认 0 不启用，设为>0 后对"提取不到answer且长度
# 接近上限"的样本叠加额外惩罚
PPO_TRUNCATED_EXTRA_PENALTY=${PPO_TRUNCATED_EXTRA_PENALTY:-2.0}
PPO_LENGTH_PENALTY_START_RATIO=${PPO_LENGTH_PENALTY_START_RATIO:-0.9}

ROLLOUT_TP=${ROLLOUT_TP:-1}
# 注意：hybrid_engine 模式下 FSDP(actor/critic/ref) 和 vLLM(rollout) colocate 在同一张卡上，
# PPO 还要额外常驻一个 Critic 模型，显存更紧张，保守设置为 0.3。
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.3}
ROLLOUT_N=${ROLLOUT_N:-1}

PROJECT_NAME=${PROJECT_NAME:-llm-rl-reasoning}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-ppo_gsm8k_qwen3_0.6b_kl${KL_LOSS_COEF}}
SAVE_FREQ=${SAVE_FREQ:-20}
TEST_FREQ=${TEST_FREQ:-2}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-15}
# wandb 需要走代理才能连通外网；仅对 wandb 客户端生效，不污染全局 http(s)_proxy
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
