#!/usr/bin/env bash
# 一键跑通 SFT(已完成,自动跳过) -> GRPO -> PPO -> DPO 全流程，每阶段跑完自动做 pass@k 评估，
# 并把结果汇总回写 docs/实验结果.md / docs/流程.md。
#
# 设计目标：挂后台整晚跑，无需人工介入；某一阶段失败不阻塞后续阶段（相互独立执行），
# 每个阶段、每条命令的完整 stdout/stderr 和推理生成结果都落盘，便于早上起来复盘。
#
# 用法：
#   nohup bash training/run_full_pipeline.sh > logs/full_pipeline_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#
# 可通过环境变量覆盖关键路径/超参，详见下方 user-adjustable 区域。

set -uo pipefail  # 注意：不用 -e，单个阶段失败要继续跑后面的阶段

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

########################### user-adjustable ###########################
CONDA_ENV=${CONDA_ENV:-rl}
CONDA_SH=${CONDA_SH:-/home/sankuai/conda/etc/profile.d/conda.sh}

SFT_MODEL_PATH=${SFT_MODEL_PATH:-${REPO_ROOT}/../models/sft_coldstart/qwen3-0.6b/global_step_42}
BASE_MODEL_PATH=${BASE_MODEL_PATH:-${REPO_ROOT}/../model/Qwen3-0.6B-Base}

GSM8K_TRAIN=${GSM8K_TRAIN:-${REPO_ROOT}/data/processed/gsm8k/train.parquet}
GSM8K_TEST=${GSM8K_TEST:-${REPO_ROOT}/data/processed/gsm8k/test.parquet}
MATH_TEST=${MATH_TEST:-${REPO_ROOT}/data/processed/math/test.parquet}

# GPU 分配：0-3 共 4 张卡用于训练（GPU7 被其他任务占用，不使用）。
# 注意：n_gpus_per_node 必须能整除 train_batch_size / ppo_mini_batch_size，
# 256 和 64 都能被 4 整除，所以选 4 卡而不是 6 卡（256/6、64/6 不整除会导致 dp_size 报错）。
TRAIN_GPUS=${TRAIN_GPUS:-0,1,2,3}
N_TRAIN_GPUS=${N_TRAIN_GPUS:-4}
EVAL_GPU=${EVAL_GPU:-4}

EVAL_MAX_SAMPLES=${EVAL_MAX_SAMPLES:-500}
EVAL_K_LIST=${EVAL_K_LIST:-1,4,8}

# RL 训练规模控制（整夜可跑完为目标，非追求最优效果）
RL_TOTAL_EPOCHS=${RL_TOTAL_EPOCHS:-4}
RL_TRAIN_BATCH_SIZE=${RL_TRAIN_BATCH_SIZE:-256}
RL_SAVE_FREQ=${RL_SAVE_FREQ:-10}  # 每10 step保存一次checkpoint，确保训练早期就有可评估的产物

# DPO 偏好对构造规模
DPO_NUM_SAMPLES=${DPO_NUM_SAMPLES:-8}
DPO_MAX_PROMPTS=${DPO_MAX_PROMPTS:-4000}
DPO_EPOCHS=${DPO_EPOCHS:-1}

LOG_DIR=${LOG_DIR:-${REPO_ROOT}/logs/pipeline}
RESULT_DIR=${RESULT_DIR:-${REPO_ROOT}/eval/results}
########################### end user-adjustable ###########################

mkdir -p "${LOG_DIR}" "${RESULT_DIR}"

# conda 的 activate 脚本内部有一些未声明变量引用（如 ADDR2LINE），
# 与本脚本的 `set -u` 冲突，激活期间临时关闭 nounset。
set +u
source "${CONDA_SH}"
conda activate "${CONDA_ENV}"
set -u

STAGE_STATUS_FILE="${LOG_DIR}/stage_status.txt"
: > "${STAGE_STATUS_FILE}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

mark_stage() {
    local stage="$1" status="$2"
    echo "${stage}: ${status}" >> "${STAGE_STATUS_FILE}"
}

run_stage() {
    # $1 = stage name, $2 = log file, 剩余参数 = 要执行的命令
    local stage="$1"
    local logfile="$2"
    shift 2
    log "===== [START] ${stage} ====="
    if "$@" > "${logfile}" 2>&1; then
        log "===== [DONE ] ${stage} ====="
        mark_stage "${stage}" "SUCCESS"
        return 0
    else
        local ec=$?
        log "===== [FAIL ] ${stage} (exit ${ec})，详见 ${logfile} ====="
        mark_stage "${stage}" "FAILED(exit=${ec})"
        return 1
    fi
}

############################################
# Stage 0: 环境 & GPU 检查
############################################
log "===== [START] stage0_env_check ====="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv > "${LOG_DIR}/gpu_status_start.txt" 2>&1
python -c "import verl, vllm, torch; print('verl:', verl.__file__); print('vllm:', vllm.__version__); print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())" \
    > "${LOG_DIR}/env_check.log" 2>&1
cat "${LOG_DIR}/env_check.log"
mark_stage "stage0_env_check" "SUCCESS"

############################################
# Stage 1: SFT 冷启动模型评估（若已有新版结果则跳过）
############################################
SFT_RESULT="${RESULT_DIR}/sft_coldstart_qwen3_0.6b.json"
if [ -f "${SFT_RESULT}" ] && python -c "import json,sys; d=json.load(open('${SFT_RESULT}')); sys.exit(0 if 'strict_format_rate' in d.get('gsm8k',{}).get('metrics',{}) else 1)" 2>/dev/null; then
    log "===== [SKIP ] stage1_eval_sft（结果已存在且含格式合规率字段） ====="
    mark_stage "stage1_eval_sft" "SKIPPED"
else
    run_stage "stage1_eval_sft" "${LOG_DIR}/stage1_eval_sft.log" \
        bash -c "CUDA_VISIBLE_DEVICES=${EVAL_GPU} python eval/eval_pass_at_k.py \
            --model_path '${SFT_MODEL_PATH}' \
            --test_parquet '${GSM8K_TEST}' '${MATH_TEST}' \
            --k_list '${EVAL_K_LIST}' \
            --max_samples ${EVAL_MAX_SAMPLES} \
            --output_path '${SFT_RESULT}'"
fi

############################################
# Stage 2: GRPO 训练
############################################
GRPO_EXPERIMENT_NAME="grpo_qwen3_0.6b"
GRPO_CKPT_DIR="${REPO_ROOT}/models/grpo_ckpt/${GRPO_EXPERIMENT_NAME}"
run_stage "stage2_train_grpo" "${LOG_DIR}/stage2_train_grpo.log" \
    env CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" \
        MODEL_PATH="${SFT_MODEL_PATH}" \
        GSM8K_TRAIN_FILE="${GSM8K_TRAIN}" \
        GSM8K_TEST_FILE="${GSM8K_TEST}" \
        TRAIN_BATCH_SIZE="${RL_TRAIN_BATCH_SIZE}" \
        TOTAL_EPOCHS="${RL_TOTAL_EPOCHS}" \
        SAVE_FREQ="${RL_SAVE_FREQ}" \
        EXPERIMENT_NAME="${GRPO_EXPERIMENT_NAME}" \
        RAY_DISABLE_DASHBOARD=1 \
        bash training/run_grpo.sh "${N_TRAIN_GPUS}" \
        trainer.logger='["console"]' \
        "actor_rollout_ref.actor.checkpoint.contents=[model,optimizer,extra,hf_model]"
grpo_train_ok=$?
sleep 15  # 等待 Ray/vLLM 进程完全释放 GPU 显存

############################################
# Stage 3: GRPO 模型评估（找最后一个 global_step checkpoint）
############################################
if [ ${grpo_train_ok} -eq 0 ]; then
    GRPO_LAST_CKPT=$(ls -d "${GRPO_CKPT_DIR}"/global_step_* 2>/dev/null | sort -t_ -k3 -n | tail -1)
    if [ -n "${GRPO_LAST_CKPT}" ]; then
        # actor 权重的 huggingface 格式子目录（veRL FSDP 保存结构）
        GRPO_HF_PATH="${GRPO_LAST_CKPT}/actor/huggingface"
        [ -d "${GRPO_HF_PATH}" ] || GRPO_HF_PATH="${GRPO_LAST_CKPT}"
        run_stage "stage3_eval_grpo" "${LOG_DIR}/stage3_eval_grpo.log" \
            bash -c "CUDA_VISIBLE_DEVICES=${EVAL_GPU} python eval/eval_pass_at_k.py \
                --model_path '${GRPO_HF_PATH}' \
                --test_parquet '${GSM8K_TEST}' '${MATH_TEST}' \
                --k_list '${EVAL_K_LIST}' \
                --max_samples ${EVAL_MAX_SAMPLES} \
                --output_path '${RESULT_DIR}/grpo_qwen3_0.6b.json'"
    else
        log "===== [SKIP ] stage3_eval_grpo（找不到 checkpoint，GRPO 训练可能失败） ====="
        mark_stage "stage3_eval_grpo" "SKIPPED(no_ckpt)"
    fi
else
    log "===== [SKIP ] stage3_eval_grpo（GRPO 训练阶段失败） ====="
    mark_stage "stage3_eval_grpo" "SKIPPED(train_failed)"
fi
sleep 15

############################################
# Stage 4: PPO 训练
############################################
PPO_EXPERIMENT_NAME="ppo_qwen3_0.6b"
PPO_CKPT_DIR="${REPO_ROOT}/models/ppo_ckpt/${PPO_EXPERIMENT_NAME}"
run_stage "stage4_train_ppo" "${LOG_DIR}/stage4_train_ppo.log" \
    env CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" \
        MODEL_PATH="${SFT_MODEL_PATH}" \
        GSM8K_TRAIN_FILE="${GSM8K_TRAIN}" \
        GSM8K_TEST_FILE="${GSM8K_TEST}" \
        TRAIN_BATCH_SIZE="${RL_TRAIN_BATCH_SIZE}" \
        TOTAL_EPOCHS="${RL_TOTAL_EPOCHS}" \
        SAVE_FREQ="${RL_SAVE_FREQ}" \
        EXPERIMENT_NAME="${PPO_EXPERIMENT_NAME}" \
        RAY_DISABLE_DASHBOARD=1 \
        bash training/run_ppo.sh "${N_TRAIN_GPUS}" \
        trainer.logger='["console"]' \
        "actor_rollout_ref.actor.checkpoint.contents=[model,optimizer,extra,hf_model]"
ppo_train_ok=$?
sleep 15  # 等待 Ray/vLLM 进程完全释放 GPU 显存

############################################
# Stage 5: PPO 模型评估
############################################
if [ ${ppo_train_ok} -eq 0 ]; then
    PPO_LAST_CKPT=$(ls -d "${PPO_CKPT_DIR}"/global_step_* 2>/dev/null | sort -t_ -k3 -n | tail -1)
    if [ -n "${PPO_LAST_CKPT}" ]; then
        PPO_HF_PATH="${PPO_LAST_CKPT}/actor/huggingface"
        [ -d "${PPO_HF_PATH}" ] || PPO_HF_PATH="${PPO_LAST_CKPT}"
        run_stage "stage5_eval_ppo" "${LOG_DIR}/stage5_eval_ppo.log" \
            bash -c "CUDA_VISIBLE_DEVICES=${EVAL_GPU} python eval/eval_pass_at_k.py \
                --model_path '${PPO_HF_PATH}' \
                --test_parquet '${GSM8K_TEST}' '${MATH_TEST}' \
                --k_list '${EVAL_K_LIST}' \
                --max_samples ${EVAL_MAX_SAMPLES} \
                --output_path '${RESULT_DIR}/ppo_qwen3_0.6b.json'"
    else
        log "===== [SKIP ] stage5_eval_ppo（找不到 checkpoint，PPO 训练可能失败） ====="
        mark_stage "stage5_eval_ppo" "SKIPPED(no_ckpt)"
    fi
else
    log "===== [SKIP ] stage5_eval_ppo（PPO 训练阶段失败） ====="
    mark_stage "stage5_eval_ppo" "SKIPPED(train_failed)"
fi
sleep 15

############################################
# Stage 6: DPO 偏好对构造（rejection sampling，用 SFT 模型）
############################################
DPO_PAIRS_PATH="${REPO_ROOT}/data/processed/gsm8k_dpo_pairs.jsonl"
run_stage "stage6_build_dpo_pairs" "${LOG_DIR}/stage6_build_dpo_pairs.log" \
    bash -c "CUDA_VISIBLE_DEVICES=${EVAL_GPU} python data/scripts/build_dpo_pairs.py \
        --model_path '${SFT_MODEL_PATH}' \
        --input_parquet '${GSM8K_TRAIN}' \
        --output_path '${DPO_PAIRS_PATH}' \
        --num_samples ${DPO_NUM_SAMPLES} \
        --max_prompts ${DPO_MAX_PROMPTS}"
dpo_pairs_ok=$?
sleep 15  # 等待 vLLM 进程完全释放 GPU 显存

############################################
# Stage 7: DPO 训练
############################################
DPO_OUTPUT_DIR="${REPO_ROOT}/models/dpo_ckpt/qwen3-0.6b"
if [ ${dpo_pairs_ok} -eq 0 ]; then
    run_stage "stage7_train_dpo" "${LOG_DIR}/stage7_train_dpo.log" \
        env CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" \
        accelerate launch --num_processes "${N_TRAIN_GPUS}" training/run_dpo.py \
            --model_path "${SFT_MODEL_PATH}" \
            --data_path "${DPO_PAIRS_PATH}" \
            --output_dir "${DPO_OUTPUT_DIR}" \
            --num_train_epochs "${DPO_EPOCHS}" \
            --report_to none \
            --run_name dpo_gsm8k_qwen3_0.6b
    dpo_train_ok=$?
else
    log "===== [SKIP ] stage7_train_dpo（偏好对构造失败） ====="
    mark_stage "stage7_train_dpo" "SKIPPED(pairs_failed)"
    dpo_train_ok=1
fi

############################################
# Stage 8: DPO 模型评估
############################################
if [ ${dpo_train_ok} -eq 0 ] && [ -f "${DPO_OUTPUT_DIR}/config.json" ]; then
    run_stage "stage8_eval_dpo" "${LOG_DIR}/stage8_eval_dpo.log" \
        bash -c "CUDA_VISIBLE_DEVICES=${EVAL_GPU} python eval/eval_pass_at_k.py \
            --model_path '${DPO_OUTPUT_DIR}' \
            --test_parquet '${GSM8K_TEST}' '${MATH_TEST}' \
            --k_list '${EVAL_K_LIST}' \
            --max_samples ${EVAL_MAX_SAMPLES} \
            --output_path '${RESULT_DIR}/dpo_qwen3_0.6b.json'"
else
    log "===== [SKIP ] stage8_eval_dpo（DPO 训练阶段失败或产物缺失） ====="
    mark_stage "stage8_eval_dpo" "SKIPPED(train_failed)"
fi

############################################
# Stage 9: 汇总结果，回写 docs/实验结果.md 和 docs/流程.md
############################################
run_stage "stage9_summarize_results" "${LOG_DIR}/stage9_summarize.log" \
    python eval/summarize_results.py

nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv > "${LOG_DIR}/gpu_status_end.txt" 2>&1

log "===== ALL STAGES DONE，各阶段状态汇总： ====="
cat "${STAGE_STATUS_FILE}"
log "详细日志目录：${LOG_DIR}"
