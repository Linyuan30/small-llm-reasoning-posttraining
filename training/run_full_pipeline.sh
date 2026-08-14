#!/usr/bin/env bash
# Run the full post-training pipeline: SFT (skipped if already done) -> GRPO -> PPO -> DPO.
# Each stage runs pass@k evaluation automatically on completion, and results are
# summarised back into docs/experiments.md.
#
# Design goals: run unattended overnight; a failed stage does not block subsequent
# stages (each is independent); full stdout/stderr and generated outputs are written
# to disk for post-run inspection.
#
# Usage:
#   nohup bash training/run_full_pipeline.sh > logs/full_pipeline_$(date +%Y%m%d_%H%M%S).log 2>&1 &
#
# All key paths and hyperparameters can be overridden via env vars; see the
# user-adjustable section below.

set -uo pipefail  # intentionally no -e: a failing stage should not stop later stages

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

# GPU allocation: GPUs 0-3 (4 cards) for training.
# n_gpus_per_node must evenly divide both train_batch_size and
# ppo_mini_batch_size; 256 and 64 are both divisible by 4, so 4 GPUs
# is the safe choice (6 would not divide evenly and causes dp_size errors).
TRAIN_GPUS=${TRAIN_GPUS:-0,1,2,3}
N_TRAIN_GPUS=${N_TRAIN_GPUS:-4}
EVAL_GPU=${EVAL_GPU:-4}

EVAL_MAX_SAMPLES=${EVAL_MAX_SAMPLES:-500}
EVAL_K_LIST=${EVAL_K_LIST:-1,4,8}

# RL training scale (target: fits in one overnight run, not maximum accuracy)
RL_TOTAL_EPOCHS=${RL_TOTAL_EPOCHS:-4}
RL_TRAIN_BATCH_SIZE=${RL_TRAIN_BATCH_SIZE:-256}
RL_SAVE_FREQ=${RL_SAVE_FREQ:-10}  # save every 10 steps so early checkpoints are available for eval

# DPO preference-pair construction
DPO_NUM_SAMPLES=${DPO_NUM_SAMPLES:-8}
DPO_MAX_PROMPTS=${DPO_MAX_PROMPTS:-4000}
DPO_EPOCHS=${DPO_EPOCHS:-1}

LOG_DIR=${LOG_DIR:-${REPO_ROOT}/logs/pipeline}
RESULT_DIR=${RESULT_DIR:-${REPO_ROOT}/eval/results}
########################### end user-adjustable ###########################

mkdir -p "${LOG_DIR}" "${RESULT_DIR}"

# conda's activate script references undeclared variables (e.g. ADDR2LINE),
# which conflicts with this script's `set -u`; temporarily disable nounset
# during activation.
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
    # $1 = stage name, $2 = log file, remaining args = command to run
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
        log "===== [FAIL ] ${stage} (exit ${ec}), see ${logfile} ====="
        mark_stage "${stage}" "FAILED(exit=${ec})"
        return 1
    fi
}

############################################
# Stage 0: environment & GPU sanity check
############################################
log "===== [START] stage0_env_check ====="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv > "${LOG_DIR}/gpu_status_start.txt" 2>&1
python -c "import verl, vllm, torch; print('verl:', verl.__file__); print('vllm:', vllm.__version__); print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())" \
    > "${LOG_DIR}/env_check.log" 2>&1
cat "${LOG_DIR}/env_check.log"
mark_stage "stage0_env_check" "SUCCESS"

############################################
# Stage 1: evaluate SFT cold-start model (skip if result already exists)
############################################
SFT_RESULT="${RESULT_DIR}/sft_coldstart_qwen3_0.6b.json"
if [ -f "${SFT_RESULT}" ] && python -c "import json,sys; d=json.load(open('${SFT_RESULT}')); sys.exit(0 if 'strict_format_rate' in d.get('gsm8k',{}).get('metrics',{}) else 1)" 2>/dev/null; then
    log "===== [SKIP ] stage1_eval_sft (result exists and contains strict_format_rate) ====="
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
# Stage 2: GRPO training
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
sleep 15  # wait for Ray/vLLM processes to release GPU memory

############################################
# Stage 3: evaluate GRPO model (last global_step checkpoint)
############################################
if [ ${grpo_train_ok} -eq 0 ]; then
    GRPO_LAST_CKPT=$(ls -d "${GRPO_CKPT_DIR}"/global_step_* 2>/dev/null | sort -t_ -k3 -n | tail -1)
    if [ -n "${GRPO_LAST_CKPT}" ]; then
        # veRL FSDP saves actor weights in a huggingface sub-directory
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
        log "===== [SKIP ] stage3_eval_grpo (no checkpoint found; GRPO training may have failed) ====="
        mark_stage "stage3_eval_grpo" "SKIPPED(no_ckpt)"
    fi
else
    log "===== [SKIP ] stage3_eval_grpo (GRPO training stage failed) ====="
    mark_stage "stage3_eval_grpo" "SKIPPED(train_failed)"
fi
sleep 15

############################################
# Stage 4: PPO training
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
sleep 15  # wait for Ray/vLLM processes to release GPU memory

############################################
# Stage 5: evaluate PPO model
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
        log "===== [SKIP ] stage5_eval_ppo (no checkpoint found; PPO training may have failed) ====="
        mark_stage "stage5_eval_ppo" "SKIPPED(no_ckpt)"
    fi
else
    log "===== [SKIP ] stage5_eval_ppo (PPO training stage failed) ====="
    mark_stage "stage5_eval_ppo" "SKIPPED(train_failed)"
fi
sleep 15

############################################
# Stage 6: build DPO preference pairs (rejection sampling from the SFT model)
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
sleep 15  # wait for vLLM process to release GPU memory

############################################
# Stage 7: DPO training
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
    log "===== [SKIP ] stage7_train_dpo (preference pair construction failed) ====="
    mark_stage "stage7_train_dpo" "SKIPPED(pairs_failed)"
    dpo_train_ok=1
fi

############################################
# Stage 8: evaluate DPO model
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
    log "===== [SKIP ] stage8_eval_dpo (DPO training failed or output missing) ====="
    mark_stage "stage8_eval_dpo" "SKIPPED(train_failed)"
fi

############################################
# Stage 9: summarise results
############################################
run_stage "stage9_summarize_results" "${LOG_DIR}/stage9_summarize.log" \
    python eval/summarize_results.py

nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv > "${LOG_DIR}/gpu_status_end.txt" 2>&1

log "===== ALL STAGES DONE — stage summary: ====="
cat "${STAGE_STATUS_FILE}"
log "Full logs: ${LOG_DIR}"
