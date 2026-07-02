#!/usr/bin/env bash
# Run PEFT/Trainer/TRL smoke over one or more converted RWKV-7 HF model dirs.
#
# Usage:
#   MODELS="/path/0.4b-hf /path/1.5b-hf" bash scripts/run_hf_training_matrix.sh
#   bash scripts/run_hf_training_matrix.sh /path/0.4b-hf /path/1.5b-hf

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_hf_script_common.sh"

if [[ "$#" -gt 0 ]]; then
  MODELS=("$@")
elif [[ -n "${MODELS:-}" ]]; then
  # shellcheck disable=SC2206 # MODELS is a whitespace-separated list by design.
  MODELS=(${MODELS})
else
  echo "MODELS is required. Pass model paths as arguments or set MODELS='modelA modelB'." >&2
  exit 2
fi

MAX_LENGTH="${MAX_LENGTH:-64}"
MAX_STEPS="${MAX_STEPS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
RL_BATCH_SIZE="${RL_BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
DATASET_REPEATS="${DATASET_REPEATS:-4}"
RUN_PEFT="${RUN_PEFT:-1}"
RUN_TRAINER="${RUN_TRAINER:-1}"
RUN_RL="${RUN_RL:-1}"
RUN_RESUME="${RUN_RESUME:-0}"
RL_BACKEND="${RL_BACKEND:-both}"
RUN_DEEPSPEED="${RUN_DEEPSPEED:-0}"
RESUME_FIRST_STEPS="${RESUME_FIRST_STEPS:-1}"
RESUME_STEPS="${RESUME_STEPS:-2}"

rwkv7_prepare_results
rwkv7_print_env
rwkv7_log "HF training matrix models=${MODELS[*]} device=${DEVICE} train_dtype=${TRAIN_DTYPE} results=${RESULTS}"

for model in "${MODELS[@]}"; do
  rwkv7_require_model "${model}"
  rwkv7_log "training matrix model=${model}"

  if [[ "${RUN_PEFT}" == "1" ]]; then
    rwkv7_run "${PYTHON_BIN}" tests/test_peft_lora.py \
      --model "${model}" \
      --device "${DEVICE}" \
      --attn-mode "${ATTN_MODE}"
  fi

  if [[ "${RUN_TRAINER}" == "1" ]]; then
    rwkv7_run "${PYTHON_BIN}" tests/test_hf_training_smoke.py \
      --model "${model}" \
      --device "${DEVICE}" \
      --attn-mode "${ATTN_MODE}" \
      --train-dtype "${TRAIN_DTYPE}" \
      --max-steps "${MAX_STEPS}" \
      --batch-size "${BATCH_SIZE}" \
      --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
      --dataset-repeats "${DATASET_REPEATS}" \
      --max-length "${MAX_LENGTH}" \
      --backend both \
      --results "${RESULTS}"
  fi

  if [[ "${RUN_RL}" == "1" ]]; then
    rwkv7_run "${PYTHON_BIN}" tests/test_hf_rl_training_smoke.py \
      --model "${model}" \
      --device "${DEVICE}" \
      --attn-mode "${ATTN_MODE}" \
      --train-dtype "${TRAIN_DTYPE}" \
      --max-steps "${MAX_STEPS}" \
      --batch-size "${RL_BATCH_SIZE}" \
      --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
      --dataset-repeats "${DATASET_REPEATS}" \
      --max-length "${MAX_LENGTH}" \
      --backend "${RL_BACKEND}" \
      --results "${RESULTS}"
  fi

  if [[ "${RUN_RESUME}" == "1" ]]; then
    rwkv7_run "${PYTHON_BIN}" tests/test_hf_trainer_resume_smoke.py \
      --model "${model}" \
      --device "${DEVICE}" \
      --attn-mode "${ATTN_MODE}" \
      --train-dtype "${TRAIN_DTYPE}" \
      --first-steps "${RESUME_FIRST_STEPS}" \
      --resume-steps "${RESUME_STEPS}" \
      --batch-size "${BATCH_SIZE}" \
      --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
      --dataset-repeats "${DATASET_REPEATS}" \
      --max-length "${MAX_LENGTH}" \
      --results "${RESULTS}"
  fi

  if [[ "${RUN_DEEPSPEED}" == "1" ]]; then
    RESULTS="${RESULTS}" DEVICE="${DEVICE}" TRAIN_DTYPE="${TRAIN_DTYPE}" ATTN_MODE="${ATTN_MODE}" \
      MAX_LENGTH="${MAX_LENGTH}" MAX_STEPS="${MAX_STEPS}" BATCH_SIZE="${BATCH_SIZE}" \
      GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS}" DATASET_REPEATS="${DATASET_REPEATS}" \
      "${RWKV7_SCRIPT_DIR}/run_zero_training_smoke.sh" "${model}"
  fi
done

rwkv7_log "HF training matrix complete"
