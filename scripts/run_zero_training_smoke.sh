#!/usr/bin/env bash
# One-click DeepSpeed ZeRO-2/ZeRO-3 HF Trainer smoke wrapper.
#
# Usage:
#   bash scripts/run_zero_training_smoke.sh /path/to/rwkv7-g1d-0.1b-hf
#   NPROC_PER_NODE=2 ZERO_STAGE=both bash scripts/run_zero_training_smoke.sh "$MODEL"

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_hf_script_common.sh"

MODEL="${MODEL:-${1:-}}"
rwkv7_require_model "${MODEL}"
rwkv7_prepare_results

ZERO_STAGE="${ZERO_STAGE:-both}"
CONFIG_DIR="${CONFIG_DIR:-configs/deepspeed}"
MAX_LENGTH="${MAX_LENGTH:-64}"
MAX_STEPS="${MAX_STEPS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
DATASET_REPEATS="${DATASET_REPEATS:-4}"
OPTIONAL_DEEPSPEED="${OPTIONAL_DEEPSPEED:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-}"

rwkv7_print_env
rwkv7_log "DeepSpeed ZeRO smoke model=${MODEL} zero_stage=${ZERO_STAGE} nproc=${NPROC_PER_NODE:-single} results=${RESULTS}"

script_args=(
  tests/test_deepspeed_training_smoke.py
  --model "${MODEL}"
  --config-dir "${CONFIG_DIR}"
  --zero-stage "${ZERO_STAGE}"
  --attn-mode "${ATTN_MODE}"
  --train-dtype "${TRAIN_DTYPE}"
  --max-steps "${MAX_STEPS}"
  --batch-size "${BATCH_SIZE}"
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}"
  --dataset-repeats "${DATASET_REPEATS}"
  --max-length "${MAX_LENGTH}"
  --results "${RESULTS}"
)
if [[ "${OPTIONAL_DEEPSPEED}" == "1" ]]; then
  script_args+=(--optional)
fi

if [[ -n "${NPROC_PER_NODE}" && "${NPROC_PER_NODE}" != "1" ]]; then
  torchrun_cmd=(torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" "${script_args[@]}")
  rwkv7_run "${torchrun_cmd[@]}"
else
  rwkv7_run "${PYTHON_BIN}" "${script_args[@]}"
fi

rwkv7_log "DeepSpeed ZeRO smoke complete"
