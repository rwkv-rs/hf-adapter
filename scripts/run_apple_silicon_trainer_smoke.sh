#!/usr/bin/env bash
# Tiny Apple Silicon / MPS HF Trainer smoke for the RWKV-7 native backend.

set -euo pipefail

source "$(dirname "$0")/_hf_script_common.sh"
rwkv7_prepare_results

DEVICE="${DEVICE:-auto}"
DTYPE="${DTYPE:-fp32}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LENGTH="${LENGTH:-8}"
MAX_STEPS="${MAX_STEPS:-1}"
LR="${LR:-1e-3}"
REQUIRE_PEFT="${REQUIRE_PEFT:-0}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
export RWKV7_NATIVE_MODEL="${RWKV7_NATIVE_MODEL:-1}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

args=(
  --device "${DEVICE}"
  --dtype "${DTYPE}"
  --batch-size "${BATCH_SIZE}"
  --length "${LENGTH}"
  --max-steps "${MAX_STEPS}"
  --lr "${LR}"
  --results "${RESULTS}"
  --require-apple
)
if [[ "${REQUIRE_PEFT}" != "0" && "${REQUIRE_PEFT}" != "false" && "${REQUIRE_PEFT}" != "False" ]]; then
  args+=(--require-peft)
fi

rwkv7_print_env
rwkv7_log "Apple Silicon native/MPS HF Trainer smoke"
rwkv7_run "${PYTHON_BIN}" tests/test_apple_silicon_trainer_smoke.py "${args[@]}"
