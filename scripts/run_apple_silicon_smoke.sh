#!/usr/bin/env bash
# Apple Silicon / MPS smoke for the RWKV-7 HF native backend.

set -euo pipefail

source "$(dirname "$0")/_hf_script_common.sh"

MODEL="${1:-${MODEL:-}}"
rwkv7_require_model "${MODEL}"
rwkv7_prepare_results

DEVICE="${DEVICE:-auto}"
DTYPE="${DTYPE:-fp32}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2}"
MODEL_SIZE_LABEL="${MODEL_SIZE_LABEL:-}"
if [[ -z "${PROMPT:-}" ]]; then
  PROMPT=$'User: Hello from Apple Silicon.

Assistant:'
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
export RWKV7_NATIVE_MODEL="${RWKV7_NATIVE_MODEL:-1}"
export RWKV7_FAST_FORWARD="${RWKV7_FAST_FORWARD:-0}"
export RWKV7_FAST_CACHE="${RWKV7_FAST_CACHE:-0}"
export RWKV7_FAST_TOKEN_BACKEND="${RWKV7_FAST_TOKEN_BACKEND:-native_jit}"

rwkv7_print_env
rwkv7_log "Apple Silicon native/MPS smoke"
cmd=(
  "${PYTHON_BIN}" tests/test_apple_silicon_smoke.py
  --model "${MODEL}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --prompt "${PROMPT}" \
  --results "${RESULTS}" \
  --model-size-label "${MODEL_SIZE_LABEL}" \
  --require-apple
)
if [[ "${SKIP_TINY:-0}" == "1" ]]; then
  cmd+=(--skip-tiny)
fi
rwkv7_run "${cmd[@]}"
