#!/usr/bin/env bash
# Apple Silicon / MPS converted-model generation length sweep.

set -euo pipefail

USER_DEVICE="${DEVICE:-}"
source "$(dirname "$0")/_hf_script_common.sh"

MODEL="${1:-${MODEL:-}}"
rwkv7_require_model "${MODEL}"
rwkv7_prepare_results

DEVICE="${USER_DEVICE:-auto}"
DTYPE="${DTYPE:-fp32}"
PROMPT_LENGTHS="${PROMPT_LENGTHS:-16,64,128}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4}"
MODEL_SIZE_LABEL="${MODEL_SIZE_LABEL:-}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
export RWKV7_NATIVE_MODEL="${RWKV7_NATIVE_MODEL:-1}"
export RWKV7_FAST_FORWARD="${RWKV7_FAST_FORWARD:-0}"
export RWKV7_FAST_CACHE="${RWKV7_FAST_CACHE:-0}"
export RWKV7_FAST_TOKEN_BACKEND="${RWKV7_FAST_TOKEN_BACKEND:-native_jit}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

args=(
  --model "${MODEL}"
  --device "${DEVICE}"
  --dtype "${DTYPE}"
  --prompt-lengths "${PROMPT_LENGTHS}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --results "${RESULTS}"
  --require-apple
)
if [[ -n "${MODEL_SIZE_LABEL}" ]]; then
  args+=(--model-size-label "${MODEL_SIZE_LABEL}")
fi

rwkv7_print_env
rwkv7_log "Apple Silicon native/MPS converted-model generation sweep"
rwkv7_run "${PYTHON_BIN}" tests/test_apple_silicon_model_sweep.py "${args[@]}"
