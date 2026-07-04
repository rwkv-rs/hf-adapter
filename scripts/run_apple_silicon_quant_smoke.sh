#!/usr/bin/env bash
# Apple Silicon / MPS native MM8/MM4 quantization smoke.

set -euo pipefail

USER_DEVICE="${DEVICE:-}"
source "$(dirname "$0")/_hf_script_common.sh"
rwkv7_prepare_results

MODEL="${1:-${MODEL:-}}"
DEVICE="${USER_DEVICE:-auto}"
DTYPE="${DTYPE:-fp32}"
QUANTIZATIONS="${QUANTIZATIONS:-mm8,mm4}"
MIN_PARAMS="${MIN_PARAMS:-8000000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1}"
MODEL_SIZE_LABEL="${MODEL_SIZE_LABEL:-}"
if [[ -z "${PROMPT:-}" ]]; then
  PROMPT=$'User: Apple native quant smoke.\n\nAssistant:'
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
export RWKV7_NATIVE_MODEL="${RWKV7_NATIVE_MODEL:-1}"
export RWKV7_FAST_FORWARD="${RWKV7_FAST_FORWARD:-0}"
export RWKV7_FAST_CACHE="${RWKV7_FAST_CACHE:-0}"
export RWKV7_FAST_TOKEN_BACKEND="${RWKV7_FAST_TOKEN_BACKEND:-native_jit}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

args=(
  --device "${DEVICE}"
  --dtype "${DTYPE}"
  --quantizations "${QUANTIZATIONS}"
  --min-params "${MIN_PARAMS}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --prompt "${PROMPT}"
  --results "${RESULTS}"
  --require-apple
)
if [[ -n "${MODEL}" ]]; then
  args+=(--model "${MODEL}")
fi
if [[ -n "${MODEL_SIZE_LABEL}" ]]; then
  args+=(--model-size-label "${MODEL_SIZE_LABEL}")
fi
if [[ "${SKIP_TINY:-0}" == "1" ]]; then
  args+=(--skip-tiny)
fi

rwkv7_print_env
rwkv7_log "Apple Silicon native/MPS MM8/MM4 quantization smoke"
rwkv7_run "${PYTHON_BIN}" tests/test_apple_silicon_quant_smoke.py "${args[@]}"
