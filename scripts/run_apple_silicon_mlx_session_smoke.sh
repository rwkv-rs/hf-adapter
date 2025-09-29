#!/usr/bin/env bash
# Apple Silicon MLX serving-style generation session smoke.

set -euo pipefail

source "$(dirname "$0")/_hf_script_common.sh"
rwkv7_prepare_results

MODEL="${1:-${MODEL:-}}"
DTYPE="${DTYPE:-fp16}"
PROMPT="${PROMPT:-The quick brown fox}"
STEP_SIZES="${STEP_SIZES:-4,4}"
QUANTIZATION="${QUANTIZATION:-none}"
QUANT_MIN_PARAMS="${QUANT_MIN_PARAMS:-8000000}"
QUANT_BACKEND="${QUANT_BACKEND:-affine}"

rwkv7_require_model "${MODEL}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export RWKV7_NATIVE_MODEL_JIT="${RWKV7_NATIVE_MODEL_JIT:-0}"

args=(
  "${MODEL}"
  --prompt "${PROMPT}"
  --step-sizes "${STEP_SIZES}"
  --dtype "${DTYPE}"
  --quantization "${QUANTIZATION}"
  --quant-min-params "${QUANT_MIN_PARAMS}"
  --quant-backend "${QUANT_BACKEND}"
  --results "${RESULTS}"
  --require-mlx
)
if [[ "${JSON_ONLY:-0}" == "1" ]]; then
  args+=(--json-only)
fi
if [[ "${SKIP_SPECIAL_TOKENS:-0}" == "1" ]]; then
  args+=(--skip-special-tokens)
fi

rwkv7_print_env
rwkv7_log "Apple Silicon MLX generation session smoke"
rwkv7_run "${PYTHON_BIN}" scripts/mlx_session_smoke.py "${args[@]}"
