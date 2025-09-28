#!/usr/bin/env bash
# Apple Silicon full MLX recurrent RWKV-7 smoke.

set -euo pipefail

source "$(dirname "$0")/_hf_script_common.sh"
rwkv7_prepare_results

MODEL="${1:-${MODEL:-}}"
DTYPE="${DTYPE:-fp16}"
MODEL_SIZE_LABEL="${MODEL_SIZE_LABEL:-}"
TOKENS="${TOKENS:-1,2,3,4}"
CHUNK_SIZE="${CHUNK_SIZE:-2}"
CHUNK_TOLERANCE="${CHUNK_TOLERANCE:-1e-2}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1}"
TORCH_COMPARE_TOLERANCE="${TORCH_COMPARE_TOLERANCE:-0.2}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export RWKV7_NATIVE_MODEL_JIT="${RWKV7_NATIVE_MODEL_JIT:-0}"

args=(
  --dtype "${DTYPE}"
  --tokens "${TOKENS}"
  --chunk-size "${CHUNK_SIZE}"
  --chunk-tolerance "${CHUNK_TOLERANCE}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --torch-compare-tolerance "${TORCH_COMPARE_TOLERANCE}"
  --results "${RESULTS}"
  --require-apple
  --require-mlx
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
if [[ "${COMPARE_TORCH:-0}" == "1" ]]; then
  args+=(--compare-torch)
fi

rwkv7_print_env
rwkv7_log "Apple Silicon MLX recurrent model smoke"
rwkv7_run "${PYTHON_BIN}" tests/test_apple_silicon_mlx_model_smoke.py "${args[@]}"
