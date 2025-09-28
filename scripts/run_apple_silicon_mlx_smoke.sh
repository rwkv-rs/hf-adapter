#!/usr/bin/env bash
# Apple Silicon MLX bridge smoke for RWKV-7 HF checkpoints.

set -euo pipefail

USER_DEVICE="${DEVICE:-}"  # reserved for symmetry with MPS wrappers
source "$(dirname "$0")/_hf_script_common.sh"
rwkv7_prepare_results

MODEL="${1:-${MODEL:-}}"
DTYPE="${DTYPE:-fp16}"
BATCH_SIZE="${BATCH_SIZE:-1}"
TENSOR_NAME="${TENSOR_NAME:-model.layers.0.attn.r_proj.weight}"
MODEL_SIZE_LABEL="${MODEL_SIZE_LABEL:-}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

args=(
  --dtype "${DTYPE}"
  --batch-size "${BATCH_SIZE}"
  --tensor-name "${TENSOR_NAME}"
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

rwkv7_print_env
rwkv7_log "Apple Silicon MLX bridge smoke"
rwkv7_run "${PYTHON_BIN}" tests/test_apple_silicon_mlx_smoke.py "${args[@]}"
