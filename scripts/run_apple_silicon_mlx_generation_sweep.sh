#!/usr/bin/env bash
# Apple Silicon MLX prompt/decode length sweep.

set -euo pipefail

source "$(dirname "$0")/_hf_script_common.sh"
rwkv7_prepare_results

MODEL="${1:-${MODEL:-}}"
DTYPE="${DTYPE:-fp16}"
PROMPT_LENGTHS="${PROMPT_LENGTHS:-16,64}"
DECODE_LENGTHS="${DECODE_LENGTHS:-2,4}"
SEED_TEXT="${SEED_TEXT:-User: Apple Silicon RWKV generation sweep. Assistant: }"
CHUNK_SIZE="${CHUNK_SIZE:-0}"
CHUNK_TOLERANCE="${CHUNK_TOLERANCE:-0.2}"

rwkv7_require_model "${MODEL}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export RWKV7_NATIVE_MODEL_JIT="${RWKV7_NATIVE_MODEL_JIT:-0}"

args=(
  "${MODEL}"
  --prompt-lengths "${PROMPT_LENGTHS}"
  --decode-lengths "${DECODE_LENGTHS}"
  --seed-text "${SEED_TEXT}"
  --dtype "${DTYPE}"
  --chunk-size "${CHUNK_SIZE}"
  --chunk-tolerance "${CHUNK_TOLERANCE}"
  --results "${RESULTS}"
  --require-mlx
)
if [[ "${JSON_ONLY:-0}" == "1" ]]; then
  args+=(--json-only)
fi

rwkv7_print_env
rwkv7_log "Apple Silicon MLX prompt/decode generation sweep"
rwkv7_run "${PYTHON_BIN}" scripts/mlx_generation_sweep.py "${args[@]}"
