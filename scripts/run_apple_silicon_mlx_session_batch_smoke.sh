#!/usr/bin/env bash
# Apple Silicon MLX interleaved multi-session generation smoke.

set -euo pipefail

source "$(dirname "$0")/_hf_script_common.sh"
rwkv7_prepare_results

MODEL="${1:-${MODEL:-}}"
DTYPE="${DTYPE:-fp16}"
PROMPT_A="${PROMPT_A:-The quick brown fox}"
PROMPT_B="${PROMPT_B:-User: Apple Silicon RWKV test. Assistant:}"
PROMPT_C="${PROMPT_C:-}"
PROMPT_D="${PROMPT_D:-}"
ROUNDS="${ROUNDS:-2,2}"
REPEAT="${REPEAT:-1}"
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
  --prompt "${PROMPT_A}"
  --prompt "${PROMPT_B}"
  --rounds "${ROUNDS}"
  --repeat "${REPEAT}"
  --dtype "${DTYPE}"
  --quantization "${QUANTIZATION}"
  --quant-min-params "${QUANT_MIN_PARAMS}"
  --quant-backend "${QUANT_BACKEND}"
  --results "${RESULTS}"
  --require-mlx
)
if [[ -n "${PROMPT_C}" ]]; then
  args+=(--prompt "${PROMPT_C}")
fi
if [[ -n "${PROMPT_D}" ]]; then
  args+=(--prompt "${PROMPT_D}")
fi
if [[ "${JSON_ONLY:-0}" == "1" ]]; then
  args+=(--json-only)
fi
if [[ "${SKIP_SPECIAL_TOKENS:-0}" == "1" ]]; then
  args+=(--skip-special-tokens)
fi

rwkv7_print_env
rwkv7_log "Apple Silicon MLX interleaved session batch smoke"
rwkv7_run "${PYTHON_BIN}" scripts/mlx_session_batch_smoke.py "${args[@]}"
