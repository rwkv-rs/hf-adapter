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
PROMPT_E="${PROMPT_E:-}"
PROMPT_F="${PROMPT_F:-}"
PROMPT_G="${PROMPT_G:-}"
PROMPT_H="${PROMPT_H:-}"
PROMPTS_FILE="${PROMPTS_FILE:-}"
EXTRA_PROMPTS="${EXTRA_PROMPTS:-}"
SESSION_COUNT="${SESSION_COUNT:-0}"
ROUNDS="${ROUNDS:-2,2}"
REPEAT="${REPEAT:-1}"
QUANTIZATION="${QUANTIZATION:-none}"
QUANT_MIN_PARAMS="${QUANT_MIN_PARAMS:-8000000}"
QUANT_BACKEND="${QUANT_BACKEND:-affine}"
WKV_BACKEND="${WKV_BACKEND:-reference}"

rwkv7_require_model "${MODEL}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export RWKV7_NATIVE_MODEL_JIT="${RWKV7_NATIVE_MODEL_JIT:-0}"

prompts=()
rwkv7_add_prompt() {
  local prompt="$1"
  if [[ -n "${prompt}" ]]; then
    prompts+=("${prompt}")
  fi
}

rwkv7_add_prompt "${PROMPT_A}"
rwkv7_add_prompt "${PROMPT_B}"
rwkv7_add_prompt "${PROMPT_C}"
rwkv7_add_prompt "${PROMPT_D}"
rwkv7_add_prompt "${PROMPT_E}"
rwkv7_add_prompt "${PROMPT_F}"
rwkv7_add_prompt "${PROMPT_G}"
rwkv7_add_prompt "${PROMPT_H}"

if [[ -n "${PROMPTS_FILE}" ]]; then
  if [[ ! -f "${PROMPTS_FILE}" ]]; then
    echo "PROMPTS_FILE does not exist: ${PROMPTS_FILE}" >&2
    exit 2
  fi
  while IFS= read -r prompt || [[ -n "${prompt}" ]]; do
    rwkv7_add_prompt "${prompt}"
  done < "${PROMPTS_FILE}"
fi

if [[ -n "${EXTRA_PROMPTS}" ]]; then
  while IFS= read -r prompt || [[ -n "${prompt}" ]]; do
    rwkv7_add_prompt "${prompt}"
  done <<< "${EXTRA_PROMPTS}"
fi

if ! [[ "${SESSION_COUNT}" =~ ^[0-9]+$ ]]; then
  echo "SESSION_COUNT must be a non-negative integer, got: ${SESSION_COUNT}" >&2
  exit 2
fi
while (( ${#prompts[@]} < SESSION_COUNT )); do
  next_index=$(( ${#prompts[@]} + 1 ))
  prompts+=("Synthetic concurrent MLX session ${next_index}: validate RWKV-7 state cache pressure and interleaved decode.")
done

args=(
  "${MODEL}"
  --rounds "${ROUNDS}"
  --repeat "${REPEAT}"
  --dtype "${DTYPE}"
  --quantization "${QUANTIZATION}"
  --quant-min-params "${QUANT_MIN_PARAMS}"
  --quant-backend "${QUANT_BACKEND}"
  --wkv-backend "${WKV_BACKEND}"
  --results "${RESULTS}"
  --require-mlx
)
for prompt in "${prompts[@]}"; do
  args+=(--prompt "${prompt}")
done
if [[ "${JSON_ONLY:-0}" == "1" ]]; then
  args+=(--json-only)
fi
if [[ "${SKIP_SPECIAL_TOKENS:-0}" == "1" ]]; then
  args+=(--skip-special-tokens)
fi

rwkv7_print_env
rwkv7_log "Apple Silicon MLX interleaved session batch smoke"
rwkv7_run "${PYTHON_BIN}" scripts/mlx_session_batch_smoke.py "${args[@]}"
