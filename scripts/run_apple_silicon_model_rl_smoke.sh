#!/usr/bin/env bash
# Apple Silicon / MPS 0.1B+ HF model TRL DPO/GRPO + PEFT LoRA smoke.

set -euo pipefail

USER_ATTN_MODE="${ATTN_MODE:-}"
source "$(dirname "$0")/_hf_script_common.sh"

MODEL="${1:-${MODEL:-}}"
rwkv7_require_model "${MODEL}"
rwkv7_prepare_results

DEVICE="${DEVICE:-auto}"
DTYPE="${DTYPE:-fp32}"
ATTN_MODE="${USER_ATTN_MODE:-chunk}"
MAX_LENGTH="${MAX_LENGTH:-8}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_STEPS="${MAX_STEPS:-1}"
DATASET_REPEATS="${DATASET_REPEATS:-2}"
LR="${LR:-1e-4}"
GRPO_MAX_COMPLETION_LENGTH="${GRPO_MAX_COMPLETION_LENGTH:-1}"
LORA_R="${LORA_R:-4}"
LORA_ALPHA="${LORA_ALPHA:-8}"
BACKEND="${BACKEND:-rl}"
REQUIRE_PEFT="${REQUIRE_PEFT:-1}"
REQUIRE_TRL="${REQUIRE_TRL:-1}"
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
  --attn-mode "${ATTN_MODE}"
  --max-length "${MAX_LENGTH}"
  --batch-size "${BATCH_SIZE}"
  --max-steps "${MAX_STEPS}"
  --dataset-repeats "${DATASET_REPEATS}"
  --lr "${LR}"
  --grpo-max-completion-length "${GRPO_MAX_COMPLETION_LENGTH}"
  --lora-r "${LORA_R}"
  --lora-alpha "${LORA_ALPHA}"
  --backend "${BACKEND}"
  --results "${RESULTS}"
  --require-apple
)
if [[ -n "${MODEL_SIZE_LABEL}" ]]; then
  args+=(--model-size-label "${MODEL_SIZE_LABEL}")
fi
if [[ "${REQUIRE_PEFT}" != "0" && "${REQUIRE_PEFT}" != "false" && "${REQUIRE_PEFT}" != "False" ]]; then
  args+=(--require-peft)
fi
if [[ "${REQUIRE_TRL}" != "0" && "${REQUIRE_TRL}" != "false" && "${REQUIRE_TRL}" != "False" ]]; then
  args+=(--require-trl)
fi

rwkv7_print_env
rwkv7_log "Apple Silicon native/MPS real-model TRL DPO/GRPO + PEFT LoRA smoke"
rwkv7_run "${PYTHON_BIN}" tests/test_apple_silicon_model_training_smoke.py "${args[@]}"
