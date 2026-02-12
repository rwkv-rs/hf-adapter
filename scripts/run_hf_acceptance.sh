#!/usr/bin/env bash
# One-click HF adapter acceptance smoke for a single converted RWKV-7 HF model.
#
# Usage:
#   MODEL=/path/to/rwkv7-g1d-0.1b-hf bash scripts/run_hf_acceptance.sh
#   bash scripts/run_hf_acceptance.sh /path/to/rwkv7-g1d-0.1b-hf
#
# Useful overrides:
#   DEVICE=cuda DTYPE=fp16 RESULTS=bench/results.jsonl RUN_TRAINING=1 bash scripts/run_hf_acceptance.sh "$MODEL"

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_hf_script_common.sh"

MODEL="${MODEL:-${1:-}}"
rwkv7_require_model "${MODEL}"
rwkv7_prepare_results

RUN_STATIC_TESTS="${RUN_STATIC_TESTS:-1}"
RUN_MODEL_SMOKE="${RUN_MODEL_SMOKE:-1}"
RUN_QUANT="${RUN_QUANT:-1}"
RUN_BENCH="${RUN_BENCH:-1}"
RUN_BATCH_SWEEP="${RUN_BATCH_SWEEP:-1}"
RUN_TRAINING="${RUN_TRAINING:-0}"

SMOKE_MAX_NEW_TOKENS="${SMOKE_MAX_NEW_TOKENS:-4}"
PROMPT_TOKENS="${PROMPT_TOKENS:-128}"
DECODE_TOKENS="${DECODE_TOKENS:-16}"
WARMUP="${WARMUP:-1}"
RUNS="${RUNS:-1}"
BATCH_SIZES="${BATCH_SIZES:-1 2}"

rwkv7_print_env
rwkv7_log "HF acceptance model=${MODEL} device=${DEVICE} dtype=${DTYPE} results=${RESULTS}"

if [[ "${RUN_STATIC_TESTS}" == "1" ]]; then
  rwkv7_log "static/no-GPU checks"
  rwkv7_run "${PYTHON_BIN}" tests/test_convert_config.py
  rwkv7_run "${PYTHON_BIN}" tests/test_batch_convert_manifest.py
  rwkv7_run "${PYTHON_BIN}" tests/test_result_tools.py
  rwkv7_run "${PYTHON_BIN}" tests/test_sync_hf_adapter_code.py
fi

if [[ "${RUN_MODEL_SMOKE}" == "1" ]]; then
  rwkv7_log "model API/generation checks"
  rwkv7_run "${PYTHON_BIN}" tests/smoke_hf_generate.py \
    --model "${MODEL}" \
    --device "${DEVICE}" \
    --max-new-tokens "${SMOKE_MAX_NEW_TOKENS}"
  rwkv7_run "${PYTHON_BIN}" tests/test_hf_api_contract.py \
    --model "${MODEL}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    --attn-mode "${ATTN_MODE}" \
    --fuse-norm "${FUSE_NORM}"
fi

if [[ "${RUN_QUANT}" == "1" ]]; then
  rwkv7_log "quantized loading checks"
  rwkv7_run "${PYTHON_BIN}" tests/test_quantized_inference.py \
    --model "${MODEL}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    --attn-mode "${ATTN_MODE}" \
    --quantization 8bit \
    --optional
  rwkv7_run "${PYTHON_BIN}" tests/test_quantized_inference.py \
    --model "${MODEL}" \
    --device "${DEVICE}" \
    --dtype "${DTYPE}" \
    --attn-mode "${ATTN_MODE}" \
    --quantization 4bit \
    --optional
fi

if [[ "${RUN_BENCH}" == "1" ]]; then
  rwkv7_log "single-model speed smoke"
  rwkv7_run "${PYTHON_BIN}" bench/bench_speed.py \
    --hf-dir "${MODEL}" \
    --backend hf \
    --dtype "${DTYPE}" \
    --device "${DEVICE}" \
    --prompt-tokens "${PROMPT_TOKENS}" \
    --decode-tokens "${DECODE_TOKENS}" \
    --warmup "${WARMUP}" \
    --runs "${RUNS}" \
    --attn-mode "${ATTN_MODE}" \
    --fuse-norm "${FUSE_NORM}" \
    --fast-cache "${FAST_CACHE}" \
    --fast-token-backend "${FAST_TOKEN_BACKEND}" \
    --results "${RESULTS}"
fi

if [[ "${RUN_BATCH_SWEEP}" == "1" ]]; then
  rwkv7_log "batch sweep smoke"
  # shellcheck disable=SC2086 # BATCH_SIZES is intentionally split into argv.
  rwkv7_run "${PYTHON_BIN}" bench/bench_batch_sweep.py \
    --hf-dir "${MODEL}" \
    --dtype "${DTYPE}" \
    --device "${DEVICE}" \
    --attn-mode "${ATTN_MODE}" \
    --fuse-norm "${FUSE_NORM}" \
    --fast-cache "${FAST_CACHE}" \
    --fast-token-backend "${FAST_TOKEN_BACKEND}" \
    --batch-sizes ${BATCH_SIZES} \
    --prompt-tokens "${PROMPT_TOKENS}" \
    --decode-tokens "${DECODE_TOKENS}" \
    --warmup "${WARMUP}" \
    --runs "${RUNS}" \
    --results "${RESULTS}"
fi

if [[ "${RUN_TRAINING}" == "1" ]]; then
  rwkv7_log "training smoke via run_hf_training_matrix.sh"
  MODELS="${MODEL}" RESULTS="${RESULTS}" DEVICE="${DEVICE}" ATTN_MODE="${ATTN_MODE}" \
    TRAIN_DTYPE="${TRAIN_DTYPE}" "${RWKV7_SCRIPT_DIR}/run_hf_training_matrix.sh"
fi

rwkv7_log "HF acceptance complete"
