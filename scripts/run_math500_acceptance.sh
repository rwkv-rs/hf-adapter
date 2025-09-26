#!/usr/bin/env bash
# One-click MATH500 avg@64 acceptance run for the RWKV-7 HF adapter.
#
# Example on the 4090 validation host:
#   MODEL=/workspace/models/rwkv7/rwkv7-g1d-0.4b-hf \
#   DATASET=/workspace/projects/Albatross/faster3a_2605/dataset/MATH500.jsonl \
#   OUT_DIR=/tmp/math500_hf_dynamic_full_avg64 \
#   bash scripts/run_math500_acceptance.sh
#
# Optional: compare at the end of this HF run when an Albatross summary exists:
#   ALBATROSS_SUMMARY=/tmp/albatross_math500_full_avg64/summary.json \
#   ALBATROSS_LOG=/tmp/albatross_math500_full_avg64.log \
#   bash scripts/run_math500_acceptance.sh

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_hf_script_common.sh"

MODEL="${MODEL:-${1:-}}"
rwkv7_require_model "${MODEL}"

DATASET="${DATASET:-/workspace/projects/Albatross/faster3a_2605/dataset/MATH500.jsonl}"
if [[ ! -f "${DATASET}" ]]; then
  echo "DATASET does not exist: ${DATASET}" >&2
  exit 2
fi

OUT_DIR="${OUT_DIR:-/tmp/math500_hf_dynamic_full_avg64}"
ROLLOUT="${ROLLOUT:-64}"
LIMIT="${LIMIT:-0}"
BSZ="${BSZ:-128}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1500}"
CTX_LIMIT="${CTX_LIMIT:-8192}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.28}"
TOP_K="${TOP_K:-32}"
SEED="${SEED:-43}"
PROMPT_STYLE="${PROMPT_STYLE:-fake_think}"
PROGRESS_EVERY="${PROGRESS_EVERY:-5000}"
ADD_BOS="${ADD_BOS:-1}"
PREFILL_BACKEND="${PREFILL_BACKEND:-native}"
DECODE_BACKEND="${DECODE_BACKEND:-fast_token}"
DEFER_VERIFICATION="${DEFER_VERIFICATION:-1}"
VERIFY_WORKERS="${VERIFY_WORKERS:-4}"
SUMMARY_SPEED_TIMING="${SUMMARY_SPEED_TIMING:-generation}"
DEFER_TEXT_DECODE="${DEFER_TEXT_DECODE:-1}"
ACCEPTANCE_MIN_PASS_AT_ROLLOUT="${ACCEPTANCE_MIN_PASS_AT_ROLLOUT:-0.370}"
ACCEPTANCE_MIN_SUMMARY_SPEED_RATIO="${ACCEPTANCE_MIN_SUMMARY_SPEED_RATIO:-2.0}"
ACCEPTANCE_MIN_DECODE_SPEED_RATIO="${ACCEPTANCE_MIN_DECODE_SPEED_RATIO:-2.0}"
ACCEPTANCE_FAIL_ON_GATE="${ACCEPTANCE_FAIL_ON_GATE:-1}"

rwkv7_print_env
rwkv7_log "MATH500 acceptance model=${MODEL} dataset=${DATASET} out=${OUT_DIR} rollout=${ROLLOUT} bsz=${BSZ} seed=${SEED}"

cmd=(
  "${PYTHON_BIN}" bench/eval_math500_hf.py
  --hf-dir "${MODEL}"
  --dataset "${DATASET}"
  --out-dir "${OUT_DIR}"
  --rollout "${ROLLOUT}"
  --limit "${LIMIT}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --ctx-limit "${CTX_LIMIT}"
  --temperature "${TEMPERATURE}"
  --top-p "${TOP_P}"
  --top-k "${TOP_K}"
  --seed "${SEED}"
  --prompt-style "${PROMPT_STYLE}"
  --dtype "${DTYPE}"
  --device "${DEVICE}"
  --progress-every "${PROGRESS_EVERY}"
  --dynamic-batching
  --bsz "${BSZ}"
  --prefill-backend "${PREFILL_BACKEND}"
  --decode-backend "${DECODE_BACKEND}"
  --summary-speed-timing "${SUMMARY_SPEED_TIMING}"
)
if [[ "${ADD_BOS}" == "1" ]]; then
  cmd+=(--add-bos)
fi
if [[ "${DEFER_VERIFICATION}" == "1" ]]; then
  cmd+=(--defer-verification --verify-workers "${VERIFY_WORKERS}")
fi
if [[ "${DEFER_TEXT_DECODE}" == "1" ]]; then
  cmd+=(--defer-text-decode)
fi
rwkv7_run "${cmd[@]}"

if [[ -n "${ALBATROSS_SUMMARY:-}" ]]; then
  COMPARISON_OUT_DIR="${COMPARISON_OUT_DIR:-${OUT_DIR}/comparison}"
  mkdir -p "${COMPARISON_OUT_DIR}"
  compare_cmd=(
    "${PYTHON_BIN}" bench/compare_math500_summaries.py
    --hf-summary "${OUT_DIR}/summary.json"
    --albatross-summary "${ALBATROSS_SUMMARY}"
    --require-compatible-shape
    --min-pass-at-rollout "${ACCEPTANCE_MIN_PASS_AT_ROLLOUT}"
    --min-summary-speed-ratio "${ACCEPTANCE_MIN_SUMMARY_SPEED_RATIO}"
    --json-output "${COMPARISON_OUT_DIR}/comparison.json"
    --text-output "${COMPARISON_OUT_DIR}/comparison.txt"
  )
  if [[ -n "${ALBATROSS_LOG:-}" ]]; then
    compare_cmd+=(--albatross-log "${ALBATROSS_LOG}")
    if [[ -n "${ACCEPTANCE_MIN_DECODE_SPEED_RATIO}" ]]; then
      compare_cmd+=(--min-decode-speed-ratio "${ACCEPTANCE_MIN_DECODE_SPEED_RATIO}")
    fi
  fi
  if [[ "${ACCEPTANCE_FAIL_ON_GATE}" == "1" ]]; then
    compare_cmd+=(--fail-on-gate)
  fi
  rwkv7_log "MATH500 HF vs Albatross comparison"
  rwkv7_run "${compare_cmd[@]}"
  rwkv7_log "MATH500 comparison artifacts: ${COMPARISON_OUT_DIR}/comparison.{json,txt}"
fi

rwkv7_log "MATH500 acceptance complete: ${OUT_DIR}/summary.json"
