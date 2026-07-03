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
BSZ="${BSZ:-64}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1500}"
CTX_LIMIT="${CTX_LIMIT:-8192}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.28}"
TOP_K="${TOP_K:-32}"
SEED="${SEED:-42}"
PROMPT_STYLE="${PROMPT_STYLE:-fake_think}"
PROGRESS_EVERY="${PROGRESS_EVERY:-512}"
ADD_BOS="${ADD_BOS:-1}"
PREFILL_BACKEND="${PREFILL_BACKEND:-native}"
DECODE_BACKEND="${DECODE_BACKEND:-fast_token}"

rwkv7_print_env
rwkv7_log "MATH500 acceptance model=${MODEL} dataset=${DATASET} out=${OUT_DIR} rollout=${ROLLOUT} bsz=${BSZ}"

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
)
if [[ "${ADD_BOS}" == "1" ]]; then
  cmd+=(--add-bos)
fi
rwkv7_run "${cmd[@]}"

if [[ -n "${ALBATROSS_SUMMARY:-}" ]]; then
  compare_cmd=(
    "${PYTHON_BIN}" bench/compare_math500_summaries.py
    --hf-summary "${OUT_DIR}/summary.json"
    --albatross-summary "${ALBATROSS_SUMMARY}"
  )
  if [[ -n "${ALBATROSS_LOG:-}" ]]; then
    compare_cmd+=(--albatross-log "${ALBATROSS_LOG}")
  fi
  rwkv7_log "MATH500 HF vs Albatross comparison"
  rwkv7_run "${compare_cmd[@]}"
fi

rwkv7_log "MATH500 acceptance complete: ${OUT_DIR}/summary.json"
