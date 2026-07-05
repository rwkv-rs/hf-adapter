#!/usr/bin/env bash
# One-click final MATH500 acceptance workflow for the RWKV-7 HF adapter.
#
# It runs:
#   1. best-bsz sweep;
#   2. full MATH500 avg@64 with the selected bsz;
#   3. optional HF-vs-Albatross summary gates;
#   4. optional uncheatable compression/logit alignment.
#
# Minimal:
#   MODEL=/path/to/hf_model DATASET=/path/to/MATH500.jsonl bash scripts/run_math500_final_acceptance.sh
#
# With Albatross comparison and compression reference:
#   MODEL=/path/to/hf_model \
#   DATASET=/workspace/projects/Albatross/faster3a_2605/dataset/MATH500.jsonl \
#   ALBATROSS_SUMMARY=/tmp/albatross_math500_full_avg64/summary.json \
#   ALBATROSS_LOG=/tmp/albatross_math500_full_avg64/run.log \
#   COMPRESSION_REFERENCE_KIND=albatross \
#   COMPRESSION_REFERENCE_ALBATROSS_DIR=/workspace/projects/Albatross/faster3a_2605 \
#   COMPRESSION_REFERENCE_ALBATROSS_MODEL=/dev/shm/rwkv7-g1f-1.5b-20260419-ctx8192.pth \
#   bash scripts/run_math500_final_acceptance.sh

set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/_hf_script_common.sh"

MODEL="${MODEL:-${1:-}}"
rwkv7_require_model "${MODEL}"

DATASET="${DATASET:-/workspace/projects/Albatross/faster3a_2605/dataset/MATH500.jsonl}"
if [[ ! -f "${DATASET}" ]]; then
  echo "DATASET does not exist: ${DATASET}" >&2
  exit 2
fi

OUT_DIR="${OUT_DIR:-/tmp/math500_final_acceptance_$(date +%Y%m%d_%H%M%S)}"
BSZ_LIST="${BSZ_LIST:-32 64 96 128 192}"
SWEEP_LIMIT="${SWEEP_LIMIT:-4}"
SWEEP_ROLLOUT="${SWEEP_ROLLOUT:-64}"
SWEEP_MAX_NEW_TOKENS="${SWEEP_MAX_NEW_TOKENS:-256}"
FULL_LIMIT="${FULL_LIMIT:-0}"
FULL_ROLLOUT="${FULL_ROLLOUT:-64}"
FULL_MAX_NEW_TOKENS="${FULL_MAX_NEW_TOKENS:-1500}"
CTX_LIMIT="${CTX_LIMIT:-8192}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.28}"
TOP_K="${TOP_K:-32}"
SEED="${SEED:-43}"
PROMPT_STYLE="${PROMPT_STYLE:-fake_think}"
PROGRESS_EVERY="${PROGRESS_EVERY:-5000}"
PREFILL_BACKEND="${PREFILL_BACKEND:-native}"
DECODE_BACKEND="${DECODE_BACKEND:-fast_token}"
RNG_MODE="${RNG_MODE:-global}"
VERIFY_WORKERS="${VERIFY_WORKERS:-4}"
TOKENIZER_DIR="${TOKENIZER_DIR:-${MODEL}}"

COMPRESSION_LIMIT="${COMPRESSION_LIMIT:-128}"
COMPRESSION_MAX_TOKENS_PER_TEXT="${COMPRESSION_MAX_TOKENS_PER_TEXT:-1024}"
COMPRESSION_REFERENCE_KIND="${COMPRESSION_REFERENCE_KIND:-hf}"
COMPRESSION_REFERENCE_HF_DIR="${COMPRESSION_REFERENCE_HF_DIR:-${MODEL}}"
COMPRESSION_REFERENCE_ALBATROSS_DIR="${COMPRESSION_REFERENCE_ALBATROSS_DIR:-}"
COMPRESSION_REFERENCE_ALBATROSS_MODEL="${COMPRESSION_REFERENCE_ALBATROSS_MODEL:-}"
COMPRESSION_CANDIDATE_HF_DIR="${COMPRESSION_CANDIDATE_HF_DIR:-${MODEL}}"
COMPRESSION_CANDIDATE_QUANTIZATION="${COMPRESSION_CANDIDATE_QUANTIZATION:-none}"

export RWKV7_NATIVE_MODEL="${RWKV7_NATIVE_MODEL:-1}"

rwkv7_print_env
rwkv7_log "final MATH500 acceptance model=${MODEL} dataset=${DATASET} out=${OUT_DIR} bsz_list=${BSZ_LIST}"

cmd=(
  "${PYTHON_BIN}" bench/run_math500_final_acceptance.py
  --hf-dir "${MODEL}"
  --dataset "${DATASET}"
  --out-dir "${OUT_DIR}"
  --tokenizer-dir "${TOKENIZER_DIR}"
  --bsz-list "${BSZ_LIST}"
  --sweep-limit "${SWEEP_LIMIT}"
  --sweep-rollout "${SWEEP_ROLLOUT}"
  --sweep-max-new-tokens "${SWEEP_MAX_NEW_TOKENS}"
  --full-limit "${FULL_LIMIT}"
  --full-rollout "${FULL_ROLLOUT}"
  --full-max-new-tokens "${FULL_MAX_NEW_TOKENS}"
  --ctx-limit "${CTX_LIMIT}"
  --temperature "${TEMPERATURE}"
  --top-p "${TOP_P}"
  --top-k "${TOP_K}"
  --seed "${SEED}"
  --prompt-style "${PROMPT_STYLE}"
  --dtype "${DTYPE}"
  --device "${DEVICE}"
  --progress-every "${PROGRESS_EVERY}"
  --prefill-backend "${PREFILL_BACKEND}"
  --decode-backend "${DECODE_BACKEND}"
  --rng-mode "${RNG_MODE}"
  --verify-workers "${VERIFY_WORKERS}"
  --compression-limit "${COMPRESSION_LIMIT}"
  --compression-max-tokens-per-text "${COMPRESSION_MAX_TOKENS_PER_TEXT}"
  --compression-reference-kind "${COMPRESSION_REFERENCE_KIND}"
  --compression-reference-hf-dir "${COMPRESSION_REFERENCE_HF_DIR}"
  --compression-candidate-hf-dir "${COMPRESSION_CANDIDATE_HF_DIR}"
  --compression-candidate-quantization "${COMPRESSION_CANDIDATE_QUANTIZATION}"
)

if [[ -n "${ALBATROSS_SUMMARY:-}" ]]; then
  cmd+=(--albatross-summary "${ALBATROSS_SUMMARY}")
fi
if [[ -n "${ALBATROSS_LOG:-}" ]]; then
  cmd+=(--albatross-log "${ALBATROSS_LOG}")
fi
if [[ "${COMPRESSION_REFERENCE_KIND}" == "albatross" ]]; then
  cmd+=(
    --compression-reference-albatross-dir "${COMPRESSION_REFERENCE_ALBATROSS_DIR}"
    --compression-reference-albatross-model "${COMPRESSION_REFERENCE_ALBATROSS_MODEL}"
  )
fi
if [[ "${SKIP_BSZ_SWEEP:-0}" == "1" ]]; then
  cmd+=(--skip-bsz-sweep --fixed-bsz "${FIXED_BSZ:-128}")
fi
if [[ "${SKIP_FULL:-0}" == "1" ]]; then
  cmd+=(--skip-full)
fi
if [[ "${SKIP_COMPRESSION:-0}" == "1" ]]; then
  cmd+=(--skip-compression)
fi
if [[ "${FAIL_ON_GATE:-1}" != "1" ]]; then
  cmd+=(--no-fail-on-gate)
fi

rwkv7_run "${cmd[@]}"
rwkv7_log "final MATH500 acceptance complete: ${OUT_DIR}/README.md"
