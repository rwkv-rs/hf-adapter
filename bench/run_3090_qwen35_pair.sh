#!/usr/bin/env bash
# Run one 72-cell RWKV-7/Qwen3.5 pair on an RTX 3090-class single GPU.
set -uo pipefail

PAIR_LABEL="${PAIR_LABEL:-${1:-}}"
RWKV_MODEL="${RWKV_MODEL:-${2:-}}"
QWEN_MODEL="${QWEN_MODEL:-${3:-}}"
OUT_DIR="${OUT_DIR:-${4:-}}"
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
WARMUP="${WARMUP:-1}"
RUNS="${RUNS:-1}"
QWEN_BACKEND="${QWEN_BACKEND:-auto}"
QWEN_FAST_ARGS=()
if [[ "${QWEN_BACKEND}" == "auto" ]]; then
  QWEN_FAST_ARGS+=(--require-qwen-fast-path)
fi

if [[ -z "${PAIR_LABEL}" || -z "${RWKV_MODEL}" || -z "${QWEN_MODEL}" || -z "${OUT_DIR}" ]]; then
  echo "usage: $0 PAIR_LABEL RWKV_MODEL QWEN_MODEL OUT_DIR" >&2
  exit 2
fi
if [[ ! -d "${RWKV_MODEL}" || ! -d "${QWEN_MODEL}" ]]; then
  echo "both RWKV_MODEL and QWEN_MODEL must be local model directories" >&2
  exit 2
fi

mkdir -p "${OUT_DIR}"
export CUDA_VISIBLE_DEVICES
export RWKV7_NATIVE_MODEL=0
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT}"

"${PYTHON_BIN}" bench/run_qwen35_speed_matrix.py \
  --pair "${PAIR_LABEL}=${RWKV_MODEL}::${QWEN_MODEL}" \
  --prompt-tokens 128 512 2048 \
  --decode-tokens 128 512 \
  --batch-sizes 1 2 4 8 \
  --quantizations none bnb8 bnb4 \
  --benchmark-matrix qwen35_3090_hf \
  --dtype fp16 --qwen-backend "${QWEN_BACKEND}" \
  "${QWEN_FAST_ARGS[@]}" \
  --warmup "${WARMUP}" --runs "${RUNS}" \
  --results "${OUT_DIR}/results.jsonl" --skip-existing
matrix_rc=$?

"${PYTHON_BIN}" bench/compare_qwen35_speed_matrix.py \
  --results "${OUT_DIR}/results.jsonl" --expected-cells 72 \
  --min-prefill-speedup 1.05 --min-decode-speedup 1.05 \
  --json-output "${OUT_DIR}/summary.json" \
  --markdown-output "${OUT_DIR}/summary.md" --fail-on-gate
compare_rc=$?

printf '%s\n' "${matrix_rc}" > "${OUT_DIR}/matrix_exit_code.txt"
printf '%s\n' "${compare_rc}" > "${OUT_DIR}/compare_exit_code.txt"
[[ ${matrix_rc} -eq 0 ]] || exit "${matrix_rc}"
exit "${compare_rc}"
