#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/home/ubuntu/hf-adapter}"
PYTHON_BIN="${PYTHON_BIN:-/data/venvs/rwkv3090/bin/python}"
OUT_DIR="${OUT_DIR:-/data/logs/3090/qwen35_full_matrix}"
RWKV_ROOT="${RWKV_ROOT:-/data/models}"
QWEN_ROOT="${QWEN_ROOT:-/home/ubuntu/models/qwen}"
QWEN9_MODEL="${QWEN9_MODEL:-/home/ubuntu/models/qwen/Qwen3.5-9B}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
WARMUP="${WARMUP:-1}"
RUNS="${RUNS:-1}"
QWEN_BACKEND="${QWEN_BACKEND:-auto}"
QWEN_FAST_ARGS=()
if [[ "${QWEN_BACKEND}" == "auto" ]]; then
  QWEN_FAST_ARGS+=(--require-qwen-fast-path)
fi

mkdir -p "${OUT_DIR}"
export CUDA_VISIBLE_DEVICES
export RWKV7_NATIVE_MODEL=0
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

cd "${ROOT}"
"${PYTHON_BIN}" bench/run_qwen35_speed_matrix.py \
  --pair "rwkv-1.5b__qwen3.5-2b=${RWKV_ROOT}/rwkv7-g1g-1.5b-hf::${QWEN_ROOT}/Qwen3.5-2B" \
  --pair "rwkv-2.9b__qwen3.5-4b=${RWKV_ROOT}/rwkv7-g1g-2.9b-hf::${QWEN_ROOT}/Qwen3.5-4B" \
  --pair "rwkv-7.2b__qwen3.5-9b=${RWKV_ROOT}/rwkv7-g1g-7.2b-hf::${QWEN9_MODEL}" \
  --prompt-tokens 128 512 2048 \
  --decode-tokens 128 512 \
  --batch-sizes 1 2 4 8 \
  --quantizations none bnb8 bnb4 \
  --benchmark-matrix qwen35_3090_hf \
  --dtype fp16 \
  --qwen-backend "${QWEN_BACKEND}" \
  "${QWEN_FAST_ARGS[@]}" \
  --warmup "${WARMUP}" \
  --runs "${RUNS}" \
  --results "${OUT_DIR}/results.jsonl" \
  --skip-existing
matrix_rc=$?

"${PYTHON_BIN}" bench/compare_qwen35_speed_matrix.py \
  --results "${OUT_DIR}/results.jsonl" \
  --expected-cells 216 \
  --min-prefill-speedup 1.05 \
  --min-decode-speedup 1.05 \
  --min-quant-prefill-speedup 1.00 \
  --min-quant-decode-speedup 1.00 \
  --json-output "${OUT_DIR}/summary.json" \
  --markdown-output "${OUT_DIR}/summary.md" \
  --fail-on-gate
compare_rc=$?

printf '%s\n' "${matrix_rc}" > "${OUT_DIR}/matrix_exit_code.txt"
printf '%s\n' "${compare_rc}" > "${OUT_DIR}/compare_exit_code.txt"
if [[ ${matrix_rc} -ne 0 ]]; then
  exit "${matrix_rc}"
fi
exit "${compare_rc}"
