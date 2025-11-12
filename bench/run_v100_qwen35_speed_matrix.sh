#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/home/data/wangyue/projects/rwkv7-hf-adapter-qwen35}"
PYTHON_BIN="${PYTHON_BIN:-/home/data/wangyue/envs/rwkv7/bin/python}"
OUT_DIR="${OUT_DIR:-/home/data/wangyue/bench/qwen35_v100_20260712/full_matrix}"
RWKV_ROOT="${RWKV_ROOT:-/home/data/wangyue/models/rwkv7}"
QWEN_ROOT="${QWEN_ROOT:-/home/data/wangyue/models/qwen}"
QWEN9_MODEL="${QWEN9_MODEL:-/home/wzu/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
WARMUP="${WARMUP:-1}"
RUNS="${RUNS:-1}"

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
  --dtype fp16 \
  --qwen-backend torch \
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
  --json-output "${OUT_DIR}/summary.json" \
  --markdown-output "${OUT_DIR}/summary.md"
compare_rc=$?

printf '%s\n' "${matrix_rc}" > "${OUT_DIR}/matrix_exit_code.txt"
printf '%s\n' "${compare_rc}" > "${OUT_DIR}/compare_exit_code.txt"
if [[ ${matrix_rc} -ne 0 ]]; then
  exit "${matrix_rc}"
fi
exit "${compare_rc}"
