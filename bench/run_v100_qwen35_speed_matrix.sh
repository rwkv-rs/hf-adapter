#!/usr/bin/env bash
set -uo pipefail

ROOT="${ROOT:-/home/data/wangyue/projects/rwkv7-hf-adapter-qwen35}"
PYTHON_BIN="${PYTHON_BIN:-/home/data/wangyue/envs/rwkv7/bin/python}"
OUT_DIR="${OUT_DIR:-/home/data/wangyue/bench/qwen35_v100_fla_20260713/full_matrix}"
RWKV_ROOT="${RWKV_ROOT:-/home/data/wangyue/models/rwkv7}"
QWEN_ROOT="${QWEN_ROOT:-/home/data/wangyue/models/qwen}"
FLA_ROOT="${FLA_ROOT:-/home/data/wangyue/projects/flash-linear-attention}"
QWEN2_MODEL="${QWEN2_MODEL:-${QWEN_ROOT}/Qwen3.5-2B}"
QWEN9_MODEL="${QWEN9_MODEL:-/home/wzu/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
WARMUP="${WARMUP:-1}"
RUNS="${RUNS:-1}"

mkdir -p "${OUT_DIR}"
export CUDA_VISIBLE_DEVICES
export RWKV7_NATIVE_MODEL=0
# FlashQLA only supports newer SM90/SM100 GPUs. V100 must use FLA's generic
# Triton Gated DeltaNet kernels and card-local compilation/autotune.
export FLA_FLASH_QLA=0
export PYTHONPATH="${FLA_ROOT}:${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

cd "${ROOT}"
# Fail before the 432-row run unless V100 can bind and execute the required
# Qwen FLA core contract. causal-conv1d remains separately reported. This also
# pays the first Triton compilation cost outside the measured matrix.
"${PYTHON_BIN}" bench/bench_cross_model_speed.py \
  --model "${QWEN2_MODEL}" \
  --model-kind qwen35 \
  --model-role reference \
  --model-pair "fla-preflight__qwen3.5-2b" \
  --model-size-label 2b \
  --dtype fp16 \
  --quantization none \
  --device cuda \
  --batch-size 1 \
  --prompt-tokens 128 \
  --decode-tokens 8 \
  --warmup 1 \
  --runs 1 \
  --qwen-backend fla \
  --results "${OUT_DIR}/fla_smoke.jsonl"
preflight_rc=$?
printf '%s\n' "${preflight_rc}" > "${OUT_DIR}/fla_smoke_exit_code.txt"
if [[ ${preflight_rc} -ne 0 ]]; then
  exit "${preflight_rc}"
fi

"${PYTHON_BIN}" bench/run_qwen35_speed_matrix.py \
  --pair "rwkv-1.5b__qwen3.5-2b=${RWKV_ROOT}/rwkv7-g1g-1.5b-hf::${QWEN2_MODEL}" \
  --pair "rwkv-2.9b__qwen3.5-4b=${RWKV_ROOT}/rwkv7-g1g-2.9b-hf::${QWEN_ROOT}/Qwen3.5-4B" \
  --pair "rwkv-7.2b__qwen3.5-9b=${RWKV_ROOT}/rwkv7-g1g-7.2b-hf::${QWEN9_MODEL}" \
  --prompt-tokens 128 512 2048 \
  --decode-tokens 128 512 \
  --batch-sizes 1 2 4 8 \
  --quantizations none bnb8 bnb4 \
  --benchmark-matrix qwen35_v100_hf \
  --dtype fp16 \
  --qwen-backend fla \
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
  --required-reference-backend fla \
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
