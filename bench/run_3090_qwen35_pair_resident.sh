#!/usr/bin/env bash
# Single-load 72-cell RWKV-7/Qwen3.5 acceptance runner for a 24 GiB RTX 3090.
set -uo pipefail

PAIR_LABEL="${PAIR_LABEL:-${1:-}}"
RWKV_MODEL="${RWKV_MODEL:-${2:-}}"
QWEN_MODEL="${QWEN_MODEL:-${3:-}}"
OUT_DIR="${OUT_DIR:-${4:-}}"
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
WARMUP="${WARMUP:-1}"
RUNS="${RUNS:-3}"
PREFILL_CHUNK_SIZE="${PREFILL_CHUNK_SIZE:-0}"

if [[ -z "${PAIR_LABEL}" || -z "${RWKV_MODEL}" || -z "${QWEN_MODEL}" || -z "${OUT_DIR}" ]]; then
  echo "usage: $0 PAIR_LABEL RWKV_MODEL QWEN_MODEL OUT_DIR" >&2
  exit 2
fi
if [[ ! -d "${RWKV_MODEL}" || ! -d "${QWEN_MODEL}" ]]; then
  echo "both RWKV_MODEL and QWEN_MODEL must be local model directories" >&2
  exit 2
fi

mkdir -p "${OUT_DIR}"
rm -f "${OUT_DIR}/results.jsonl" "${OUT_DIR}/progress.log"
export CUDA_VISIBLE_DEVICES
export RWKV7_NATIVE_MODEL=0
export RWKV7_FAST_TOKEN_BACKEND=native_graph
export RWKV7_BNB_INT8_THRESHOLD="${RWKV7_BNB_INT8_THRESHOLD:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT}"

failures=0
for quantization in none bnb8 bnb4; do
  for role in candidate reference; do
    if [[ "${role}" == candidate ]]; then
      model="${RWKV_MODEL}"
      kind=rwkv
      size_label="${PAIR_LABEL#rwkv-}"
      size_label="${size_label%%__*}"
      fast_args=()
    else
      model="${QWEN_MODEL}"
      kind=qwen35
      size_label="${PAIR_LABEL##*qwen3.5-}"
      fast_args=(--require-qwen-fast-path)
    fi
    printf 'START %s %s\n' "${quantization}" "${role}" | tee -a "${OUT_DIR}/progress.log"
    "${PYTHON_BIN}" bench/bench_cross_model_speed_resident.py \
      --model "${model}" --model-kind "${kind}" --model-role "${role}" \
      --model-pair "${PAIR_LABEL}" --model-size-label "${size_label}" \
      --benchmark-matrix qwen35_3090_hf --dtype fp16 --quantization "${quantization}" \
      --device cuda --batch-sizes 1 2 4 8 --prompt-tokens 128 512 2048 \
      --decode-tokens 128 512 --prefill-chunk-size "${PREFILL_CHUNK_SIZE}" \
      --warmup "${WARMUP}" --runs "${RUNS}" --rwkv-attn-mode fused_recurrent \
      --rwkv-code-source repo --qwen-backend auto "${fast_args[@]}" \
      --results "${OUT_DIR}/results.jsonl" \
      > "${OUT_DIR}/${quantization}_${role}.log" 2>&1
    rc=$?
    printf 'DONE %s %s rc=%s\n' "${quantization}" "${role}" "${rc}" | tee -a "${OUT_DIR}/progress.log"
    [[ ${rc} -eq 0 ]] || failures=$((failures + 1))
  done
done

"${PYTHON_BIN}" bench/compare_qwen35_speed_matrix.py \
  --results "${OUT_DIR}/results.jsonl" --expected-cells 72 \
  --min-prefill-speedup 1.05 --min-decode-speedup 1.05 \
  --min-quant-prefill-speedup 1.00 --min-quant-decode-speedup 1.00 \
  --require-native-candidate --require-qwen-fast-path \
  --require-quant-memory-reduction --require-prefill-mode-match \
  --require-quant-not-slower-than-dense \
  --json-output "${OUT_DIR}/summary.json" \
  --markdown-output "${OUT_DIR}/summary.md" --fail-on-gate
compare_rc=$?

printf '%s\n' "${failures}" > "${OUT_DIR}/matrix_failures.txt"
printf '%s\n' "${compare_rc}" > "${OUT_DIR}/compare_exit_code.txt"
[[ ${failures} -eq 0 ]] || exit 1
exit "${compare_rc}"
