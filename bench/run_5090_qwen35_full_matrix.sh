#!/usr/bin/env bash
# Run the complete exact-card RTX 5090 B1/B8 Qwen3.5 acceptance matrix.
set -uo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_ROOT="${OUT_ROOT:-${1:-}}"

RWKV_0P4_MODEL="${RWKV_0P4_MODEL:-}"
RWKV_1P5_MODEL="${RWKV_1P5_MODEL:-}"
RWKV_2P9_MODEL="${RWKV_2P9_MODEL:-}"
RWKV_7P2_MODEL="${RWKV_7P2_MODEL:-}"
QWEN_0P8_MODEL="${QWEN_0P8_MODEL:-}"
QWEN_2_MODEL="${QWEN_2_MODEL:-}"
QWEN_4_MODEL="${QWEN_4_MODEL:-}"
QWEN_9_MODEL="${QWEN_9_MODEL:-}"
ACCEPTANCE_BATCH_SIZES="${ACCEPTANCE_BATCH_SIZES:-1 8}"

if [[ -z "${OUT_ROOT}" ]]; then
  echo "usage: OUT_ROOT=/path $0, with all RWKV_*_MODEL and QWEN_*_MODEL variables" >&2
  exit 2
fi

pairs=(
  "rwkv-0.4b__qwen3.5-0.8b|${RWKV_0P4_MODEL}|${QWEN_0P8_MODEL}|pair_0.4b_0.8b"
  "rwkv-1.5b__qwen3.5-2b|${RWKV_1P5_MODEL}|${QWEN_2_MODEL}|pair_1.5b_2b"
  "rwkv-2.9b__qwen3.5-4b|${RWKV_2P9_MODEL}|${QWEN_4_MODEL}|pair_2.9b_4b"
  "rwkv-7.2b__qwen3.5-9b|${RWKV_7P2_MODEL}|${QWEN_9_MODEL}|pair_7.2b_9b"
)

mkdir -p "${OUT_ROOT}"
failures=0
for batch_size in ${ACCEPTANCE_BATCH_SIZES}; do
  if [[ "${batch_size}" != "1" && "${batch_size}" != "8" ]]; then
    echo "unsupported acceptance batch size: ${batch_size}" >&2
    failures=$((failures + 1))
    continue
  fi
  for spec in "${pairs[@]}"; do
    IFS='|' read -r pair rwkv_model qwen_model out_name <<< "${spec}"
    if [[ ! -d "${rwkv_model}" || ! -d "${qwen_model}" ]]; then
      echo "missing local model for B${batch_size} ${pair}: ${rwkv_model} / ${qwen_model}" >&2
      failures=$((failures + 1))
      continue
    fi
    out_dir="${OUT_ROOT}/b${batch_size}/${out_name}"
    mkdir -p "${out_dir}"
    PYTHON_BIN="${PYTHON_BIN}" \
      CORRECTNESS_BATCH_SIZE="${batch_size}" \
      BENCHMARK_MATRIX="qwen35_5090_hf_b${batch_size}_final" \
      "${ROOT}/bench/run_5090_qwen35_correctness.sh" \
      "${pair}" "${rwkv_model}" "${qwen_model}" "${out_dir}"
    correctness_rc=$?
    printf '%s\n' "${correctness_rc}" > "${out_dir}/correctness-exit-code.txt"
    if [[ ${correctness_rc} -ne 0 ]]; then
      failures=$((failures + 1))
      continue
    fi
    PYTHON_BIN="${PYTHON_BIN}" \
      BATCH_SIZES="${batch_size}" \
      BENCHMARK_MATRIX="qwen35_5090_hf_b${batch_size}_final" \
      "${ROOT}/bench/run_5090_qwen35_pair_acceptance.sh" \
      "${pair}" "${rwkv_model}" "${qwen_model}" "${out_dir}"
    pair_rc=$?
    printf '%s\n' "${pair_rc}" > "${out_dir}/pair-exit-code.txt"
    [[ ${pair_rc} -eq 0 ]] || failures=$((failures + 1))
  done
done

printf '%s\n' "${failures}" > "${OUT_ROOT}/matrix-failures.txt"
"${PYTHON_BIN}" "${ROOT}/bench/summarize_5090_qwen35_acceptance.py" \
  "${OUT_ROOT}" --output "${OUT_ROOT}/final_summary.json" \
  > "${OUT_ROOT}/final_summary.log" 2>&1
summary_rc=$?
printf '%s\n' "${summary_rc}" > "${OUT_ROOT}/summary-exit-code.txt"
pipeline_rc=0
[[ ${failures} -eq 0 && ${summary_rc} -eq 0 ]] || pipeline_rc=1
printf '%s\n' "${pipeline_rc}" > "${OUT_ROOT}/pipeline-exit-code.txt"
exit "${pipeline_rc}"
