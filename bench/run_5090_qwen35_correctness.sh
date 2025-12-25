#!/usr/bin/env bash
# Validate the exact RTX 5090 Qwen full-FLA bridge and RWKV native prefill.
set -uo pipefail

PAIR_LABEL="${PAIR_LABEL:-${1:-}}"
RWKV_MODEL="${RWKV_MODEL:-${2:-}}"
QWEN_MODEL="${QWEN_MODEL:-${3:-}}"
OUT_DIR="${OUT_DIR:-${4:-}}"
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
REQUIRED_GPU_MODEL="5090"
BENCHMARK_MATRIX="${BENCHMARK_MATRIX:-qwen35_5090_hf_final}"
CORRECTNESS_PROMPT_TOKENS="${CORRECTNESS_PROMPT_TOKENS:-512}"
CORRECTNESS_BATCH_SIZE="${CORRECTNESS_BATCH_SIZE:-8}"
QWEN_CORRECTNESS_PROMPT_TOKENS="${QWEN_CORRECTNESS_PROMPT_TOKENS:-}"

if [[ -z "${PAIR_LABEL}" || -z "${RWKV_MODEL}" || -z "${QWEN_MODEL}" || -z "${OUT_DIR}" ]]; then
  echo "usage: $0 PAIR_LABEL RWKV_MODEL QWEN_MODEL OUT_DIR" >&2
  exit 2
fi
if [[ ! -d "${RWKV_MODEL}" || ! -d "${QWEN_MODEL}" ]]; then
  echo "both RWKV_MODEL and QWEN_MODEL must be local model directories" >&2
  exit 2
fi

gpu_name="$(${PYTHON_BIN} - <<'PY'
import torch
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
PY
)"
if ! "${PYTHON_BIN}" "${ROOT}/bench/check_exact_gpu.py" \
  --model "${REQUIRED_GPU_MODEL}" --name "${gpu_name}"; then
  echo "correctness requires exact desktop RTX ${REQUIRED_GPU_MODEL}; detected: ${gpu_name:-no CUDA GPU}" >&2
  exit 2
fi

mkdir -p "${OUT_DIR}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT}"

rwkv_size="${PAIR_LABEL#rwkv-}"
rwkv_size="${rwkv_size%%__*}"
qwen_size="${PAIR_LABEL##*qwen3.5-}"
if [[ -z "${QWEN_CORRECTNESS_PROMPT_TOKENS}" ]]; then
  if [[ "${qwen_size}" == "9b" ]]; then
    QWEN_CORRECTNESS_PROMPT_TOKENS=512
  else
    QWEN_CORRECTNESS_PROMPT_TOKENS=128
  fi
fi
if [[ ! "${QWEN_CORRECTNESS_PROMPT_TOKENS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "QWEN_CORRECTNESS_PROMPT_TOKENS must be a positive integer" >&2
  exit 2
fi
failures=0

run_checked() {
  local label="$1"
  shift
  "$@" > "${OUT_DIR}/${label}.log" 2>&1
  local rc=$?
  printf '%s\n' "${rc}" > "${OUT_DIR}/${label}-exit-code.txt"
  [[ ${rc} -eq 0 ]] || failures=$((failures + 1))
}

probe_common=(
  bench/bench_cross_model_speed.py
  --model "${QWEN_MODEL}" --model-kind qwen35 --model-role reference
  --model-pair "${PAIR_LABEL}" --model-size-label "${qwen_size}"
  --benchmark-matrix "${BENCHMARK_MATRIX}" --dtype fp16 --quantization none
  --device cuda --batch-size "${CORRECTNESS_BATCH_SIZE}" \
  --prompt-tokens "${QWEN_CORRECTNESS_PROMPT_TOKENS}" --decode-tokens 8
  --qwen-backend fla --warmup 1 --runs 1 --probe-tokens 8
)
run_checked full-fla-smoke "${PYTHON_BIN}" "${probe_common[@]}" \
  --qwen-conv-backend fla_triton --require-qwen-fast-path \
  --probe-output "${OUT_DIR}/full-fla-probe.pt" \
  --results "${OUT_DIR}/full-fla-smoke.jsonl"
run_checked transformers-conv-oracle "${PYTHON_BIN}" "${probe_common[@]}" \
  --qwen-conv-backend auto \
  --probe-output "${OUT_DIR}/transformers-conv-oracle.pt" \
  --results "${OUT_DIR}/transformers-conv-oracle.jsonl"
run_checked full-fla-correctness "${PYTHON_BIN}" bench/compare_qwen35_backend_probe.py \
  --fla-probe "${OUT_DIR}/full-fla-probe.pt" \
  --torch-probe "${OUT_DIR}/transformers-conv-oracle.pt" \
  --min-cosine 0.999 \
  --output "${OUT_DIR}/full-fla-vs-transformers-conv-oracle.json" \
  --fail-on-gate

rwkv_common=(
  bench/bench_cross_model_speed.py
  --model "${RWKV_MODEL}" --model-kind rwkv --model-role candidate
  --model-pair "${PAIR_LABEL}" --model-size-label "${rwkv_size}"
  --benchmark-matrix "${BENCHMARK_MATRIX}" --dtype fp16 --device cuda
  --batch-size "${CORRECTNESS_BATCH_SIZE}" \
  --prompt-tokens "${CORRECTNESS_PROMPT_TOKENS}" --decode-tokens 8
  --warmup 1 --runs 1 --rwkv-code-source repo --probe-tokens 8
)
for quantization in none bnb8 bnb4; do
  if [[ "${quantization}" == "bnb8" ]]; then
    export RWKV7_BNB_SKIP_POLICY=decode_rk
  else
    export RWKV7_BNB_SKIP_POLICY=memory
  fi
  export RWKV7_FAST_PREFILL=0 RWKV7_FAST_TOKEN_QUANT=0
  run_checked "rwkv-prefill-reference-${quantization}" \
    "${PYTHON_BIN}" "${rwkv_common[@]}" --quantization "${quantization}" \
    --probe-output "${OUT_DIR}/rwkv-prefill-reference-${quantization}.pt" \
    --results "${OUT_DIR}/rwkv-prefill-reference.jsonl"

  export RWKV7_FAST_PREFILL=1 RWKV7_FAST_TOKEN_QUANT=1
  export RWKV7_NATIVE_PREFILL_GRAPH=1
  export RWKV7_NATIVE_PREFILL_EXTERNAL_QUANT_GRAPH=1
  export RWKV7_NATIVE_PREFILL_FUSED_SCAN=1
  export RWKV7_NATIVE_PREFILL_FUSED_SHIFT_MIX=1
  export RWKV7_NATIVE_PREFILL_FUSED_STATE_PREP=1
  export RWKV7_NATIVE_PREFILL_FUSED_OUTPUT=1
  run_checked "rwkv-prefill-native-${quantization}" \
    "${PYTHON_BIN}" "${rwkv_common[@]}" --quantization "${quantization}" \
    --probe-output "${OUT_DIR}/rwkv-prefill-native-${quantization}.pt" \
    --results "${OUT_DIR}/rwkv-prefill-native.jsonl"

  run_checked "rwkv-prefill-compare-${quantization}" \
    "${PYTHON_BIN}" bench/compare_rwkv_prefill_probe.py \
    --reference-probe "${OUT_DIR}/rwkv-prefill-reference-${quantization}.pt" \
    --native-probe "${OUT_DIR}/rwkv-prefill-native-${quantization}.pt" \
    --min-cosine 0.9999 \
    --output "${OUT_DIR}/rwkv-prefill-correctness-${quantization}.json" \
    --fail-on-gate
done

printf '%s\n' "${failures}" > "${OUT_DIR}/correctness-failures.txt"
[[ ${failures} -eq 0 ]]
