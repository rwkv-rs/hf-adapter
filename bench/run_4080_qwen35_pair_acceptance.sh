#!/usr/bin/env bash
# Exact RTX 4080 B1/B8 RWKV-7 / Qwen3.5 acceptance.
set -uo pipefail

PAIR_LABEL="${PAIR_LABEL:-${1:-rwkv-1.5b__qwen3.5-2b}}"
RWKV_MODEL="${RWKV_MODEL:-${2:-}}"
QWEN_MODEL="${QWEN_MODEL:-${3:-}}"
OUT_DIR="${OUT_DIR:-${4:-}}"
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
REQUIRED_GPU_MODEL="4080"
WARMUP="${WARMUP:-3}"
RUNS="${RUNS:-7}"
TIMING_REPEATS="${TIMING_REPEATS:-7}"
BATCH_SIZE="${BATCH_SIZE:-8}"
DENSE_PREFILL_GATE="${DENSE_PREFILL_GATE:-1.00}"
QUANT_SPEED_GATE="${QUANT_SPEED_GATE:-1.00}"
QUANT_COSINE_GATE="${QUANT_COSINE_GATE:-0.999}"

if [[ -z "${RWKV_MODEL}" || -z "${QWEN_MODEL}" || -z "${OUT_DIR}" ]]; then
  echo "usage: $0 [PAIR_LABEL] RWKV_MODEL QWEN_MODEL OUT_DIR" >&2
  exit 2
fi
if [[ "${BATCH_SIZE}" != "1" && "${BATCH_SIZE}" != "8" ]]; then
  echo "BATCH_SIZE must be 1 or 8" >&2
  exit 2
fi
case "${PAIR_LABEL}" in
  rwkv-0.4b__qwen3.5-0.8b)
    rwkv_size="0.4b"
    qwen_size="0.8b"
    default_prefill_chunk_size="512"
    ;;
  rwkv-1.5b__qwen3.5-2b)
    rwkv_size="1.5b"
    qwen_size="2b"
    default_prefill_chunk_size="0"
    ;;
  rwkv-2.9b__qwen3.5-4b)
    rwkv_size="2.9b"
    qwen_size="4b"
    default_prefill_chunk_size="512"
    ;;
  *)
    echo "unsupported RTX 4080 pair: ${PAIR_LABEL}" >&2
    exit 2
    ;;
esac
if [[ "${BATCH_SIZE}" == "1" ]]; then
  default_active_work_gate="1.00"
else
  default_active_work_gate="1.75"
fi
DENSE_DECODE_GATE="${DENSE_DECODE_GATE:-1.00}"
ACTIVE_WORK_DECODE_GATE="${ACTIVE_WORK_DECODE_GATE:-${default_active_work_gate}}"
PREFILL_CHUNK_SIZE="${PREFILL_CHUNK_SIZE:-${default_prefill_chunk_size}}"
if [[ ! -d "${RWKV_MODEL}" || ! -d "${QWEN_MODEL}" ]]; then
  echo "RWKV_MODEL and QWEN_MODEL must be local model directories" >&2
  exit 2
fi

gpu_name="$(${PYTHON_BIN} - <<'PY'
import torch
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
PY
)"
if ! "${PYTHON_BIN}" "${ROOT}/bench/check_exact_gpu.py" \
  --model "${REQUIRED_GPU_MODEL}" --name "${gpu_name}"; then
  echo "acceptance requires exact desktop RTX ${REQUIRED_GPU_MODEL}; detected: ${gpu_name:-no CUDA GPU}" >&2
  exit 2
fi
runtime_error="$(${PYTHON_BIN} - <<'PY'
from importlib.metadata import PackageNotFoundError, version

import torch
import triton

expected = {
    "torch": "2.6.0+cu124",
    "triton": "3.2.0",
    "torchao": "0.16.0",
}
try:
    actual = {
        "torch": str(torch.__version__),
        "triton": str(triton.__version__),
        "torchao": version("torchao"),
    }
except PackageNotFoundError as exc:
    print(f"missing validated RTX 4080 dependency: {exc.name}")
    raise SystemExit(1)
mismatches = [
    f"{name}={actual[name]} (expected {expected[name]})"
    for name in expected
    if actual[name] != expected[name]
]
if mismatches:
    print("validated RTX 4080 runtime mismatch: " + ", ".join(mismatches))
    raise SystemExit(1)
PY
)"
runtime_rc=$?
if [[ ${runtime_rc} -ne 0 ]]; then
  echo "${runtime_error}" >&2
  echo "use the generic benchmark entrypoints for an unvalidated runtime experiment" >&2
  exit 2
fi

mkdir -p "${OUT_DIR}/logs"
rm -f "${OUT_DIR}"/{dense,memory,paired_quant}.jsonl \
  "${OUT_DIR}"/{summary.json,summary.md,matrix_failures.txt,pipeline_exit_code.txt}
rm -f "${OUT_DIR}/logs"/*.log

export CUDA_VISIBLE_DEVICES
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "${ROOT}"

failures=0
run_resident() {
  local label="$1" role="$2" kind="$3" model="$4" size="$5" quant="$6" output="$7"
  shift 7
  "${PYTHON_BIN}" bench/bench_cross_model_speed_resident.py \
    --model "${model}" --model-kind "${kind}" --model-role "${role}" \
    --model-pair "${PAIR_LABEL}" --model-size-label "${size}" \
    --benchmark-matrix qwen35_4080_hf_final --dtype fp16 --quantization "${quant}" \
    --device cuda --batch-sizes "${BATCH_SIZE}" --prompt-tokens 128 512 2048 \
    --decode-tokens 128 512 --prefill-chunk-size "${PREFILL_CHUNK_SIZE}" \
    --warmup "${WARMUP}" --runs "${RUNS}" --rwkv-attn-mode fused_recurrent \
    --rwkv-code-source repo "$@" --results "${output}" \
    > "${OUT_DIR}/logs/${label}.log" 2>&1
  local rc=$?
  [[ ${rc} -eq 0 ]] || failures=$((failures + 1))
  printf '%s rc=%s\n' "${label}" "${rc}"
}

export RWKV7_FAST_TOKEN_BACKEND=native_graph
export RWKV7_NATIVE_PREFILL_GRAPH=1
export RWKV7_NATIVE_GRAPH_FUSED_NORM_MIX=1
export RWKV7_NATIVE_GRAPH_FUSED_NORM_MIX_NUM_WARPS=4
export RWKV7_NATIVE_GRAPH_ADA_LINEAR=0
export RWKV7_NATIVE_GRAPH_ADA_SPARSE_FFN=0
run_resident dense_candidate candidate rwkv "${RWKV_MODEL}" "${rwkv_size}" none "${OUT_DIR}/dense.jsonl"

unset RWKV7_FAST_TOKEN_BACKEND RWKV7_NATIVE_PREFILL_GRAPH \
  RWKV7_NATIVE_GRAPH_FUSED_NORM_MIX RWKV7_NATIVE_GRAPH_FUSED_NORM_MIX_NUM_WARPS \
  RWKV7_NATIVE_GRAPH_ADA_LINEAR RWKV7_NATIVE_GRAPH_ADA_SPARSE_FFN
run_resident dense_reference reference qwen35 "${QWEN_MODEL}" "${qwen_size}" none \
  "${OUT_DIR}/dense.jsonl" --qwen-backend fla --qwen-conv-backend auto \
  --require-qwen-fast-path

# BnB lanes are full-model memory routes. Their acceptance gate is footprint
# plus functional execution; they are not used for quantized speed claims.
export RWKV7_FAST_TOKEN_BACKEND=native_graph
export RWKV7_NATIVE_PREFILL_GRAPH=1
export RWKV7_BNB_SKIP_POLICY=memory
export RWKV7_BNB_INT8_THRESHOLD=0
for quant in bnb8 bnb4; do
  run_resident "memory_${quant}" candidate rwkv "${RWKV_MODEL}" "${rwkv_size}" "${quant}" \
    "${OUT_DIR}/memory.jsonl"
done
unset RWKV7_BNB_SKIP_POLICY RWKV7_BNB_INT8_THRESHOLD

# The speed routes quantize the output head. Every shape measures dense and
# quantized execution in one process to remove cross-process clock bias. Match
# the RTX 3090/4090 contract by gating decode plus complete-cell latency while
# retaining prefill as explicit, non-gating telemetry.
export RWKV7_FAST_TOKEN_QUANT=1
export RWKV7_FAST_PREFILL_QUANT=1
export RWKV7_NATIVE_GRAPH_EXTERNAL_QUANT=1
export RWKV7_NATIVE_PREFILL_EXTERNAL_QUANT=1
export RWKV7_NATIVE_PREFILL_EXTERNAL_QUANT_GRAPH=1
for quant in a8w8 torchao_w4; do
  for prompt in 128 512 2048; do
    for decode in 128 512; do
      label="paired_${quant}_p${prompt}_d${decode}"
      "${PYTHON_BIN}" bench/bench_native_quant_e2e_decode.py \
        --hf-dir "${RWKV_MODEL}" --code-source repo --model-size-label "${rwkv_size}" \
        --dtype fp16 --device cuda --attn-mode fused_recurrent \
        --fast-token-backend native_graph --single-quantization "${quant}" \
        --policy speed --min-params 1 --group-size 128 \
        --batch-size "${BATCH_SIZE}" --prompt-tokens "${prompt}" --decode-tokens "${decode}" \
        --warmup "${WARMUP}" --timing-repeats "${TIMING_REPEATS}" \
        --paired-baseline --results "${OUT_DIR}/paired_quant.jsonl" \
        > "${OUT_DIR}/logs/${label}.log" 2>&1
      rc=$?
      [[ ${rc} -eq 0 ]] || failures=$((failures + 1))
      printf '%s rc=%s\n' "${label}" "${rc}"
    done
  done
done

"${PYTHON_BIN}" bench/summarize_4080_qwen35_acceptance.py "${OUT_DIR}" \
  --model-pair "${PAIR_LABEL}" \
  --batch-size "${BATCH_SIZE}" \
  --min-dense-prefill "${DENSE_PREFILL_GATE}" \
  --min-dense-decode "${DENSE_DECODE_GATE}" \
  --min-active-work-decode "${ACTIVE_WORK_DECODE_GATE}" \
  --min-quant-speed "${QUANT_SPEED_GATE}" \
  --min-quant-cosine "${QUANT_COSINE_GATE}" \
  --output "${OUT_DIR}/summary.json" --markdown-output "${OUT_DIR}/summary.md"
summary_rc=$?

printf '%s\n' "${failures}" > "${OUT_DIR}/matrix_failures.txt"
pipeline_rc=0
[[ ${failures} -eq 0 && ${summary_rc} -eq 0 ]] || pipeline_rc=1
printf '%s\n' "${pipeline_rc}" > "${OUT_DIR}/pipeline_exit_code.txt"
exit "${pipeline_rc}"
