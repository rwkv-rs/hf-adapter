#!/usr/bin/env bash
# Run one Qwen3.5 checkpoint through the fail-closed 5090 best-HF reference lane.
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_DIR="${OUT_DIR:-${1:-}}"
MODEL="${MODEL:-${2:-}}"
MODEL_PAIR="${MODEL_PAIR:-${3:-}}"
MODEL_SIZE_LABEL="${MODEL_SIZE_LABEL:-${4:-}}"
RESULT_NAME="${RESULT_NAME:-qwen_${MODEL_SIZE_LABEL//./p}.jsonl}"
BENCHMARK_MATRIX="qwen35_best_optimized_hf_v1"
OPTIMIZATION_LANE="qwen_best_optimized_hf"
FLA_SOURCE_COMMIT="${FLA_SOURCE_COMMIT:-}"
CAUSAL_CONV1D_SOURCE_COMMIT="${CAUSAL_CONV1D_SOURCE_COMMIT:-}"
QWEN_COMPILE_MODE="${QWEN_COMPILE_MODE:-max-autotune}"
QWEN_DECODE_OPTIMIZATION="${QWEN_DECODE_OPTIMIZATION:-static_cache_inductor_cudagraph}"
REPOSITORY_COMMIT="${REPOSITORY_COMMIT:-}"
EXPECTED_GPU_MODEL="${EXPECTED_GPU_MODEL:-5090}"

if [[ -z "${OUT_DIR}" || -z "${MODEL}" || -z "${MODEL_PAIR}" || -z "${MODEL_SIZE_LABEL}" ]]; then
  echo "usage: $0 OUT_DIR MODEL MODEL_PAIR MODEL_SIZE_LABEL" >&2
  exit 2
fi
if [[ ! -d "${MODEL}" ]]; then
  echo "MODEL must name a local Qwen3.5 directory: ${MODEL}" >&2
  exit 2
fi
if [[ -z "${FLA_SOURCE_COMMIT}" || -z "${CAUSAL_CONV1D_SOURCE_COMMIT}" || -z "${REPOSITORY_COMMIT}" ]]; then
  echo "FLA_SOURCE_COMMIT, CAUSAL_CONV1D_SOURCE_COMMIT and REPOSITORY_COMMIT are required" >&2
  exit 2
fi
if [[ "${QWEN_DECODE_OPTIMIZATION}" != "static_cache_inductor_cudagraph" && "${QWEN_DECODE_OPTIMIZATION}" != "static_cache_raw_cudagraph" ]]; then
  echo "QWEN_DECODE_OPTIMIZATION must be a supported StaticCache CUDA Graph route" >&2
  exit 2
fi
if [[ "${QWEN_DECODE_OPTIMIZATION}" == "static_cache_inductor_cudagraph" && "${QWEN_COMPILE_MODE}" != "reduce-overhead" && "${QWEN_COMPILE_MODE}" != "max-autotune" ]]; then
  echo "QWEN_COMPILE_MODE must be reduce-overhead or max-autotune" >&2
  exit 2
fi

mkdir -p "${OUT_DIR}/logs"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
export FLA_SOURCE_COMMIT CAUSAL_CONV1D_SOURCE_COMMIT REPOSITORY_COMMIT

gpu_name="$(${PYTHON_BIN} - <<'PY'
import torch
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
PY
)"
"${PYTHON_BIN}" "${ROOT}/bench/check_exact_gpu.py" --model "${EXPECTED_GPU_MODEL}" --name "${gpu_name}"

result="${OUT_DIR}/${RESULT_NAME}"
log="${OUT_DIR}/logs/${RESULT_NAME%.jsonl}.log"
rm -f "${result}"

cd "${ROOT}"
"${PYTHON_BIN}" bench/bench_cross_model_speed_resident.py \
  --model "${MODEL}" \
  --model-kind qwen35 \
  --model-role reference \
  --model-pair "${MODEL_PAIR}" \
  --model-size-label "${MODEL_SIZE_LABEL}" \
  --benchmark-matrix "${BENCHMARK_MATRIX}" \
  --optimization-lane "${OPTIMIZATION_LANE}" \
  --dtype fp16 \
  --quantization none \
  --device cuda \
  --batch-sizes 1 8 \
  --prompt-tokens 128 512 2048 \
  --decode-tokens 128 512 \
  --prefill-chunk-size 512 \
  --warmup 3 \
  --runs 7 \
  --qwen-backend fla \
  --qwen-conv-backend causal_conv1d \
  --require-qwen-fast-path \
  --qwen-decode-optimization "${QWEN_DECODE_OPTIMIZATION}" \
  --qwen-compile-mode "${QWEN_COMPILE_MODE}" \
  --qwen-graph-probe-tokens 16 \
  --fail-fast \
  --results "${result}" > "${log}" 2>&1

echo "wrote ${result}"
