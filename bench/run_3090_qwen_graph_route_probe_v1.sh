#!/usr/bin/env bash
# Select one whole-model Qwen StaticCache CUDA Graph route on an exact RTX 3090.
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OUT_DIR="${OUT_DIR:-${1:-}}"
MODEL="${MODEL:-${2:-}}"
MODEL_PAIR="${MODEL_PAIR:-${3:-}}"
MODEL_SIZE_LABEL="${MODEL_SIZE_LABEL:-${4:-}}"
REPOSITORY_COMMIT="${REPOSITORY_COMMIT:-}"
FLA_SOURCE_COMMIT="${FLA_SOURCE_COMMIT:-}"
CAUSAL_CONV1D_SOURCE_COMMIT="${CAUSAL_CONV1D_SOURCE_COMMIT:-}"

if [[ -z "${OUT_DIR}" || -z "${MODEL}" || -z "${MODEL_PAIR}" || -z "${MODEL_SIZE_LABEL}" ]]; then
  echo "usage: $0 OUT_DIR MODEL MODEL_PAIR MODEL_SIZE_LABEL" >&2
  exit 2
fi
if [[ ! "${REPOSITORY_COMMIT}" =~ ^[0-9a-fA-F]{40}$ ]]; then
  echo "REPOSITORY_COMMIT must be 40 hexadecimal characters" >&2
  exit 2
fi
if [[ -z "${FLA_SOURCE_COMMIT}" || -z "${CAUSAL_CONV1D_SOURCE_COMMIT}" ]]; then
  echo "FLA_SOURCE_COMMIT and CAUSAL_CONV1D_SOURCE_COMMIT are required" >&2
  exit 2
fi

ROOT="$(realpath -e -- "${ROOT}")"
MODEL="$(realpath -e -- "${MODEL}")"
OUT_DIR="$(realpath -m -- "${OUT_DIR}")"
if [[ "${PYTHON_BIN}" == */* ]]; then
  python_dir="$(realpath -e -- "$(dirname -- "${PYTHON_BIN}")")"
  PYTHON_BIN="${python_dir}/$(basename -- "${PYTHON_BIN}")"
else
  PYTHON_BIN="$(command -v -- "${PYTHON_BIN}")"
fi
[[ ! -e "${OUT_DIR}" ]] || { echo "refusing existing OUT_DIR ${OUT_DIR}" >&2; exit 2; }
[[ "$(git -C "${ROOT}" rev-parse HEAD)" == "${REPOSITORY_COMMIT}" ]] || {
  echo "repository commit mismatch" >&2; exit 2;
}
[[ -z "$(git -C "${ROOT}" status --porcelain --untracked-files=all)" ]] || {
  echo "route probes require a clean repository" >&2; exit 2;
}

mkdir -p "${OUT_DIR}/logs" "${OUT_DIR}/cache"
export PYTHONPATH="${ROOT}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_CUDA_ARCH_LIST=8.6
export CUDA_VISIBLE_DEVICES=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export REPOSITORY_COMMIT FLA_SOURCE_COMMIT CAUSAL_CONV1D_SOURCE_COMMIT

gpu_name="$(${PYTHON_BIN} -c 'import torch; print(torch.cuda.get_device_name(0))')"
"${PYTHON_BIN}" "${ROOT}/bench/check_exact_gpu.py" --model 3090 --name "${gpu_name}"

run_route() {
  local route="$1" result="${OUT_DIR}/${1}.jsonl" log="${OUT_DIR}/logs/${1}.log"
  local cache="${OUT_DIR}/cache/${1}"
  mkdir -p "${cache}/inductor" "${cache}/triton" "${cache}/xdg"
  : > "${result}"
  set +e
  XDG_CACHE_HOME="${cache}/xdg" TORCHINDUCTOR_CACHE_DIR="${cache}/inductor" \
    TRITON_CACHE_DIR="${cache}/triton" "${PYTHON_BIN}" \
    "${ROOT}/bench/bench_cross_model_speed_resident.py" \
      --model "${MODEL}" --model-kind qwen35 --model-role reference \
      --model-pair "${MODEL_PAIR}" --model-size-label "${MODEL_SIZE_LABEL}" \
      --benchmark-matrix qwen35_best_optimized_hf_v1 \
      --optimization-lane qwen_best_optimized_hf --dtype fp16 --quantization none \
      --device cuda --cells 1x128x128 8x128x128 1x2048x512 8x2048x512 \
      --prefill-chunk-size 512 --warmup 3 --runs 7 --qwen-backend fla \
      --qwen-conv-backend causal_conv1d --require-qwen-fast-path \
      --qwen-decode-optimization "${route}" --qwen-compile-mode max-autotune \
      --qwen-graph-probe-tokens 16 --fail-fast --results "${result}" \
      > "${log}" 2>&1
  printf '%s\n' "$?" > "${OUT_DIR}/${route}.exit_code.txt"
  set -e
}

run_route static_cache_inductor_cudagraph
run_route static_cache_raw_cudagraph

"${PYTHON_BIN}" - "${OUT_DIR}" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
short={(1,128,128),(8,128,128)}
boundary={(1,2048,512),(8,2048,512)}
for route in ("static_cache_inductor_cudagraph","static_cache_raw_cudagraph"):
 rows=[]
 for line in (root/f"{route}.jsonl").read_text(encoding="utf-8").splitlines():
  if line.strip(): rows.append(json.loads(line))
 for label,keys in (("short",short),("boundary",boundary)):
  with (root/f"{route}_{label}.jsonl").open("x",encoding="utf-8",newline="\n") as out:
   for row in rows:
    if (row.get("batch_size"),row.get("prompt_tokens"),row.get("decode_tokens")) in keys:
     out.write(json.dumps(row,separators=(",",":"))+"\n")
PY

"${PYTHON_BIN}" "${ROOT}/bench/select_qwen35_graph_route_v1.py" \
  --inductor-short "${OUT_DIR}/static_cache_inductor_cudagraph_short.jsonl" \
  --raw-short "${OUT_DIR}/static_cache_raw_cudagraph_short.jsonl" \
  --inductor-boundary "${OUT_DIR}/static_cache_inductor_cudagraph_boundary.jsonl" \
  --raw-boundary "${OUT_DIR}/static_cache_raw_cudagraph_boundary.jsonl" \
  --expected-device "NVIDIA GeForce RTX 3090" \
  --summary "${OUT_DIR}/route_selection.json"
