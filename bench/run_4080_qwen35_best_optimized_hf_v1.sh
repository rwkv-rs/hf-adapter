#!/usr/bin/env bash
# Run one Qwen3.5 checkpoint through the strict RTX 4080 best-HF reference lane.
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OUT_DIR="${OUT_DIR:-${1:-}}"
MODEL="${MODEL:-${2:-}}"
MODEL_PAIR="${MODEL_PAIR:-${3:-}}"
MODEL_SIZE_LABEL="${MODEL_SIZE_LABEL:-${4:-}}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RESULT_NAME="${RESULT_NAME:-qwen_${MODEL_SIZE_LABEL//./p}.jsonl}"
REPOSITORY_COMMIT="${REPOSITORY_COMMIT:-}"
QWEN_COMPILE_MODE="${QWEN_COMPILE_MODE:-max-autotune}"
QWEN_DECODE_OPTIMIZATION="${QWEN_DECODE_OPTIMIZATION:-static_cache_inductor_cudagraph}"
CACHE_ROOT="${CACHE_ROOT:-}"
CUDA_TOOLKIT_VIEW="${CUDA_TOOLKIT_VIEW:-}"

if [[ -z "${OUT_DIR}" || -z "${CACHE_ROOT}" || -z "${CUDA_TOOLKIT_VIEW}" || -z "${MODEL}" || -z "${MODEL_PAIR}" || -z "${MODEL_SIZE_LABEL}" ]]; then
  echo "usage: CACHE_ROOT=/dedicated/cache CUDA_TOOLKIT_VIEW=/dedicated/cuda $0 OUT_DIR MODEL MODEL_PAIR MODEL_SIZE_LABEL" >&2
  exit 2
fi
if [[ ! "${REPOSITORY_COMMIT}" =~ ^[0-9a-fA-F]{40}$ ]]; then
  echo "REPOSITORY_COMMIT must be the explicit 40-hex commit under test" >&2
  exit 2
fi
if [[ "${QWEN_DECODE_OPTIMIZATION}" != "module_call_dynamic" && "${QWEN_DECODE_OPTIMIZATION}" != "static_cache_inductor_cudagraph" && "${QWEN_DECODE_OPTIMIZATION}" != "static_cache_raw_cudagraph" ]]; then
  echo "QWEN_DECODE_OPTIMIZATION must be module_call_dynamic or a supported StaticCache CUDA Graph route" >&2
  exit 2
fi
if [[ "${QWEN_DECODE_OPTIMIZATION}" == "static_cache_inductor_cudagraph" && "${QWEN_COMPILE_MODE}" != "reduce-overhead" && "${QWEN_COMPILE_MODE}" != "max-autotune" ]]; then
  echo "QWEN_COMPILE_MODE must be reduce-overhead or max-autotune" >&2
  exit 2
fi

ROOT="$(realpath -e -- "${ROOT}")"
OUT_DIR="$(realpath -m -- "${OUT_DIR}")"
MODEL="$(realpath -e -- "${MODEL}")"
CACHE_ROOT="$(realpath -m -- "${CACHE_ROOT}")"
CUDA_TOOLKIT_VIEW="$(realpath -e -- "${CUDA_TOOLKIT_VIEW}")"
if [[ "${PYTHON_BIN}" == */* ]]; then
  python_dir="$(realpath -e -- "$(dirname -- "${PYTHON_BIN}")")"
  PYTHON_BIN="${python_dir}/$(basename -- "${PYTHON_BIN}")"
else
  PYTHON_BIN="$(command -v -- "${PYTHON_BIN}")"
fi
if [[ ! -x "${PYTHON_BIN}" || ! -d "${MODEL}" ]]; then
  echo "PYTHON_BIN and MODEL must resolve to an executable and a local directory" >&2
  exit 2
fi
case "${OUT_DIR}/" in
  "${ROOT}/"*|"${MODEL}/"*)
    echo "OUT_DIR must be outside the repository and model directory" >&2
    exit 2
    ;;
esac
case "${ROOT}/" in "${OUT_DIR}/"*) echo "OUT_DIR contains the repository" >&2; exit 2;; esac
case "${MODEL}/" in "${OUT_DIR}/"*) echo "OUT_DIR contains the model directory" >&2; exit 2;; esac
case "${CACHE_ROOT}/" in
  "${ROOT}/"*|"${MODEL}/"*)
    echo "CACHE_ROOT must be outside the repository and model directory" >&2
    exit 2
    ;;
esac
case "${ROOT}/" in "${CACHE_ROOT}/"*) echo "CACHE_ROOT contains the repository" >&2; exit 2;; esac
case "${MODEL}/" in "${CACHE_ROOT}/"*) echo "CACHE_ROOT contains the model directory" >&2; exit 2;; esac

validate_repository_provenance() {
  local actual repo_root
  repo_root="$(realpath -e -- "$(git -C "${ROOT}" rev-parse --show-toplevel)")"
  actual="$(git -C "${ROOT}" rev-parse HEAD)"
  if [[ "${repo_root}" != "${ROOT}" || "${actual,,}" != "${REPOSITORY_COMMIT,,}" ]]; then
    echo "repository provenance does not match ROOT/REPOSITORY_COMMIT" >&2
    exit 2
  fi
  if [[ -n "$(git -C "${ROOT}" status --porcelain --untracked-files=all)" ]]; then
    echo "strict reference capture requires a completely clean repository" >&2
    exit 2
  fi
}
validate_repository_provenance

result="${OUT_DIR}/${RESULT_NAME}"
log="${OUT_DIR}/logs/${RESULT_NAME%.jsonl}.log"
model_hashes="${OUT_DIR}/${RESULT_NAME%.jsonl}_model_hashes.sha256"
model_hashes_after="${OUT_DIR}/${RESULT_NAME%.jsonl}_model_hashes.after.sha256"
route_manifest="${OUT_DIR}/${RESULT_NAME%.jsonl}_route.json"
for path in "${result}" "${log}" "${model_hashes}" "${model_hashes_after}" "${route_manifest}"; do
  if [[ -e "${path}" ]]; then
    echo "refusing to overwrite existing artifact: ${path}" >&2
    exit 2
  fi
done
mkdir -p "${OUT_DIR}/logs"
if [[ -e "${CACHE_ROOT}" ]]; then
  [[ -d "${CACHE_ROOT}" && -z "$(find "${CACHE_ROOT}" -mindepth 1 -maxdepth 1 -print -quit)" ]] || {
    echo "CACHE_ROOT must be absent or empty: ${CACHE_ROOT}" >&2
    exit 2
  }
fi
mkdir -p "${CACHE_ROOT}/torchinductor" "${CACHE_ROOT}/triton" "${CACHE_ROOT}/huggingface"

COMMON_ENV=(
  "HOME=${HOME}" "LANG=C.UTF-8" "PATH=$(dirname "${PYTHON_BIN}"):${CUDA_TOOLKIT_VIEW}/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin"
  "CUDA_VISIBLE_DEVICES=0" "CUDA_DEVICE_ORDER=PCI_BUS_ID" "PYTHONPATH=${ROOT}"
  "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" "TORCH_CUDA_ARCH_LIST=8.9"
  "HF_HUB_OFFLINE=1" "TRANSFORMERS_OFFLINE=1" "TOKENIZERS_PARALLELISM=false"
  "CUDA_HOME=${CUDA_TOOLKIT_VIEW}" "LD_LIBRARY_PATH=${CUDA_TOOLKIT_VIEW}/lib64"
  "XDG_CACHE_HOME=${CACHE_ROOT}" "HF_HOME=${CACHE_ROOT}/huggingface"
  "TORCHINDUCTOR_CACHE_DIR=${CACHE_ROOT}/torchinductor" "TRITON_CACHE_DIR=${CACHE_ROOT}/triton"
  "REPOSITORY_COMMIT=${REPOSITORY_COMMIT}"
)

hash_model() {
  local output="$1"
  env -i "${COMMON_ENV[@]}" "${PYTHON_BIN}" - "${output}" "${MODEL}" <<'PY'
import hashlib, sys
from pathlib import Path
output, root = Path(sys.argv[1]), Path(sys.argv[2]).resolve(strict=True)
files = sorted((path for path in root.rglob("*") if path.is_file()), key=lambda path: path.relative_to(root).as_posix())
if not files:
    raise SystemExit(f"empty model directory: {root}")
lines = [f"[{root.as_posix()}]"]
for path in files:
    lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root).as_posix()}")
output.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
}
hash_model "${model_hashes}"

gpu_name="$(env -i "${COMMON_ENV[@]}" "${PYTHON_BIN}" - <<'PY'
import torch
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
PY
)"
env -i "${COMMON_ENV[@]}" "${PYTHON_BIN}" "${ROOT}/bench/check_exact_gpu.py" --model 4080 --name "${gpu_name}"

env -i "${COMMON_ENV[@]}" "${PYTHON_BIN}" - <<'PY'
import platform
from importlib.metadata import version

import torch
import transformers
import triton

actual = {
    "python": platform.python_version(),
    "torch": str(torch.__version__),
    "torch_cuda": str(torch.version.cuda),
    "triton": str(triton.__version__),
    "transformers": str(transformers.__version__),
    "fla": version("flash-linear-attention"),
    "causal_conv1d": version("causal-conv1d"),
}
expected = {
    "python": "3.12.2",
    "torch": "2.11.0+cu130",
    "torch_cuda": "13.0",
    "triton": "3.6.0",
    "transformers": "5.12.1",
    "fla": "0.5.1",
    "causal_conv1d": "1.6.2.post1",
}
if actual != expected:
    raise RuntimeError(f"strict RTX 4080 runtime mismatch: {actual!r} != {expected!r}")
PY

cd "${ROOT}"
env -i "${COMMON_ENV[@]}" "${PYTHON_BIN}" bench/bench_cross_model_speed_resident.py \
  --model "${MODEL}" \
  --model-kind qwen35 \
  --model-role reference \
  --model-pair "${MODEL_PAIR}" \
  --model-size-label "${MODEL_SIZE_LABEL}" \
  --benchmark-matrix qwen35_best_optimized_hf_v1 \
  --optimization-lane qwen_best_optimized_hf \
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

hash_model "${model_hashes_after}"
cmp --silent "${model_hashes}" "${model_hashes_after}" || {
  echo "Qwen model inputs changed during formal capture" >&2
  exit 2
}
validate_repository_provenance
env -i "${COMMON_ENV[@]}" "${PYTHON_BIN}" - "${route_manifest}" "${result}" "${model_hashes}" "${model_hashes_after}" "${MODEL_PAIR}" "${MODEL_SIZE_LABEL}" "${MODEL}" "${QWEN_DECODE_OPTIMIZATION}" "${QWEN_COMPILE_MODE}" "${CACHE_ROOT}" "${REPOSITORY_COMMIT}" <<'PY'
import hashlib, json, sys
from pathlib import Path

manifest, result, before, after = map(Path, sys.argv[1:5])
pair, size, model, route, compile_mode, cache_root, commit = sys.argv[5:12]

def artifact(path):
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

doc = {
    "schema_version": 1,
    "protocol": "qwen35_best_optimized_hf_4080_v1",
    "benchmark_repository_commit": commit,
    "repository_clean_pre_and_post": True,
    "model_pair": pair,
    "model_size_label": size,
    "model_path": str(Path(model).resolve()),
    "result": artifact(result),
    "model_hash_contract": {
        "algorithm": "sha256",
        "scope": "every recursive regular file",
        "before": artifact(before),
        "after": artifact(after),
        "byte_identical": before.read_bytes() == after.read_bytes(),
    },
    "decode_route": route,
    "compile_mode": compile_mode if route == "static_cache_inductor_cudagraph" else None,
    "forced_environment": {
        "CUDA_VISIBLE_DEVICES": "0",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "PYTHONPATH": str(Path.cwd().resolve()),
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_CUDA_ARCH_LIST": "8.9",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "CACHE_ROOT": str(Path(cache_root).resolve()),
    },
}
if not doc["model_hash_contract"]["byte_identical"]:
    raise SystemExit("Qwen model hash snapshots changed")
manifest.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
PY
echo "wrote ${result}"
