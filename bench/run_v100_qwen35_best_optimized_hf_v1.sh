#!/usr/bin/env bash
# Run one Qwen3.5 checkpoint through the strict Tesla V100 best-HF reference lane.
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
QWEN_DECODE_OPTIMIZATION="${QWEN_DECODE_OPTIMIZATION:-static_cache_raw_cudagraph}"
QWEN_SDPA_POLICY="${QWEN_SDPA_POLICY:-auto}"
CACHE_ROOT="${CACHE_ROOT:-}"
CUDA_TOOLKIT_VIEW="${CUDA_TOOLKIT_VIEW:-}"
FLA_TARGET="${FLA_TARGET:-}"
TRITON_TARGET="${TRITON_TARGET:-}"

if [[ -z "${OUT_DIR}" || -z "${CACHE_ROOT}" || -z "${CUDA_TOOLKIT_VIEW}" || -z "${FLA_TARGET}" || -z "${TRITON_TARGET}" || -z "${MODEL}" || -z "${MODEL_PAIR}" || -z "${MODEL_SIZE_LABEL}" ]]; then
  echo "usage: CACHE_ROOT=/dedicated/cache CUDA_TOOLKIT_VIEW=/dedicated/cuda FLA_TARGET=/dedicated/fla TRITON_TARGET=/dedicated/triton $0 OUT_DIR MODEL MODEL_PAIR MODEL_SIZE_LABEL" >&2
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
if [[ "${QWEN_SDPA_POLICY}" != "auto" && "${QWEN_SDPA_POLICY}" != "math_only" ]]; then
  echo "QWEN_SDPA_POLICY must be auto or math_only" >&2
  exit 2
fi

ROOT="$(realpath -e -- "${ROOT}")"
OUT_DIR="$(realpath -m -- "${OUT_DIR}")"
MODEL="$(realpath -e -- "${MODEL}")"
CACHE_ROOT="$(realpath -m -- "${CACHE_ROOT}")"
CUDA_TOOLKIT_VIEW="$(realpath -e -- "${CUDA_TOOLKIT_VIEW}")"
FLA_TARGET="$(realpath -e -- "${FLA_TARGET}")"
TRITON_TARGET="$(realpath -e -- "${TRITON_TARGET}")"
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
  "CUDA_VISIBLE_DEVICES=0" "CUDA_DEVICE_ORDER=PCI_BUS_ID" "PYTHONPATH=${ROOT}:${TRITON_TARGET}:${FLA_TARGET}"
  "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" "TORCH_CUDA_ARCH_LIST=7.0"
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
env -i "${COMMON_ENV[@]}" "${PYTHON_BIN}" "${ROOT}/bench/check_exact_gpu.py" --exact-name "Tesla V100-PCIE-32GB" --name "${gpu_name}"

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
    "causal_conv1d": None,
}
expected = {
    "python": "3.11.15",
    "torch": "2.5.1+cu124",
    "torch_cuda": "12.4",
    "triton": "3.4.0",
    "transformers": "5.12.1",
    "fla": "0.5.1",
    "causal_conv1d": None,
}
if actual != expected:
    raise RuntimeError(f"strict Tesla V100 runtime mismatch: {actual!r} != {expected!r}")
PY

cd "${ROOT}"
env -i "${COMMON_ENV[@]}" "${PYTHON_BIN}" bench/bench_cross_model_speed_resident.py \
  --model "${MODEL}" \
  --model-kind qwen35 \
  --model-role reference \
  --model-pair "${MODEL_PAIR}" \
  --model-size-label "${MODEL_SIZE_LABEL}" \
  --benchmark-matrix qwen35_v100_best_optimized_hf_v1 \
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
  --qwen-conv-backend fla_triton \
  --qwen-sdpa-policy "${QWEN_SDPA_POLICY}" \
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
env -i "${COMMON_ENV[@]}" "${PYTHON_BIN}" - "${route_manifest}" "${result}" "${model_hashes}" "${model_hashes_after}" "${MODEL_PAIR}" "${MODEL_SIZE_LABEL}" "${MODEL}" "${QWEN_DECODE_OPTIMIZATION}" "${QWEN_COMPILE_MODE}" "${QWEN_SDPA_POLICY}" "${CACHE_ROOT}" "${REPOSITORY_COMMIT}" "${TRITON_TARGET}" "${FLA_TARGET}" <<'PY'
import hashlib, json, sys
from pathlib import Path

manifest, result, before, after = map(Path, sys.argv[1:5])
pair, size, model, route, compile_mode, sdpa_policy, cache_root, commit, triton_target, fla_target = sys.argv[5:15]

def artifact(path):
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

doc = {
    "schema_version": 1,
    "protocol": "qwen35_v100_best_optimized_hf_v1",
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
    "sdpa_policy": sdpa_policy,
    "compile_mode": compile_mode if route == "static_cache_inductor_cudagraph" else None,
    "forced_environment": {
        "CUDA_VISIBLE_DEVICES": "0",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "PYTHONPATH": f"{Path.cwd().resolve()}:{Path(triton_target).resolve()}:{Path(fla_target).resolve()}",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_CUDA_ARCH_LIST": "7.0",
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
