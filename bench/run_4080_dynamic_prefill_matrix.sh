#!/usr/bin/env bash
# Validate the RTX 4080 shape-safe dynamic prefill envelope.
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="${MODEL:-}"
OUT_DIR="${OUT_DIR:-${1:-}}"
MODE="${MODE:-smoke}"
BATCHES="${BATCHES:-1,2,3,4,5,6,7,8}"

case "${MODE}" in
  smoke) PROMPTS="${PROMPTS:-127,128,129,511,512,513}"; WARMUP="${WARMUP:-1}"; STEPS="${STEPS:-3}" ;;
  full) PROMPTS="${PROMPTS:-31,32,33,63,64,65,127,128,129,255,256,257,511,512,513,1023,1024,1025,2047,2048,2049}"; WARMUP="${WARMUP:-1}"; STEPS="${STEPS:-3}" ;;
  *) echo "MODE must be smoke or full" >&2; exit 2 ;;
esac

if [[ -z "${MODEL}" || -z "${OUT_DIR}" ]]; then
  echo "MODEL and OUT_DIR are required" >&2
  exit 2
fi
ROOT="$(realpath -e -- "${ROOT}")"
MODEL="$(realpath -e -- "${MODEL}")"
OUT_DIR="$(realpath -m -- "${OUT_DIR}")"
if [[ -e "${OUT_DIR}" ]]; then
  echo "OUT_DIR must not already exist: ${OUT_DIR}" >&2
  exit 2
fi
if [[ "${PYTHON_BIN}" == */* ]]; then
  python_dir="$(realpath -e -- "$(dirname -- "${PYTHON_BIN}")")"
  PYTHON_BIN="${python_dir}/$(basename -- "${PYTHON_BIN}")"
else
  PYTHON_BIN="$(command -v -- "${PYTHON_BIN}")"
fi
[[ -f "${MODEL}/config.json" ]] || { echo "missing ${MODEL}/config.json" >&2; exit 2; }
compgen -G "${MODEL}/*.safetensors" >/dev/null || { echo "missing weights in ${MODEL}" >&2; exit 2; }

mkdir -p "${OUT_DIR}"
RESULTS="${OUT_DIR}/dynamic_prefill.jsonl"
SUMMARY="${OUT_DIR}/summary.json"
SYSTEM="${OUT_DIR}/system.json"
LOG="${OUT_DIR}/run.log"

export PYTHONPATH="${ROOT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false

gpu_name="$(${PYTHON_BIN} -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")')"
"${PYTHON_BIN}" "${ROOT}/bench/validators/check_exact_gpu.py" --model 4080 --name "${gpu_name}"

"${PYTHON_BIN}" - "${SYSTEM}" "${MODE}" "${BATCHES}" "${PROMPTS}" <<'PY'
import json, platform, sys
import torch, transformers
try:
    import triton
    triton_version = str(triton.__version__)
except Exception:
    triton_version = None
doc = {
    "schema_version": 1,
    "mode": sys.argv[2],
    "batches": sys.argv[3],
    "prompts": sys.argv[4],
    "python": platform.python_version(),
    "torch": str(torch.__version__),
    "torch_cuda": str(torch.version.cuda),
    "transformers": str(transformers.__version__),
    "triton": triton_version,
    "device": torch.cuda.get_device_name(0),
    "compute_capability": list(torch.cuda.get_device_capability(0)),
}
open(sys.argv[1], "w", encoding="utf-8").write(json.dumps(doc, indent=2) + "\n")
PY

set -o pipefail
"${PYTHON_BIN}" "${ROOT}/bench/runners/bench_native_prefill_scan.py" \
  --model "${MODEL}" --code-source repo --device cuda --dtype fp16 \
  --reference-backend native-direct --fused-scan auto \
  --batch-sizes "${BATCHES}" --prompt-tokens "${PROMPTS}" \
  --warmup "${WARMUP}" --steps "${STEPS}" --timing cuda-event \
  --min-cosine 0.999 --results "${RESULTS}" 2>&1 | tee "${LOG}"

"${PYTHON_BIN}" "${ROOT}/bench/validators/check_dynamic_prefill_matrix.py" \
  --results "${RESULTS}" --batches "${BATCHES}" --prompts "${PROMPTS}" \
  --require-safe-fusions \
  --max-padding-latency-ratio 1.5 --max-boundary-throughput-ratio 1.35 \
  --max-cross-route-boundary-ratio 3.0 \
  --summary "${SUMMARY}"

echo "PASS: ${SUMMARY}"
