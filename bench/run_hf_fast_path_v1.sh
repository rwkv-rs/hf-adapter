#!/usr/bin/env bash
# Run one fail-closed 96-row RWKV/Qwen HF fast-path v1 card matrix.
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OUT_DIR="${OUT_DIR:-${1:-}}"
GPU_MODEL="${GPU_MODEL:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
RUNTIME_LOCK="${RUNTIME_LOCK:-}"
WRITE_RUNTIME_LOCK="${WRITE_RUNTIME_LOCK:-}"
FLA_SOURCE_COMMIT="${FLA_SOURCE_COMMIT:-}"
CAUSAL_CONV1D_SOURCE_COMMIT="${CAUSAL_CONV1D_SOURCE_COMMIT:-}"
BENCHMARK_MATRIX="hf_fast_path_v1"

required=(
  RWKV_04_MODEL RWKV_15_MODEL RWKV_29_MODEL RWKV_72_MODEL
  QWEN_08_MODEL QWEN_2_MODEL QWEN_4_MODEL QWEN_9_MODEL
)
if [[ -z "${OUT_DIR}" || ! "${GPU_MODEL}" =~ ^(3090|4090|5090)$ ]]; then
  echo "OUT_DIR and GPU_MODEL=3090|4090|5090 are required" >&2
  exit 2
fi
if [[ -n "${RUNTIME_LOCK}" && -n "${WRITE_RUNTIME_LOCK}" ]]; then
  echo "set exactly one of RUNTIME_LOCK or WRITE_RUNTIME_LOCK" >&2
  exit 2
fi
if [[ -z "${RUNTIME_LOCK}" && -z "${WRITE_RUNTIME_LOCK}" ]]; then
  echo "RUNTIME_LOCK is required; use WRITE_RUNTIME_LOCK only to establish the first-card lock" >&2
  exit 2
fi
if [[ -z "${FLA_SOURCE_COMMIT}" || -z "${CAUSAL_CONV1D_SOURCE_COMMIT}" ]]; then
  echo "FLA_SOURCE_COMMIT and CAUSAL_CONV1D_SOURCE_COMMIT are required" >&2
  exit 2
fi
for name in "${required[@]}"; do
  if [[ -z "${!name:-}" || ! -d "${!name}" ]]; then
    echo "${name} must name a local model directory" >&2
    exit 2
  fi
done

mkdir -p "${OUT_DIR}/logs"
rm -f \
  "${OUT_DIR}"/qwen_*.jsonl \
  "${OUT_DIR}"/rwkv_*.jsonl \
  "${OUT_DIR}/main_table.jsonl" \
  "${OUT_DIR}/validation.json" \
  "${OUT_DIR}/qwen_official_fast_path_status.json"

export CUDA_VISIBLE_DEVICES
export TORCH_CUDA_ARCH_LIST="8.6;8.9;12.0"
export FLA_SOURCE_COMMIT CAUSAL_CONV1D_SOURCE_COMMIT
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Remove inherited experiment flags before establishing the two protocol flags.
while IFS= read -r name; do
  unset "${name}"
done < <(compgen -e | grep '^RWKV7_' || true)

gpu_name="$(${PYTHON_BIN} - <<'PY'
import torch
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
PY
)"
"${PYTHON_BIN}" "${ROOT}/bench/check_exact_gpu.py" --model "${GPU_MODEL}" --name "${gpu_name}"

nvidia-smi \
  --query-gpu=name,compute_cap,driver_version,memory.total,power.limit,clocks.current.sm,clocks.max.sm \
  --format=csv > "${OUT_DIR}/system.csv"

environment_args=(
  --root "${ROOT}"
  --output "${OUT_DIR}/environment.json"
  --pip-freeze-output "${OUT_DIR}/pip-freeze.txt"
  --require-python 3.10
)
if [[ -n "${RUNTIME_LOCK}" ]]; then
  environment_args+=(--lock "${RUNTIME_LOCK}")
else
  environment_args+=(--write-lock "${WRITE_RUNTIME_LOCK}")
fi
"${PYTHON_BIN}" "${ROOT}/bench/capture_hf_fast_path_v1_environment.py" \
  "${environment_args[@]}" > "${OUT_DIR}/logs/environment.log" 2>&1

{
  for name in "${required[@]}"; do
    printf '[%s]\n' "${name}"
    find "${!name}" -maxdepth 1 -type f \
      \( -name 'config.json' -o -name '*.safetensors' \) \
      ! -name 'SHA256SUMS.safetensors' -print0 \
      | sort -z \
      | xargs -0 --no-run-if-empty sha256sum \
      | sed "s#${!name}/#<model>/#g"
  done
} > "${OUT_DIR}/model_hashes.sha256"

cd "${ROOT}"

run_qwen() {
  local model="$1" pair="$2" size="$3" output="$4" log="$5"
  "${PYTHON_BIN}" bench/bench_cross_model_speed_resident.py \
    --model "${model}" \
    --model-kind qwen35 \
    --model-role reference \
    --model-pair "${pair}" \
    --model-size-label "${size}" \
    --benchmark-matrix "${BENCHMARK_MATRIX}" \
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
    --fail-fast \
    --results "${output}" > "${log}" 2>&1
}

run_all_qwen() {
  run_qwen "${QWEN_08_MODEL}" rwkv-0.4b__qwen3.5-0.8b 0.8b \
    "${OUT_DIR}/qwen_0p8.jsonl" "${OUT_DIR}/logs/qwen_0p8.log" || return
  run_qwen "${QWEN_2_MODEL}" rwkv-1.5b__qwen3.5-2b 2b \
    "${OUT_DIR}/qwen_2b.jsonl" "${OUT_DIR}/logs/qwen_2b.log" || return
  run_qwen "${QWEN_4_MODEL}" rwkv-2.9b__qwen3.5-4b 4b \
    "${OUT_DIR}/qwen_4b.jsonl" "${OUT_DIR}/logs/qwen_4b.log" || return
  run_qwen "${QWEN_9_MODEL}" rwkv-7.2b__qwen3.5-9b 9b \
    "${OUT_DIR}/qwen_9b.jsonl" "${OUT_DIR}/logs/qwen_9b.log" || return
}

if ! run_all_qwen; then
  status_label="official HF FLA + causal_conv1d path failed"
  if [[ "${GPU_MODEL}" == "5090" ]]; then
    status_label="SM120 official HF fast path unverified"
  fi
  "${PYTHON_BIN}" - "${OUT_DIR}/qwen_official_fast_path_status.json" \
    "${GPU_MODEL}" "${gpu_name}" "${status_label}" <<'PY'
import json
import sys
from pathlib import Path

path, gpu_model, gpu_name, reason = sys.argv[1:]
Path(path).write_text(json.dumps({
    "status": "unverified",
    "protocol": "hf_fast_path_v1",
    "gpu_model": gpu_model,
    "device": gpu_name,
    "reason": reason,
    "main_table_eligible": False,
    "fallback_attempted": False,
}, indent=2) + "\n", encoding="utf-8")
PY
  echo "${status_label}; no unified main table was produced" >&2
  exit 3
fi

"${PYTHON_BIN}" - "${OUT_DIR}/qwen_official_fast_path_status.json" \
  "${GPU_MODEL}" "${gpu_name}" <<'PY'
import json
import sys
from pathlib import Path

path, gpu_model, gpu_name = sys.argv[1:]
Path(path).write_text(json.dumps({
    "status": "pass",
    "protocol": "hf_fast_path_v1",
    "gpu_model": gpu_model,
    "device": gpu_name,
    "main_table_eligible": True,
    "fallback_attempted": False,
}, indent=2) + "\n", encoding="utf-8")
PY

export RWKV7_FAST_TOKEN_BACKEND=native_jit
export RWKV7_NATIVE_PREFILL_GRAPH=0

run_rwkv() {
  local model="$1" pair="$2" size="$3" output="$4" log="$5"
  "${PYTHON_BIN}" bench/bench_cross_model_speed_resident.py \
    --model "${model}" \
    --model-kind rwkv \
    --model-role candidate \
    --model-pair "${pair}" \
    --model-size-label "${size}" \
    --benchmark-matrix "${BENCHMARK_MATRIX}" \
    --dtype fp16 \
    --quantization none \
    --device cuda \
    --batch-sizes 1 8 \
    --prompt-tokens 128 512 2048 \
    --decode-tokens 128 512 \
    --prefill-chunk-size 512 \
    --warmup 3 \
    --runs 7 \
    --rwkv-attn-mode fused_recurrent \
    --rwkv-code-source repo \
    --fail-fast \
    --results "${output}" > "${log}" 2>&1
}

run_rwkv "${RWKV_04_MODEL}" rwkv-0.4b__qwen3.5-0.8b 0.4b \
  "${OUT_DIR}/rwkv_0p4.jsonl" "${OUT_DIR}/logs/rwkv_0p4.log"
run_rwkv "${RWKV_15_MODEL}" rwkv-1.5b__qwen3.5-2b 1.5b \
  "${OUT_DIR}/rwkv_1p5.jsonl" "${OUT_DIR}/logs/rwkv_1p5.log"
run_rwkv "${RWKV_29_MODEL}" rwkv-2.9b__qwen3.5-4b 2.9b \
  "${OUT_DIR}/rwkv_2p9.jsonl" "${OUT_DIR}/logs/rwkv_2p9.log"
run_rwkv "${RWKV_72_MODEL}" rwkv-7.2b__qwen3.5-9b 7.2b \
  "${OUT_DIR}/rwkv_7p2.jsonl" "${OUT_DIR}/logs/rwkv_7p2.log"

"${PYTHON_BIN}" bench/validate_hf_fast_path_v1.py \
  --candidate-results \
    "${OUT_DIR}/rwkv_0p4.jsonl" "${OUT_DIR}/rwkv_1p5.jsonl" \
    "${OUT_DIR}/rwkv_2p9.jsonl" "${OUT_DIR}/rwkv_7p2.jsonl" \
  --reference-results \
    "${OUT_DIR}/qwen_0p8.jsonl" "${OUT_DIR}/qwen_2b.jsonl" \
    "${OUT_DIR}/qwen_4b.jsonl" "${OUT_DIR}/qwen_9b.jsonl" \
  --expected-device "${gpu_name}" \
  --summary "${OUT_DIR}/validation.json" \
  --main-table "${OUT_DIR}/main_table.jsonl"
