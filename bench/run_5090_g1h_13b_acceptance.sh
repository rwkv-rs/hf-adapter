#!/usr/bin/env bash
# Validate the latest g1h 13.3B checkpoint at the RTX 5090 fit boundary.
set -uo pipefail

HF_MODEL="${HF_MODEL:-${1:-}}"
CHECKPOINT="${CHECKPOINT:-${2:-}}"
OUT_DIR="${OUT_DIR:-${3:-}}"
ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
REQUIRED_GPU_SUBSTRING="RTX 5090"

if [[ -z "${HF_MODEL}" || -z "${CHECKPOINT}" || -z "${OUT_DIR}" ]]; then
  echo "usage: $0 HF_MODEL CHECKPOINT OUT_DIR" >&2
  exit 2
fi
if [[ ! -d "${HF_MODEL}" || ! -f "${CHECKPOINT}" ]]; then
  echo "HF model directory and source checkpoint must exist" >&2
  exit 2
fi

gpu_name="$(${PYTHON_BIN} - <<'PY'
import torch
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
PY
)"
if [[ "${gpu_name}" != *"${REQUIRED_GPU_SUBSTRING}"* ]]; then
  echo "acceptance requires ${REQUIRED_GPU_SUBSTRING}; detected: ${gpu_name:-no CUDA GPU}" >&2
  exit 2
fi

mkdir -p "${OUT_DIR}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT}"

checkpoint_sha256="$(sha256sum "${CHECKPOINT}" | awk '{print $1}')"
checkpoint_size="$(stat -c '%s' "${CHECKPOINT}")"
printf '%s  %s\n' "${checkpoint_sha256}" "${CHECKPOINT}" > "${OUT_DIR}/checkpoint-sha256.txt"

"${PYTHON_BIN}" bench/bench_larger_model_smoke.py \
  --hf-dir "${HF_MODEL}" --model-size-label 13.3b \
  --checkpoint-path "${CHECKPOINT}" \
  --checkpoint-sha256 "${checkpoint_sha256}" \
  --checkpoint-size-bytes "${checkpoint_size}" \
  --dtype fp16 --device cuda --attn-mode fused_recurrent \
  --fast-token-backend native_jit --max-new-tokens 4 \
  --results "${OUT_DIR}/13p3_smoke.jsonl" \
  > "${OUT_DIR}/13p3_smoke.log" 2>&1
smoke_rc=$?
printf '%s\n' "${smoke_rc}" > "${OUT_DIR}/smoke-exit-code.txt"

"${PYTHON_BIN}" bench/run_blackwell_quant_matrix.py \
  --model "13.3b=${HF_MODEL}" \
  --prompt-tokens 128 --decode-tokens 128 --batch-sizes 8 \
  --quantizations none mm8 mm4 --min-params 8000000 --policy speed \
  --warmup 16 --timing-repeats 3 --dtype fp16 --device cuda \
  --fast-token-backend native_jit --attn-mode fused_recurrent \
  --paired-baseline --fail-fast \
  --results "${OUT_DIR}/quant_13p3_boundary.jsonl" \
  > "${OUT_DIR}/quant_13p3_boundary.log" 2>&1
quant_rc=$?
printf '%s\n' "${quant_rc}" > "${OUT_DIR}/quant-exit-code.txt"

"${PYTHON_BIN}" bench/summarize_blackwell_quant_matrix.py \
  "${OUT_DIR}/quant_13p3_boundary.jsonl" \
  --gate --expected-rows 3 --min-speed-ratio 0.98 \
  > "${OUT_DIR}/quant_13p3_summary.md" 2> "${OUT_DIR}/quant_13p3_summary.log"
gate_rc=$?
printf '%s\n' "${gate_rc}" > "${OUT_DIR}/gate-exit-code.txt"

pipeline_rc=0
[[ ${smoke_rc} -eq 0 && ${quant_rc} -eq 0 && ${gate_rc} -eq 0 ]] || pipeline_rc=1
printf '%s\n' "${pipeline_rc}" > "${OUT_DIR}/pipeline-exit-code.txt"
exit "${pipeline_rc}"
