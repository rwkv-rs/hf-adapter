#!/usr/bin/env bash
# Reproduce and gate the latest g1i 7.2B/Qwen3.5-9B RTX 4090 matrix.
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OUT_DIR="${OUT_DIR:-${1:-}}"
RWKV_PYTHON_BIN="${RWKV_PYTHON_BIN:-python}"
QWEN_PYTHON_BIN="${QWEN_PYTHON_BIN:-python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
WARMUP="${WARMUP:-3}"
RUNS="${RUNS:-7}"

if [[ -z "${OUT_DIR}" ]]; then
  echo "usage: OUT_DIR=... RWKV_72_MODEL=... QWEN_9_MODEL=... $0" >&2
  exit 2
fi
for name in RWKV_72_MODEL QWEN_9_MODEL; do
  if [[ -z "${!name:-}" || ! -d "${!name}" ]]; then
    echo "${name} must name a local model directory" >&2
    exit 2
  fi
done

export CUDA_VISIBLE_DEVICES
for python_bin in "${RWKV_PYTHON_BIN}" "${QWEN_PYTHON_BIN}"; do
  gpu_name="$(${python_bin} - <<'PY'
import torch
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
PY
)"
  "${python_bin}" "${ROOT}/bench/check_exact_gpu.py" --model 4090 --name "${gpu_name}"
done

"${RWKV_PYTHON_BIN}" - <<'PY'
import torch, transformers, triton
actual = (
    str(torch.__version__), str(torch.version.cuda),
    str(triton.__version__), str(transformers.__version__),
)
expected = ("2.7.1+cu126", "12.6", "3.3.1", "5.12.1")
assert actual == expected, f"RWKV runtime {actual} != validated {expected}"
assert hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation")
PY
"${QWEN_PYTHON_BIN}" - <<'PY'
from importlib.metadata import version
import torch, transformers, triton
actual = (
    str(torch.__version__), str(torch.version.cuda),
    str(triton.__version__), str(transformers.__version__),
    version("flash-linear-attention"), version("fla-core"), version("einops"),
)
expected = (
    "2.6.0+cu124", "12.4", "3.2.0", "5.12.1",
    "0.5.1", "0.5.1", "0.8.2",
)
assert actual == expected, f"Qwen runtime {actual} != validated {expected}"
PY

mkdir -p "${OUT_DIR}/logs"
rm -f \
  "${OUT_DIR}/accumulation_ab.jsonl" \
  "${OUT_DIR}/candidate.jsonl" \
  "${OUT_DIR}/qwen_reference.jsonl" \
  "${OUT_DIR}/summary.json" \
  "${OUT_DIR}/summary.md" \
  "${OUT_DIR}/logs/accumulation_ab.log" \
  "${OUT_DIR}/logs/candidate.log" \
  "${OUT_DIR}/logs/reference.log"

export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "${ROOT}"

"${RWKV_PYTHON_BIN}" bench/bench_native_prefill_accum_ab.py \
  --model "${RWKV_72_MODEL}" --device cuda --dtype fp16 \
  --batch-sizes 1 8 --prompt-tokens 128 512 2048 --chunk-size 512 \
  --orders both --warmup 1 --steps 3 --min-cosine 0.9999 \
  --code-source repo --results "${OUT_DIR}/accumulation_ab.jsonl" \
  > "${OUT_DIR}/logs/accumulation_ab.log" 2>&1

"${RWKV_PYTHON_BIN}" bench/bench_cross_model_speed_resident.py \
  --model "${RWKV_72_MODEL}" --model-kind rwkv --model-role candidate \
  --model-pair rwkv-7.2b__qwen3.5-9b --model-size-label 7.2b \
  --benchmark-matrix qwen35_4090_g1i_7p2_adjusted_pd \
  --dtype fp16 --quantization none --device cuda \
  --batch-sizes 1 8 --prompt-tokens 128 512 2048 \
  --decode-tokens 128 512 --prefill-chunk-size 512 \
  --warmup "${WARMUP}" --runs "${RUNS}" \
  --rwkv-attn-mode fused_recurrent --rwkv-code-source repo --fail-fast \
  --results "${OUT_DIR}/candidate.jsonl" \
  > "${OUT_DIR}/logs/candidate.log" 2>&1

"${QWEN_PYTHON_BIN}" bench/bench_cross_model_speed_resident.py \
  --model "${QWEN_9_MODEL}" --model-kind qwen35 --model-role reference \
  --model-pair rwkv-7.2b__qwen3.5-9b --model-size-label 9b \
  --benchmark-matrix qwen35_4090_g1i_7p2_adjusted_pd \
  --dtype fp16 --quantization none --device cuda \
  --batch-sizes 1 8 --prompt-tokens 128 512 2048 \
  --decode-tokens 128 512 --prefill-chunk-size 512 \
  --warmup "${WARMUP}" --runs "${RUNS}" \
  --qwen-backend fla --qwen-conv-backend fla_triton \
  --require-qwen-fast-path --fail-fast \
  --results "${OUT_DIR}/qwen_reference.jsonl" \
  > "${OUT_DIR}/logs/reference.log" 2>&1

"${RWKV_PYTHON_BIN}" bench/summarize_4080_adjusted_pd.py \
  "${OUT_DIR}/candidate.jsonl" "${OUT_DIR}/qwen_reference.jsonl" \
  --expected-device "NVIDIA GeForce RTX 4090" \
  --axis rtx4090_g1i_7p2_parameter_adjusted_pd \
  --expected-qwen-backend qwen_fla_gated_delta_rule_fla_triton_conv \
  --output "${OUT_DIR}/summary.json" \
  --markdown-output "${OUT_DIR}/summary.md"
