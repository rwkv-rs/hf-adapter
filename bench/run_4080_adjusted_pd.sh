#!/usr/bin/env bash
# Reproduce the six RTX 4080 parameter-adjusted Prefill/Decode rows.
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OUT_DIR="${OUT_DIR:-${1:-}}"
RWKV_PYTHON_BIN="${RWKV_PYTHON_BIN:-python}"
QWEN_PYTHON_BIN="${QWEN_PYTHON_BIN:-python}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
WARMUP="${WARMUP:-3}"
RUNS="${RUNS:-7}"

required=(
  RWKV_04_MODEL RWKV_15_MODEL RWKV_29_MODEL
  QWEN_08_MODEL QWEN_2_MODEL QWEN_4_MODEL
)
if [[ -z "${OUT_DIR}" ]]; then
  echo "usage: OUT_DIR=... RWKV_04_MODEL=... RWKV_15_MODEL=... RWKV_29_MODEL=... \\" >&2
  echo "       QWEN_08_MODEL=... QWEN_2_MODEL=... QWEN_4_MODEL=... $0" >&2
  exit 2
fi
for name in "${required[@]}"; do
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
  "${python_bin}" "${ROOT}/bench/check_exact_gpu.py" --model 4080 --name "${gpu_name}"
done

"${RWKV_PYTHON_BIN}" - <<'PY'
import torch, transformers, triton
actual = (str(torch.__version__), str(torch.version.cuda), str(triton.__version__), str(transformers.__version__))
expected = ("2.11.0+cu130", "13.0", "3.6.0", "5.8.0")
assert actual == expected, f"RWKV runtime {actual} != validated {expected}"
assert hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation")
PY
"${QWEN_PYTHON_BIN}" - <<'PY'
from importlib.metadata import version
import torch, transformers, triton
actual = (
    str(torch.__version__), str(torch.version.cuda), str(triton.__version__),
    str(transformers.__version__), version("flash-linear-attention"),
    version("causal-conv1d"),
)
expected = ("2.6.0+cu124", "12.4", "3.2.0", "5.12.1", "0.5.1", "1.5.0.post8")
assert actual == expected, f"Qwen runtime {actual} != validated {expected}"
PY

mkdir -p "${OUT_DIR}"/{candidate,reference,logs}
rm -f "${OUT_DIR}"/{candidate,reference}/*.jsonl "${OUT_DIR}/logs"/*.log \
  "${OUT_DIR}"/{candidate.jsonl,qwen_reference.jsonl,summary.json,summary.md}
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "${ROOT}"

run_matrix() {
  local python_bin="$1" role="$2" kind="$3" model="$4" pair="$5"
  local size="$6" chunk="$7" label="$8" batch_sizes="$9"
  local -a batch_args
  read -r -a batch_args <<< "${batch_sizes}"
  shift 9
  "${python_bin}" bench/bench_cross_model_speed_resident.py \
    --model "${model}" --model-kind "${kind}" --model-role "${role}" \
    --model-pair "${pair}" --model-size-label "${size}" \
    --benchmark-matrix qwen35_4080_adjusted_pd --dtype fp16 --quantization none \
    --device cuda --batch-sizes "${batch_args[@]}" --prompt-tokens 128 512 2048 \
    --decode-tokens 128 512 --prefill-chunk-size "${chunk}" \
    --warmup "${WARMUP}" --runs "${RUNS}" "$@" \
    --results "${OUT_DIR}/${role}/${label}.jsonl" \
    > "${OUT_DIR}/logs/${role}_${label}.log" 2>&1
}

export RWKV7_FAST_TOKEN_BACKEND=native_graph
export RWKV7_NATIVE_PREFILL_GRAPH=1
export RWKV7_NATIVE_GRAPH_FUSED_NORM_MIX=1
export RWKV7_NATIVE_GRAPH_FUSED_NORM_MIX_NUM_WARPS=4
export RWKV7_NATIVE_GRAPH_ADA_LINEAR=0
export RWKV7_NATIVE_GRAPH_ADA_SPARSE_FFN=0
run_matrix "${RWKV_PYTHON_BIN}" candidate rwkv "${RWKV_04_MODEL}" \
  rwkv-0.4b__qwen3.5-0.8b 0.4b 512 0p4 "1 8" \
  --rwkv-attn-mode fused_recurrent --rwkv-code-source repo
run_matrix "${RWKV_PYTHON_BIN}" candidate rwkv "${RWKV_15_MODEL}" \
  rwkv-1.5b__qwen3.5-2b 1.5b 0 1p5 "1 8" \
  --rwkv-attn-mode fused_recurrent --rwkv-code-source repo
run_matrix "${RWKV_PYTHON_BIN}" candidate rwkv "${RWKV_29_MODEL}" \
  rwkv-2.9b__qwen3.5-4b 2.9b 0 2p9_b1 1 \
  --rwkv-attn-mode fused_recurrent --rwkv-code-source repo
run_matrix "${RWKV_PYTHON_BIN}" candidate rwkv "${RWKV_29_MODEL}" \
  rwkv-2.9b__qwen3.5-4b 2.9b 512 2p9_b8 8 \
  --rwkv-attn-mode fused_recurrent --rwkv-code-source repo

unset RWKV7_FAST_TOKEN_BACKEND RWKV7_NATIVE_PREFILL_GRAPH \
  RWKV7_NATIVE_GRAPH_FUSED_NORM_MIX RWKV7_NATIVE_GRAPH_FUSED_NORM_MIX_NUM_WARPS \
  RWKV7_NATIVE_GRAPH_ADA_LINEAR RWKV7_NATIVE_GRAPH_ADA_SPARSE_FFN
run_matrix "${QWEN_PYTHON_BIN}" reference qwen35 "${QWEN_08_MODEL}" \
  rwkv-0.4b__qwen3.5-0.8b 0.8b 512 0p8 "1 8" \
  --qwen-backend fla --qwen-conv-backend auto --require-qwen-fast-path
run_matrix "${QWEN_PYTHON_BIN}" reference qwen35 "${QWEN_2_MODEL}" \
  rwkv-1.5b__qwen3.5-2b 2b 0 2b "1 8" \
  --qwen-backend fla --qwen-conv-backend auto --require-qwen-fast-path
run_matrix "${QWEN_PYTHON_BIN}" reference qwen35 "${QWEN_4_MODEL}" \
  rwkv-2.9b__qwen3.5-4b 4b 512 4b "1 8" \
  --qwen-backend fla --qwen-conv-backend auto --require-qwen-fast-path

cat "${OUT_DIR}"/candidate/*.jsonl > "${OUT_DIR}/candidate.jsonl"
cat "${OUT_DIR}"/reference/*.jsonl > "${OUT_DIR}/qwen_reference.jsonl"
"${RWKV_PYTHON_BIN}" bench/summarize_4080_adjusted_pd.py \
  "${OUT_DIR}/candidate.jsonl" "${OUT_DIR}/qwen_reference.jsonl" \
  --output "${OUT_DIR}/summary.json" --markdown-output "${OUT_DIR}/summary.md"
