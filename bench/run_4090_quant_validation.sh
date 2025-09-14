#!/usr/bin/env bash
# 4090/Ada focused native-quant validation for rwkv7-hf-adapter.
#
# This gate tracks the RWKV-native W8/W4 fused dequant-GEMV path separately
# from generic bitsandbytes compatibility.  It is telemetry-first: the current
# target is to prove packed footprint reduction, correctness against the
# separate quant reference, and the remaining speed gap versus fp16 cuBLAS.
#
# Example:
#   source /workspace/activate_rwkv7.sh
#   bash bench/run_4090_quant_validation.sh \
#     HF_DIR=/workspace/models/rwkv7/rwkv7-g1d-0.4b-hf
#
# Overridable env:
#   HF_DIR, RESULTS, DTYPE, DEVICE, LAYERS, BATCH_SIZE, BLOCK_M, BLOCK_K,
#   STEPS, WARMUP.
set -euo pipefail

for arg in "$@"; do
  case "${arg}" in
    *=*) export "${arg}" ;;
    *) echo "unknown argument: ${arg}" >&2; exit 2 ;;
  esac
done

HF_DIR="${HF_DIR:-/workspace/models/rwkv7/rwkv7-g1d-0.4b-hf}"
RESULTS="${RESULTS:-bench/results_4090_quant.jsonl}"
DTYPE="${DTYPE:-fp16}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LAYERS="${LAYERS:-0 1 23}"
BLOCK_M="${BLOCK_M:-8 16 32 64}"
BLOCK_K="${BLOCK_K:-32 64 128}"
STEPS="${STEPS:-128}"
WARMUP="${WARMUP:-8}"

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export RWKV_V7_ON="${RWKV_V7_ON:-1}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
# 4090 is Ada/sm_89. Leave unset only if a caller intentionally wants auto.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"

run() {
  echo "+ $*" >&2
  "$@"
}

mkdir -p "$(dirname "${RESULTS}")"

echo "# 4090 native quant validation"
echo "date=$(date -Is)"
echo "hf_dir=${HF_DIR}"
echo "dtype=${DTYPE} device=${DEVICE} batch_size=${BATCH_SIZE} layers=${LAYERS}"
echo "block_m=${BLOCK_M} block_k=${BLOCK_K} steps=${STEPS} warmup=${WARMUP} results=${RESULTS}"
python - <<'PY'
import torch
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu', torch.cuda.get_device_name(0), 'capability', torch.cuda.get_device_capability(0))
PY

run python -m py_compile \
  rwkv7_hf/native_quant.py \
  bench/bench_native_quant_rkv_sweep.py \
  bench/analyze_results.py

run python bench/bench_native_quant_rkv_sweep.py \
  --hf-dir "${HF_DIR}" \
  --dtype "${DTYPE}" \
  --device "${DEVICE}" \
  --attn-mode fused_recurrent \
  --fuse-norm false \
  --batch-size "${BATCH_SIZE}" \
  --layers ${LAYERS} \
  --quantizations w8 w4 \
  --block-m ${BLOCK_M} \
  --block-k ${BLOCK_K} \
  --warmup "${WARMUP}" \
  --steps "${STEPS}" \
  --results "${RESULTS}"

run python bench/analyze_results.py --results "${RESULTS}" --device "NVIDIA GeForce RTX 4090" --dtype "${DTYPE}" --json > "${RESULTS%.jsonl}.report.json"
echo "wrote ${RESULTS}"
echo "wrote ${RESULTS%.jsonl}.report.json"
