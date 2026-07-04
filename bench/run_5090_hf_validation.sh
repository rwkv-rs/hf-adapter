#!/usr/bin/env bash
# RTX 5090 / Blackwell focused HF adapter validation.
#
# This mirrors the 4090 smoke requirements for a 50-series node: remote-code
# load/generate, HF API contract, native prefill, W8/W4 loadability, dynamic
# batching, and a small batch-sweep telemetry table.  Full MATH500 avg@64 still
# uses scripts/run_math500_acceptance.sh with a real MATH500 dataset/model.
#
# Example:
#   source /workspace/venvs/rwkv7-5090/bin/activate
#   bash bench/run_5090_hf_validation.sh HF_DIR=/workspace/models/rwkv7-g1d-0.1b-hf
set -euo pipefail

for arg in "$@"; do
  case "${arg}" in
    *=*) export "${arg}" ;;
    *) echo "unknown argument: ${arg}" >&2; exit 2 ;;
  esac
done

HF_DIR="${HF_DIR:-/workspace/models/rwkv7-g1d-0.1b-hf}"
OUT_DIR="${OUT_DIR:-bench/5090_blackwell_validation_$(date +%Y%m%d_%H%M%S)}"
RESULTS="${RESULTS:-${OUT_DIR}/results_5090.jsonl}"
DTYPE="${DTYPE:-fp16}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZES="${BATCH_SIZES:-1 2 4 8}"
PROMPT_TOKENS="${PROMPT_TOKENS:-32}"
DECODE_TOKENS="${DECODE_TOKENS:-32}"
MATH_ROLLOUT="${MATH_ROLLOUT:-4}"
MATH_LIMIT="${MATH_LIMIT:-2}"
MATH_MAX_NEW_TOKENS="${MATH_MAX_NEW_TOKENS:-64}"
MATH_BSZ="${MATH_BSZ:-8}"
MATH_SMOKE_DATASET="${MATH_SMOKE_DATASET:-${OUT_DIR}/math500_smoke.jsonl}"

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export RWKV_V7_ON="${RWKV_V7_ON:-1}"
# The early PyTorch 2.6 + Triton 3.3 Blackwell image has an Inductor/
# AttrsDescriptor failure in FLA sqrelu.  The adapter has a runtime fallback,
# but setting the env before Python import keeps logs and behavior deterministic.
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"

run() {
  echo "+ $*" >&2
  "$@"
}

mkdir -p "${OUT_DIR}"
if [[ ! -f "${MATH_SMOKE_DATASET}" ]]; then
  cat > "${MATH_SMOKE_DATASET}" <<'JSONL'
{"problem":"Compute 1+1. Put the answer in \\boxed{}.","answer":"2","subject":"smoke","level":"1","unique_id":"smoke-1"}
{"problem":"Compute 2+3. Put the answer in \\boxed{}.","answer":"5","subject":"smoke","level":"1","unique_id":"smoke-2"}
JSONL
fi

{
  echo "# RTX 5090 HF validation"
  echo "date=$(date -Is)"
  echo "hf_dir=${HF_DIR}"
  echo "out_dir=${OUT_DIR} results=${RESULTS}"
  echo "dtype=${DTYPE} device=${DEVICE} batch_sizes=${BATCH_SIZES}"
  python - <<'PY'
import os
import torch
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu', torch.cuda.get_device_name(0), 'capability', torch.cuda.get_device_capability(0))
print('TORCH_COMPILE_DISABLE', os.environ.get('TORCH_COMPILE_DISABLE'))
PY
} | tee "${OUT_DIR}/env.log"

run python -m py_compile \
  rwkv7_hf/triton_compat.py \
  rwkv7_hf/modeling_rwkv7.py \
  rwkv7_hf/native_jit.py \
  bench/eval_math500_hf.py \
  bench/bench_batch_sweep.py \
  bench/analyze_results.py \
  tests/smoke_hf_generate.py \
  tests/test_hf_api_contract.py \
  tests/test_fast_prefill_forward.py \
  tests/test_quantized_inference.py

run python tests/smoke_hf_generate.py --model "${HF_DIR}" --device "${DEVICE}" --max-new-tokens 4 \
  | tee "${OUT_DIR}/smoke_hf_generate.log"
run python tests/test_hf_api_contract.py --model "${HF_DIR}" --device "${DEVICE}" --dtype "${DTYPE}" --beam-new-tokens 2 \
  | tee "${OUT_DIR}/hf_api_contract.log"
run python tests/test_fast_prefill_forward.py --model "${HF_DIR}" --device "${DEVICE}" --prompt-tokens "${PROMPT_TOKENS}" --gen-tokens 2 \
  | tee "${OUT_DIR}/fast_prefill_forward.log"

run python tests/test_quantized_inference.py --model "${HF_DIR}" --device "${DEVICE}" --dtype "${DTYPE}" --quantization 8bit --max-new-tokens 2 --optional --skip-fast-forward-check \
  | tee "${OUT_DIR}/quant_8bit.log"
run python tests/test_quantized_inference.py --model "${HF_DIR}" --device "${DEVICE}" --dtype "${DTYPE}" --quantization 4bit --max-new-tokens 2 --optional --skip-fast-forward-check \
  | tee "${OUT_DIR}/quant_4bit.log"

run python bench/bench_batch_sweep.py \
  --hf-dir "${HF_DIR}" \
  --dtype "${DTYPE}" --device "${DEVICE}" \
  --attn-mode fused_recurrent --fuse-norm false --fast-cache true \
  --fast-decode-api true --fast-token-backend native_graph \
  --batch-sizes ${BATCH_SIZES} \
  --prompt-tokens "${PROMPT_TOKENS}" --decode-tokens "${DECODE_TOKENS}" \
  --warmup 1 --runs 1 \
  --results "${RESULTS}" \
  | tee "${OUT_DIR}/batch_sweep.log"

run python bench/eval_math500_hf.py \
  --hf-dir "${HF_DIR}" \
  --dataset "${MATH_SMOKE_DATASET}" \
  --out-dir "${OUT_DIR}/math500_native_prefill_smoke" \
  --rollout "${MATH_ROLLOUT}" --limit "${MATH_LIMIT}" \
  --max-new-tokens "${MATH_MAX_NEW_TOKENS}" --ctx-limit 512 \
  --dynamic-batching --bsz "${MATH_BSZ}" \
  --prefill-backend native --decode-backend fast_token \
  --dtype "${DTYPE}" --device "${DEVICE}" --progress-every 16 \
  --defer-verification --verify-workers 1 \
  --summary-speed-timing generation --defer-text-decode \
  | tee "${OUT_DIR}/math500_native_prefill_smoke.log"

run python bench/analyze_results.py --results "${RESULTS}" --device "NVIDIA GeForce RTX 5090" --dtype "${DTYPE}" --json \
  > "${RESULTS%.jsonl}.report.json"

echo "wrote ${OUT_DIR}"
echo "wrote ${RESULTS}"
echo "wrote ${RESULTS%.jsonl}.report.json"
