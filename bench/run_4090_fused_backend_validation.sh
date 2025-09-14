#!/usr/bin/env bash
# 4090/Ada focused fused-backend validation for rwkv7-hf-adapter.
#
# This is intentionally narrower than the V100 full gate: it validates the
# current HF-native fast-token path on an Ada card, proves the default
# fused recurrent+output kernel across active batch sizes, and samples opt-in
# fusion probes that have historically been microbench-positive but
# end-to-end-neutral/negative.
#
# Example:
#   source /workspace/activate_rwkv7.sh
#   bash bench/run_4090_fused_backend_validation.sh \
#     HF_DIR=/workspace/models/rwkv7/rwkv7-g1d-0.4b-hf
#
# Overridable env:
#   HF_DIR, RESULTS, DTYPE, DEVICE, BATCH_SIZES, PROMPT_TOKENS,
#   BATCH_SWEEP_DECODE_TOKENS, STEPS, WARMUP, RUN_NEGATIVE_PROBES.
set -euo pipefail

for arg in "$@"; do
  case "${arg}" in
    *=*) export "${arg}" ;;
    *) echo "unknown argument: ${arg}" >&2; exit 2 ;;
  esac
done

HF_DIR="${HF_DIR:-/workspace/models/rwkv7/rwkv7-g1d-0.4b-hf}"
RESULTS="${RESULTS:-bench/results_4090_fused_backend.jsonl}"
DTYPE="${DTYPE:-fp16}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZES="${BATCH_SIZES:-1 2 4 8}"
PROMPT_TOKENS="${PROMPT_TOKENS:-64}"
BATCH_SWEEP_DECODE_TOKENS="${BATCH_SWEEP_DECODE_TOKENS:-64}"
STEPS="${STEPS:-48}"
NEGATIVE_STEPS="${NEGATIVE_STEPS:-32}"
WARMUP="${WARMUP:-4}"
RUN_NEGATIVE_PROBES="${RUN_NEGATIVE_PROBES:-1}"

export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export RWKV_V7_ON="${RWKV_V7_ON:-1}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
# 4090 is Ada/sm_89. Leave unset if a caller wants PyTorch/Triton auto-detect,
# but use the right value when compiling CUDA extensions from this shell.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"

run() {
  echo "+ $*" >&2
  "$@"
}

mkdir -p "$(dirname "${RESULTS}")"

echo "# 4090 fused backend validation"
echo "date=$(date -Is)"
echo "hf_dir=${HF_DIR}"
echo "dtype=${DTYPE} device=${DEVICE} batch_sizes=${BATCH_SIZES} prompt_tokens=${PROMPT_TOKENS} steps=${STEPS} warmup=${WARMUP} results=${RESULTS}"
python - <<'PY'
import torch
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu', torch.cuda.get_device_name(0), 'capability', torch.cuda.get_device_capability(0))
PY

run python -m py_compile \
  rwkv7_hf/native_jit.py \
  rwkv7_hf/modeling_rwkv7.py \
  bench/bench_batch_sweep.py \
  bench/bench_native_graph_overhead.py \
  bench/bench_native_graph_fused_recurrent_output.py \
  bench/bench_native_graph_fused_wavg_lora.py \
  bench/bench_native_graph_fused_projection.py \
  bench/bench_native_graph_fused_output_project.py \
  bench/analyze_results.py

run python bench/bench_batch_sweep.py \
  --hf-dir "${HF_DIR}" \
  --dtype "${DTYPE}" \
  --device "${DEVICE}" \
  --attn-mode fused_recurrent \
  --fuse-norm false \
  --fast-cache true \
  --fast-decode-api true \
  --fast-token-backend native_graph \
  --batch-sizes ${BATCH_SIZES} \
  --prompt-tokens "${PROMPT_TOKENS}" \
  --decode-tokens "${BATCH_SWEEP_DECODE_TOKENS}" \
  --warmup 2 \
  --runs 2 \
  --results "${RESULTS}"

run python bench/bench_native_graph_overhead.py \
  --hf-dir "${HF_DIR}" \
  --dtype "${DTYPE}" \
  --device "${DEVICE}" \
  --attn-mode fused_recurrent \
  --fuse-norm false \
  --fast-cache true \
  --batch-sizes ${BATCH_SIZES} \
  --prompt-tokens "${PROMPT_TOKENS}" \
  --warmup "${WARMUP}" \
  --steps "${NEGATIVE_STEPS}" \
  --fixed-token \
  --results "${RESULTS}"

for bsz in ${BATCH_SIZES}; do
  run python bench/bench_native_graph_fused_recurrent_output.py \
    --hf-dir "${HF_DIR}" \
    --dtype "${DTYPE}" \
    --device "${DEVICE}" \
    --attn-mode fused_recurrent \
    --fuse-norm false \
    --fast-cache true \
    --batch-size "${bsz}" \
    --prompt-tokens "${PROMPT_TOKENS}" \
    --warmup "${WARMUP}" \
    --steps "${STEPS}" \
    --fixed-token \
    --results "${RESULTS}"
done

if [[ "${RUN_NEGATIVE_PROBES}" != "0" ]]; then
  for bsz in 1 4 8; do
    run python bench/bench_native_graph_fused_wavg_lora.py \
      --hf-dir "${HF_DIR}" \
      --dtype "${DTYPE}" \
      --device "${DEVICE}" \
      --attn-mode fused_recurrent \
      --fuse-norm false \
      --fast-cache true \
      --fused-recurrent-output \
      --fused-output \
      --batch-size "${bsz}" \
      --prompt-tokens "${PROMPT_TOKENS}" \
      --warmup "${WARMUP}" \
      --steps "${NEGATIVE_STEPS}" \
      --fixed-token \
      --results "${RESULTS}"
  done

  run python bench/bench_native_graph_fused_projection.py \
    --hf-dir "${HF_DIR}" \
    --dtype "${DTYPE}" \
    --device "${DEVICE}" \
    --attn-mode fused_recurrent \
    --fuse-norm false \
    --fast-cache true \
    --fused-output \
    --batch-size 4 \
    --prompt-tokens "${PROMPT_TOKENS}" \
    --warmup "${WARMUP}" \
    --steps "${NEGATIVE_STEPS}" \
    --fixed-token \
    --results "${RESULTS}"

  run python bench/bench_native_graph_fused_output_project.py \
    --hf-dir "${HF_DIR}" \
    --dtype "${DTYPE}" \
    --device "${DEVICE}" \
    --attn-mode fused_recurrent \
    --fuse-norm false \
    --fast-cache true \
    --fused-output \
    --batch-size 4 \
    --prompt-tokens "${PROMPT_TOKENS}" \
    --warmup "${WARMUP}" \
    --steps "${NEGATIVE_STEPS}" \
    --fixed-token \
    --results "${RESULTS}"
fi

run python bench/analyze_results.py --results "${RESULTS}" --device "NVIDIA GeForce RTX 4090" --dtype "${DTYPE}" --json > "${RESULTS%.jsonl}.report.json"
echo "wrote ${RESULTS}"
echo "wrote ${RESULTS%.jsonl}.report.json"
