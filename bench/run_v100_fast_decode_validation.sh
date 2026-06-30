#!/usr/bin/env bash
# Run the V100 fast-decode validation bundle for the RWKV-7 HF adapter.
#
# Expected environment: activated Python env with torch/transformers/fla/rwkv.
# Optional env vars:
#   HF_DIR, PTH, DTYPE, DEVICE, PROMPT_TOKENS, DECODE_TOKENS, MICRO_STEPS,
#   FORWARD_FAST_STEPS, GENERATE_BATCH_SIZE, GENERATE_NEW_TOKENS, WARMUP_BATCH_SIZES,
#   NATIVE_GRAPH_CACHE_SIZE, COMPONENT_STEPS, NATIVE_DECODE_TOKENS, RESULTS, LOG_DIR
set -euo pipefail

export RWKV_V7_ON="${RWKV_V7_ON:-1}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

HF_DIR="${HF_DIR:-/home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf}"
PTH="${PTH:-/home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-20260129-ctx8192.pth}"
DTYPE="${DTYPE:-fp16}"
DEVICE="${DEVICE:-cuda}"
PROMPT_TOKENS="${PROMPT_TOKENS:-512}"
DECODE_TOKENS="${DECODE_TOKENS:-128}"
MICRO_STEPS="${MICRO_STEPS:-128}"
FORWARD_FAST_STEPS="${FORWARD_FAST_STEPS:-32}"
GENERATE_BATCH_SIZE="${GENERATE_BATCH_SIZE:-2}"
GENERATE_NEW_TOKENS="${GENERATE_NEW_TOKENS:-16}"
WARMUP_BATCH_SIZES="${WARMUP_BATCH_SIZES:-1 2 4 8}"
NATIVE_GRAPH_CACHE_SIZE="${NATIVE_GRAPH_CACHE_SIZE:-8}"
COMPONENT_STEPS="${COMPONENT_STEPS:-32}"
NATIVE_DECODE_TOKENS="${NATIVE_DECODE_TOKENS:-64}"
RESULTS="${RESULTS:-bench/results.jsonl}"
LOG_DIR="${LOG_DIR:-bench/logs}"

mkdir -p "${LOG_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG="${LOG_DIR}/v100_fast_decode_${STAMP}.log"
PROFILE_OUT="${LOG_DIR}/profile_hf_fast_decode_${STAMP}.json"

run() {
  echo
  echo "+ $*"
  "$@"
}

{
  echo "date=$(date -Is)"
  echo "hf_dir=${HF_DIR}"
  echo "pth=${PTH}"
  echo "dtype=${DTYPE} device=${DEVICE} prompt_tokens=${PROMPT_TOKENS} decode_tokens=${DECODE_TOKENS} micro_steps=${MICRO_STEPS} forward_fast_steps=${FORWARD_FAST_STEPS} generate_batch_size=${GENERATE_BATCH_SIZE} generate_new_tokens=${GENERATE_NEW_TOKENS} warmup_batch_sizes=${WARMUP_BATCH_SIZES} native_graph_cache_size=${NATIVE_GRAPH_CACHE_SIZE} component_steps=${COMPONENT_STEPS}"
  echo "results=${RESULTS} profile_out=${PROFILE_OUT}"

  run python tests/test_fast_decode_api.py \
    --model "${HF_DIR}" \
    --dtype "${DTYPE}" \
    --device "${DEVICE}" \
    --fuse-norm false \
    --batch-sizes 1 2 4 \
    --decode-steps 32 \
    --max-diff 0.2

  run python tests/test_fast_decode_api.py \
    --model "${HF_DIR}" \
    --dtype "${DTYPE}" \
    --device "${DEVICE}" \
    --fuse-norm false \
    --batch-sizes 1 2 4 \
    --fast-token-layouts 3d \
    --fast-token-backends native_jit \
    --decode-steps 16 \
    --max-diff 0.2

  run python tests/test_fast_decode_api.py \
    --model "${HF_DIR}" \
    --dtype "${DTYPE}" \
    --device "${DEVICE}" \
    --fuse-norm false \
    --batch-sizes 1 2 4 \
    --fast-token-layouts 3d \
    --fast-token-backends native_graph \
    --decode-steps 16 \
    --max-diff 0.2

  run python tests/test_batch_cache.py \
    --model "${HF_DIR}" \
    --dtype "${DTYPE}" \
    --device "${DEVICE}" \
    --fuse-norm false \
    --batch-sizes 1 2 4 \
    --prompt-tokens 64 \
    --decode-steps 8

  run python tests/test_dynamic_batch_cache.py \
    --model "${HF_DIR}" \
    --dtype "${DTYPE}" \
    --device "${DEVICE}" \
    --fuse-norm false \
    --batch-size 3 \
    --prompt-tokens 64 \
    --decode-steps 4 \
    --max-diff 0.2

  run python tests/test_hf_api_contract.py \
    --model "${HF_DIR}" \
    --dtype "${DTYPE}" \
    --device "${DEVICE}" \
    --fuse-norm false \
    --attn-mode fused_recurrent \
    --beam-new-tokens 2

  run python tests/test_chunked_prefill.py \
    --model "${HF_DIR}" \
    --dtype "${DTYPE}" \
    --device "${DEVICE}" \
    --fuse-norm false \
    --attn-mode fused_recurrent \
    --batch-size 2 \
    --chunk-sizes 32 64 128 \
    --max-diff 0.2

  run python tests/test_quantized_inference.py \
    --model "${HF_DIR}" \
    --dtype "${DTYPE}" \
    --device "${DEVICE}" \
    --attn-mode fused_recurrent \
    --quantization 8bit \
    --max-new-tokens 2 \
    --optional

  run python bench/bench_speed.py \
    --hf-dir "${HF_DIR}" \
    --pth "${PTH}" \
    --backend both \
    --dtype "${DTYPE}" \
    --prompt-tokens "${PROMPT_TOKENS}" \
    --decode-tokens "${DECODE_TOKENS}" \
    --device "${DEVICE}" \
    --warmup 2 \
    --runs 3 \
    --hf-logits-to-keep 1 \
    --attn-mode fused_recurrent \
    --fuse-norm false \
    --fast-cache true \
    --hf-decode-api rwkv7_forward_token \
    --results "${RESULTS}"

  run python bench/bench_speed.py \
    --hf-dir "${HF_DIR}" \
    --pth "${PTH}" \
    --backend hf \
    --dtype "${DTYPE}" \
    --prompt-tokens "${PROMPT_TOKENS}" \
    --decode-tokens "${DECODE_TOKENS}" \
    --device "${DEVICE}" \
    --warmup 2 \
    --runs 3 \
    --hf-logits-to-keep 1 \
    --attn-mode fused_recurrent \
    --fuse-norm false \
    --fast-cache true \
    --hf-decode-api rwkv7_forward_token \
    --fast-token-backend native_jit \
    --results "${RESULTS}"

  run python bench/bench_speed.py \
    --hf-dir "${HF_DIR}" \
    --pth "${PTH}" \
    --backend hf \
    --dtype "${DTYPE}" \
    --prompt-tokens "${PROMPT_TOKENS}" \
    --decode-tokens "${DECODE_TOKENS}" \
    --device "${DEVICE}" \
    --warmup 3 \
    --runs 3 \
    --hf-logits-to-keep 1 \
    --attn-mode fused_recurrent \
    --fuse-norm false \
    --fast-cache true \
    --hf-decode-api rwkv7_forward_token \
    --fast-token-backend native_graph \
    --results "${RESULTS}"

  run python bench/bench_batch_sweep.py \
    --hf-dir "${HF_DIR}" \
    --dtype "${DTYPE}" \
    --device "${DEVICE}" \
    --attn-mode fused_recurrent \
    --fuse-norm false \
    --fast-cache true \
    --fast-decode-api auto \
    --fast-token-backend native_jit \
    --batch-sizes 1 2 4 8 \
    --prompt-tokens "${PROMPT_TOKENS}" \
    --decode-tokens "${DECODE_TOKENS}" \
    --warmup 2 \
    --runs 3 \
    --results "${RESULTS}"

  run python bench/bench_batch_sweep.py \
    --hf-dir "${HF_DIR}" \
    --dtype "${DTYPE}" \
    --device "${DEVICE}" \
    --attn-mode fused_recurrent \
    --fuse-norm false \
    --fast-cache true \
    --fast-decode-api auto \
    --fast-token-backend native_graph \
    --batch-sizes 1 2 4 8 \
    --prompt-tokens "${PROMPT_TOKENS}" \
    --decode-tokens "${DECODE_TOKENS}" \
    --warmup 3 \
    --runs 3 \
    --results "${RESULTS}"

  run python bench/bench_dynamic_batch.py \
    --hf-dir "${HF_DIR}" \
    --dtype "${DTYPE}" \
    --device "${DEVICE}" \
    --attn-mode fused_recurrent \
    --fuse-norm false \
    --fast-cache true \
    --fast-token-backend native_jit \
    --decode-apis forward rwkv7_forward_token \
    --batch-size 8 \
    --min-batch-size 2 \
    --prompt-tokens 256 \
    --decode-steps "${DECODE_TOKENS}" \
    --warmup 8 \
    --reorder-every 4 \
    --drop-every 32 \
    --results "${RESULTS}"

  run python bench/bench_dynamic_batch.py \
    --hf-dir "${HF_DIR}" \
    --dtype "${DTYPE}" \
    --device "${DEVICE}" \
    --attn-mode fused_recurrent \
    --fuse-norm false \
    --fast-cache true \
    --fast-token-backend native_graph \
    --decode-apis forward rwkv7_forward_token \
    --batch-size 8 \
    --min-batch-size 2 \
    --prompt-tokens 256 \
    --decode-steps "${DECODE_TOKENS}" \
    --warmup 8 \
    --reorder-every 4 \
    --drop-every 32 \
    --results "${RESULTS}"

  run python bench/bench_chunked_prefill.py \
    --hf-dir "${HF_DIR}" \
    --dtype "${DTYPE}" \
    --device "${DEVICE}" \
    --attn-mode fused_recurrent \
    --fuse-norm false \
    --batch-size 2 \
    --prompt-tokens "${PROMPT_TOKENS}" \
    --chunk-sizes 64 128 256 \
    --warmup 1 \
    --runs 3 \
    --results "${RESULTS}"

  run python bench/bench_quantization.py \
    --hf-dir "${HF_DIR}" \
    --dtype "${DTYPE}" \
    --device "${DEVICE}" \
    --attn-mode fused_recurrent \
    --quantizations none 8bit 4bit \
    --prompt-tokens 256 \
    --decode-tokens 32 \
    --warmup 1 \
    --runs 2 \
    --optional \
    --results "${RESULTS}"

  run python bench/bench_decode_breakdown.py \
    --hf-dir "${HF_DIR}" \
    --pth "${PTH}" \
    --dtype "${DTYPE}" \
    --device "${DEVICE}" \
    --prompt-tokens "${PROMPT_TOKENS}" \
    --decode-tokens "${DECODE_TOKENS}" \
    --warmup 2 \
    --runs 3 \
    --attn-modes chunk fused_recurrent \
    --fuse-norm false \
    --fast-cache true \
    --fast-decode-api true \
    --results "${RESULTS}"

  run python bench/bench_decode_micro.py \
    --hf-dir "${HF_DIR}" \
    --dtype "${DTYPE}" \
    --device "${DEVICE}" \
    --attn-mode fused_recurrent \
    --fuse-norm false \
    --fast-cache true \
    --fast-decode-api auto \
    --fast-token-backend native_jit \
    --prompt-tokens 128 \
    --warmup 8 \
    --steps "${MICRO_STEPS}" \
    --results "${RESULTS}"

  run python bench/bench_forward_fast_path.py \
    --hf-dir "${HF_DIR}" \
    --dtype "${DTYPE}" \
    --device "${DEVICE}" \
    --attn-mode fused_recurrent \
    --fuse-norm false \
    --fast-cache true \
    --fast-token-backend auto \
    --prompt-tokens 64 \
    --warmup 2 \
    --steps "${FORWARD_FAST_STEPS}" \
    --results "${RESULTS}"

  run python bench/bench_generate_fast_path.py \
    --hf-dir "${HF_DIR}" \
    --dtype "${DTYPE}" \
    --device "${DEVICE}" \
    --attn-mode fused_recurrent \
    --fuse-norm false \
    --fast-cache true \
    --fast-token-backend auto \
    --batch-size "${GENERATE_BATCH_SIZE}" \
    --max-new-tokens "${GENERATE_NEW_TOKENS}" \
    --warmup-new-tokens 2 \
    --warmup 1 \
    --runs 2 \
    --results "${RESULTS}"

  run python bench/bench_fast_token_warmup.py \
    --hf-dir "${HF_DIR}" \
    --dtype "${DTYPE}" \
    --device "${DEVICE}" \
    --attn-mode fused_recurrent \
    --fuse-norm false \
    --fast-cache true \
    --fast-token-backend auto \
    --batch-sizes ${WARMUP_BATCH_SIZES} \
    --native-graph-cache-size "${NATIVE_GRAPH_CACHE_SIZE}" \
    --results "${RESULTS}"

  run python bench/bench_native_decode.py \
    --hf-dir "${HF_DIR}" \
    --dtype "${DTYPE}" \
    --device "${DEVICE}" \
    --prompt-tokens 32 \
    --decode-tokens "${NATIVE_DECODE_TOKENS}" \
    --greedy-check-tokens 16 \
    --results "${RESULTS}"

  run python bench/bench_decode_components.py \
    --hf-dir "${HF_DIR}" \
    --dtype "${DTYPE}" \
    --device "${DEVICE}" \
    --attn-mode fused_recurrent \
    --fuse-norm false \
    --fast-cache true \
    --batch-size 1 \
    --prompt-tokens 128 \
    --warmup 8 \
    --steps "${COMPONENT_STEPS}" \
    --fixed-token \
    --results "${RESULTS}"

  run python bench/bench_projection_lora.py \
    --hf-dir "${HF_DIR}" \
    --dtype "${DTYPE}" \
    --device "${DEVICE}" \
    --attn-mode fused_recurrent \
    --fuse-norm false \
    --fast-cache true \
    --batch-size 1 \
    --layers 0 1 11 \
    --warmup 16 \
    --steps 256 \
    --results "${RESULTS}"

  run python bench/profile_decode.py \
    --backend hf \
    --hf-dir "${HF_DIR}" \
    --dtype "${DTYPE}" \
    --device "${DEVICE}" \
    --attn-mode fused_recurrent \
    --fuse-norm false \
    --fast-cache true \
    --fixed-token \
    --hf-decode-api rwkv7_forward_token \
    --out "${PROFILE_OUT}"

  run python bench/summarize_results.py \
    --results "${RESULTS}" \
    --device V100 \
    --last 12

  run python bench/summarize_results.py \
    --results "${RESULTS}" \
    --device V100 \
    --require-fast-decode \
    --last 8

  run python bench/analyze_results.py \
    --results "${RESULTS}" \
    --device V100 \
    --dtype "${DTYPE}"

  run python bench/check_results.py \
    --results "${RESULTS}" \
    --device V100 \
    --dtype "${DTYPE}"

  echo
  echo "DONE log=${LOG} profile=${PROFILE_OUT}"
} 2>&1 | tee "${LOG}"
