#!/usr/bin/env bash
# Run the V100 fast-decode validation bundle for the RWKV-7 HF adapter.
#
# Expected environment: activated Python env with torch/transformers/fla/rwkv.
# Optional env vars:
#   HF_DIR, PTH, DTYPE, DEVICE, PROMPT_TOKENS, DECODE_TOKENS, MICRO_STEPS,
#   FORWARD_FAST_STEPS, GENERATE_BATCH_SIZE, GENERATE_NEW_TOKENS, WARMUP_BATCH_SIZES,
#   NATIVE_GRAPH_CACHE_SIZE, NATIVE_GRAPH_OVERHEAD_BATCH_SIZES, NATIVE_GRAPH_OVERHEAD_STEPS,
#   COMPONENT_STEPS, FUSED_PROJECTION_STEPS, FUSED_WA_LORA_STEPS, FUSED_WAG_LORA_STEPS, FUSED_RKV_WAG_PROJECTION_STEPS,
#   FUSED_ATTN_OUTPUT_STEPS, FUSED_ATTN_OUTPUT_INPUT_SCALE, FUSED_FFN_STEPS, FUSED_SHIFT_MIX_STEPS,
#   FUSED_RECURRENT_STEPS, NATIVE_GRAPH_FUSED_RECURRENT_STEPS, NATIVE_GRAPH_FUSED_OUTPUT_STEPS,
#   NATIVE_QUANT_GEMV_STEPS, NATIVE_QUANT_W4_GEMV_STEPS,
#   NATIVE_QUANT_RKV_STEPS, NATIVE_QUANT_W4_RKV_STEPS,
#   NATIVE_DECODE_TOKENS, RUN_LARGER_MODEL_SMOKE, LARGER_HF_DIR,
#   LARGER_PTH, LARGER_MODEL_SIZE_LABEL, LARGER_MAX_NEW_TOKENS,
#   RUN_15B_MODEL_SMOKE, LARGER_15_HF_DIR, LARGER_15_PTH,
#   LARGER_FAST_TOKEN_BACKEND, LARGER_15_MAX_NEW_TOKENS,
#   LARGER_15_FAST_TOKEN_BACKEND, RUN_29B_MODEL_SMOKE, LARGER_29_HF_DIR,
#   LARGER_29_PTH, LARGER_29_MAX_NEW_TOKENS, LARGER_29_FAST_TOKEN_BACKEND,
#   RUN_72B_MODEL_SMOKE, LARGER_72_HF_DIR, LARGER_72_PTH,
#   LARGER_72_MAX_NEW_TOKENS, LARGER_72_FAST_TOKEN_BACKEND,
#   RUN_133B_MODEL_SMOKE, LARGER_133_HF_DIR, LARGER_133_PTH,
#   LARGER_133_MAX_NEW_TOKENS, LARGER_133_FAST_TOKEN_BACKEND,
#   RUN_DEVICE_MAP_SMOKE, DEVICE_MAP_MAX_NEW_TOKENS,
#   RUN_SPECULATIVE_BENCH, SPEC_TARGET_HF_DIR, SPEC_DRAFT_HF_DIR,
#   SPEC_MAX_NEW_TOKENS, SPEC_DRAFT_TOKENS, RESULTS, LOG_DIR
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
NATIVE_GRAPH_OVERHEAD_BATCH_SIZES="${NATIVE_GRAPH_OVERHEAD_BATCH_SIZES:-1 2 4 8}"
NATIVE_GRAPH_OVERHEAD_STEPS="${NATIVE_GRAPH_OVERHEAD_STEPS:-32}"
COMPONENT_STEPS="${COMPONENT_STEPS:-32}"
FUSED_PROJECTION_STEPS="${FUSED_PROJECTION_STEPS:-128}"
FUSED_WA_LORA_STEPS="${FUSED_WA_LORA_STEPS:-128}"
FUSED_WAG_LORA_STEPS="${FUSED_WAG_LORA_STEPS:-128}"
FUSED_RKV_WAG_PROJECTION_STEPS="${FUSED_RKV_WAG_PROJECTION_STEPS:-128}"
FUSED_ATTN_OUTPUT_STEPS="${FUSED_ATTN_OUTPUT_STEPS:-128}"
FUSED_ATTN_OUTPUT_INPUT_SCALE="${FUSED_ATTN_OUTPUT_INPUT_SCALE:-0.3}"
FUSED_FFN_STEPS="${FUSED_FFN_STEPS:-128}"
FUSED_SHIFT_MIX_STEPS="${FUSED_SHIFT_MIX_STEPS:-512}"
FUSED_RECURRENT_STEPS="${FUSED_RECURRENT_STEPS:-256}"
NATIVE_GRAPH_FUSED_RECURRENT_STEPS="${NATIVE_GRAPH_FUSED_RECURRENT_STEPS:-32}"
NATIVE_GRAPH_FUSED_OUTPUT_STEPS="${NATIVE_GRAPH_FUSED_OUTPUT_STEPS:-32}"
NATIVE_QUANT_GEMV_STEPS="${NATIVE_QUANT_GEMV_STEPS:-128}"
NATIVE_QUANT_W4_GEMV_STEPS="${NATIVE_QUANT_W4_GEMV_STEPS:-128}"
NATIVE_QUANT_RKV_STEPS="${NATIVE_QUANT_RKV_STEPS:-128}"
NATIVE_QUANT_W4_RKV_STEPS="${NATIVE_QUANT_W4_RKV_STEPS:-128}"
NATIVE_DECODE_TOKENS="${NATIVE_DECODE_TOKENS:-64}"
RUN_LARGER_MODEL_SMOKE="${RUN_LARGER_MODEL_SMOKE:-auto}"
LARGER_HF_DIR="${LARGER_HF_DIR:-/home/data/wangyue/models/rwkv7/rwkv7-g1d-0.4b-hf}"
LARGER_PTH="${LARGER_PTH:-/home/data/wangyue/models/rwkv7/rwkv7-g1d-0.4b-20260210-ctx8192.pth}"
LARGER_MODEL_SIZE_LABEL="${LARGER_MODEL_SIZE_LABEL:-0.4b}"
LARGER_MAX_NEW_TOKENS="${LARGER_MAX_NEW_TOKENS:-4}"
LARGER_FAST_TOKEN_BACKEND="${LARGER_FAST_TOKEN_BACKEND:-auto}"
RUN_15B_MODEL_SMOKE="${RUN_15B_MODEL_SMOKE:-auto}"
LARGER_15_HF_DIR="${LARGER_15_HF_DIR:-/home/data/wangyue/models/rwkv7/rwkv7-g1g-1.5b-hf}"
LARGER_15_PTH="${LARGER_15_PTH:-/home/data/wangyue/models/rwkv7/rwkv7-g1g-1.5b-20260526-ctx8192.pth}"
LARGER_15_MAX_NEW_TOKENS="${LARGER_15_MAX_NEW_TOKENS:-2}"
LARGER_15_FAST_TOKEN_BACKEND="${LARGER_15_FAST_TOKEN_BACKEND:-auto}"
RUN_29B_MODEL_SMOKE="${RUN_29B_MODEL_SMOKE:-auto}"
LARGER_29_HF_DIR="${LARGER_29_HF_DIR:-/home/data/wangyue/models/rwkv7/rwkv7-g1g-2.9b-hf}"
LARGER_29_PTH="${LARGER_29_PTH:-/home/data/wangyue/models/rwkv7/rwkv7-g1g-2.9b-20260526-ctx8192.pth}"
LARGER_29_MAX_NEW_TOKENS="${LARGER_29_MAX_NEW_TOKENS:-2}"
LARGER_29_FAST_TOKEN_BACKEND="${LARGER_29_FAST_TOKEN_BACKEND:-auto}"
RUN_72B_MODEL_SMOKE="${RUN_72B_MODEL_SMOKE:-auto}"
LARGER_72_HF_DIR="${LARGER_72_HF_DIR:-/home/data/wangyue/models/rwkv7/rwkv7-g1g-7.2b-hf}"
LARGER_72_PTH="${LARGER_72_PTH:-/home/data/wangyue/models/rwkv7/rwkv7-g1g-7.2b-20260523-ctx8192.pth}"
LARGER_72_MAX_NEW_TOKENS="${LARGER_72_MAX_NEW_TOKENS:-2}"
LARGER_72_FAST_TOKEN_BACKEND="${LARGER_72_FAST_TOKEN_BACKEND:-auto}"
RUN_133B_MODEL_SMOKE="${RUN_133B_MODEL_SMOKE:-auto}"
LARGER_133_HF_DIR="${LARGER_133_HF_DIR:-/home/data/wangyue/models/rwkv7/rwkv7-g1g-13.3b-hf}"
LARGER_133_PTH="${LARGER_133_PTH:-/home/data/wangyue/models/rwkv7/rwkv7-g1g-13.3b-20260523-ctx8192.pth}"
LARGER_133_MAX_NEW_TOKENS="${LARGER_133_MAX_NEW_TOKENS:-2}"
# 13.3B fits V100 fp16 smoke with native_jit; native_graph capture may reserve
# too much extra memory on 32GB cards, so keep the default conservative.
LARGER_133_FAST_TOKEN_BACKEND="${LARGER_133_FAST_TOKEN_BACKEND:-native_jit}"
RUN_DEVICE_MAP_SMOKE="${RUN_DEVICE_MAP_SMOKE:-auto}"
DEVICE_MAP_MAX_NEW_TOKENS="${DEVICE_MAP_MAX_NEW_TOKENS:-4}"
RUN_SPECULATIVE_BENCH="${RUN_SPECULATIVE_BENCH:-auto}"
SPEC_TARGET_HF_DIR="${SPEC_TARGET_HF_DIR:-/home/data/wangyue/models/rwkv7/rwkv7-g1d-0.4b-hf}"
SPEC_DRAFT_HF_DIR="${SPEC_DRAFT_HF_DIR:-/home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf}"
SPEC_MAX_NEW_TOKENS="${SPEC_MAX_NEW_TOKENS:-8}"
SPEC_DRAFT_TOKENS="${SPEC_DRAFT_TOKENS:-4}"
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

should_run_larger_smoke() {
  local mode="$1"
  local hf_dir="$2"
  local pth="$3"
  [[ "${mode}" == "1" || "${mode}" == "true" || ( "${mode}" == "auto" && -d "${hf_dir}" && -f "${pth}" ) ]]
}

has_two_cuda_devices() {
  python - <<'PYCHECK'
import torch
raise SystemExit(0 if torch.cuda.is_available() and torch.cuda.device_count() >= 2 else 1)
PYCHECK
}

should_run_device_map_smoke() {
  local mode="$1"
  [[ "${mode}" == "1" || "${mode}" == "true" ]] && return 0
  [[ "${mode}" == "auto" ]] && has_two_cuda_devices
}

should_run_speculative_bench() {
  local mode="$1"
  local target_hf_dir="$2"
  local draft_hf_dir="$3"
  [[ "${mode}" == "1" || "${mode}" == "true" ]] && return 0
  [[ "${mode}" == "auto" && -d "${target_hf_dir}" && -d "${draft_hf_dir}" ]]
}

run_larger_smoke() {
  local hf_dir="$1"
  local pth="$2"
  local label="$3"
  local max_new="$4"
  local fast_backend="${5:-auto}"
  run python bench/bench_larger_model_smoke.py \
    --hf-dir "${hf_dir}" \
    --model-size-label "${label}" \
    --checkpoint-path "${pth}" \
    --dtype "${DTYPE}" \
    --device "${DEVICE}" \
    --attn-mode fused_recurrent \
    --fast-token-backend "${fast_backend}" \
    --max-new-tokens "${max_new}" \
    --results "${RESULTS}"
}

{
  echo "date=$(date -Is)"
  echo "hf_dir=${HF_DIR}"
  echo "pth=${PTH}"
  echo "dtype=${DTYPE} device=${DEVICE} prompt_tokens=${PROMPT_TOKENS} decode_tokens=${DECODE_TOKENS} micro_steps=${MICRO_STEPS} forward_fast_steps=${FORWARD_FAST_STEPS} generate_batch_size=${GENERATE_BATCH_SIZE} generate_new_tokens=${GENERATE_NEW_TOKENS} warmup_batch_sizes=${WARMUP_BATCH_SIZES} native_graph_cache_size=${NATIVE_GRAPH_CACHE_SIZE} native_graph_overhead_batch_sizes=${NATIVE_GRAPH_OVERHEAD_BATCH_SIZES} native_graph_overhead_steps=${NATIVE_GRAPH_OVERHEAD_STEPS} component_steps=${COMPONENT_STEPS} fused_projection_steps=${FUSED_PROJECTION_STEPS} fused_wa_lora_steps=${FUSED_WA_LORA_STEPS} fused_wag_lora_steps=${FUSED_WAG_LORA_STEPS} fused_rkv_wag_projection_steps=${FUSED_RKV_WAG_PROJECTION_STEPS} fused_attn_output_steps=${FUSED_ATTN_OUTPUT_STEPS} fused_attn_output_input_scale=${FUSED_ATTN_OUTPUT_INPUT_SCALE} fused_ffn_steps=${FUSED_FFN_STEPS} fused_shift_mix_steps=${FUSED_SHIFT_MIX_STEPS} fused_recurrent_steps=${FUSED_RECURRENT_STEPS} native_graph_fused_recurrent_steps=${NATIVE_GRAPH_FUSED_RECURRENT_STEPS} native_graph_fused_output_steps=${NATIVE_GRAPH_FUSED_OUTPUT_STEPS} native_quant_gemv_steps=${NATIVE_QUANT_GEMV_STEPS} native_quant_w4_gemv_steps=${NATIVE_QUANT_W4_GEMV_STEPS} native_quant_rkv_steps=${NATIVE_QUANT_RKV_STEPS} native_quant_w4_rkv_steps=${NATIVE_QUANT_W4_RKV_STEPS}"
  echo "larger_smoke=${RUN_LARGER_MODEL_SMOKE} larger_hf_dir=${LARGER_HF_DIR} larger_pth=${LARGER_PTH} larger_model_size_label=${LARGER_MODEL_SIZE_LABEL} larger_max_new_tokens=${LARGER_MAX_NEW_TOKENS} larger_fast_token_backend=${LARGER_FAST_TOKEN_BACKEND}"
  echo "larger_15_smoke=${RUN_15B_MODEL_SMOKE} larger_15_hf_dir=${LARGER_15_HF_DIR} larger_15_pth=${LARGER_15_PTH} larger_15_max_new_tokens=${LARGER_15_MAX_NEW_TOKENS} larger_15_fast_token_backend=${LARGER_15_FAST_TOKEN_BACKEND}"
  echo "larger_29_smoke=${RUN_29B_MODEL_SMOKE} larger_29_hf_dir=${LARGER_29_HF_DIR} larger_29_pth=${LARGER_29_PTH} larger_29_max_new_tokens=${LARGER_29_MAX_NEW_TOKENS} larger_29_fast_token_backend=${LARGER_29_FAST_TOKEN_BACKEND}"
  echo "larger_72_smoke=${RUN_72B_MODEL_SMOKE} larger_72_hf_dir=${LARGER_72_HF_DIR} larger_72_pth=${LARGER_72_PTH} larger_72_max_new_tokens=${LARGER_72_MAX_NEW_TOKENS} larger_72_fast_token_backend=${LARGER_72_FAST_TOKEN_BACKEND}"
  echo "larger_133_smoke=${RUN_133B_MODEL_SMOKE} larger_133_hf_dir=${LARGER_133_HF_DIR} larger_133_pth=${LARGER_133_PTH} larger_133_max_new_tokens=${LARGER_133_MAX_NEW_TOKENS} larger_133_fast_token_backend=${LARGER_133_FAST_TOKEN_BACKEND}"
  echo "device_map_smoke=${RUN_DEVICE_MAP_SMOKE} device_map_max_new_tokens=${DEVICE_MAP_MAX_NEW_TOKENS}"
  echo "speculative_bench=${RUN_SPECULATIVE_BENCH} spec_target_hf_dir=${SPEC_TARGET_HF_DIR} spec_draft_hf_dir=${SPEC_DRAFT_HF_DIR} spec_max_new_tokens=${SPEC_MAX_NEW_TOKENS} spec_draft_tokens=${SPEC_DRAFT_TOKENS}"
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

  if should_run_device_map_smoke "${RUN_DEVICE_MAP_SMOKE}"; then
    run python tests/test_device_map_generate.py \
      --model "${HF_DIR}" \
      --dtype "${DTYPE}" \
      --attn-mode fused_recurrent \
      --max-new-tokens "${DEVICE_MAP_MAX_NEW_TOKENS}" \
      --compare-single-device \
      --results "${RESULTS}"
  else
    echo "SKIP device_map smoke: RUN_DEVICE_MAP_SMOKE=${RUN_DEVICE_MAP_SMOKE} requires >=2 CUDA devices"
  fi

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

  run python bench/bench_native_graph_overhead.py \
    --hf-dir "${HF_DIR}" \
    --dtype "${DTYPE}" \
    --device "${DEVICE}" \
    --attn-mode fused_recurrent \
    --fuse-norm false \
    --fast-cache true \
    --batch-sizes ${NATIVE_GRAPH_OVERHEAD_BATCH_SIZES} \
    --prompt-tokens 64 \
    --warmup 4 \
    --steps "${NATIVE_GRAPH_OVERHEAD_STEPS}" \
    --fixed-token \
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

  run python bench/bench_fused_projection.py \
    --hf-dir "${HF_DIR}" \
    --dtype "${DTYPE}" \
    --device "${DEVICE}" \
    --attn-mode fused_recurrent \
    --fuse-norm false \
    --batch-size 1 \
    --layers 0 1 11 \
    --warmup 8 \
    --steps "${FUSED_PROJECTION_STEPS}" \
    --results "${RESULTS}"

  run python bench/bench_fused_wa_lora.py \
    --hf-dir "${HF_DIR}" \
    --dtype "${DTYPE}" \
    --device "${DEVICE}" \
    --attn-mode fused_recurrent \
    --fuse-norm false \
    --batch-size 1 \
    --layers 0 1 11 \
    --block-m 64 \
    --block-r 64 \
    --block-k 64 \
    --warmup 8 \
    --steps "${FUSED_WA_LORA_STEPS}" \
    --results "${RESULTS}"

  run python bench/bench_fused_wag_lora.py \
    --hf-dir "${HF_DIR}" \
    --dtype "${DTYPE}" \
    --device "${DEVICE}" \
    --attn-mode fused_recurrent \
    --fuse-norm false \
    --batch-size 1 \
    --layers 0 1 11 \
    --block-m 64 \
    --block-r 64 \
    --block-k 64 \
    --warmup 8 \
    --steps "${FUSED_WAG_LORA_STEPS}" \
    --results "${RESULTS}"

  run python bench/bench_fused_rkv_wag_projection.py \
    --hf-dir "${HF_DIR}" \
    --dtype "${DTYPE}" \
    --device "${DEVICE}" \
    --attn-mode fused_recurrent \
    --fuse-norm false \
    --batch-size 1 \
    --layers 0 1 11 \
    --block-m 64 \
    --block-r 64 \
    --block-k 64 \
    --warmup 8 \
    --steps "${FUSED_RKV_WAG_PROJECTION_STEPS}" \
    --results "${RESULTS}"

  run python bench/bench_fused_attn_output.py \
    --hf-dir "${HF_DIR}" \
    --dtype "${DTYPE}" \
    --device "${DEVICE}" \
    --attn-mode fused_recurrent \
    --fuse-norm false \
    --batch-size 1 \
    --layers 0 1 11 \
    --input-scale "${FUSED_ATTN_OUTPUT_INPUT_SCALE}" \
    --warmup 16 \
    --steps "${FUSED_ATTN_OUTPUT_STEPS}" \
    --results "${RESULTS}"

  run python bench/bench_fused_ffn.py \
    --hf-dir "${HF_DIR}" \
    --dtype "${DTYPE}" \
    --device "${DEVICE}" \
    --attn-mode fused_recurrent \
    --fuse-norm false \
    --batch-size 1 \
    --layers 0 1 11 \
    --block-m 128 \
    --block-k 128 \
    --warmup 8 \
    --steps "${FUSED_FFN_STEPS}" \
    --results "${RESULTS}"

  run python bench/bench_fused_shift_mix.py \
    --hf-dir "${HF_DIR}" \
    --dtype "${DTYPE}" \
    --device "${DEVICE}" \
    --attn-mode fused_recurrent \
    --fuse-norm false \
    --batch-size 1 \
    --input-rank 2 \
    --layers 0 1 11 \
    --warmup 32 \
    --steps "${FUSED_SHIFT_MIX_STEPS}" \
    --results "${RESULTS}"

  run python bench/bench_fused_recurrent.py \
    --hf-dir "${HF_DIR}" \
    --dtype "${DTYPE}" \
    --device "${DEVICE}" \
    --attn-mode fused_recurrent \
    --fuse-norm false \
    --batch-size 1 \
    --layers 0 1 11 \
    --warmup 16 \
    --steps "${FUSED_RECURRENT_STEPS}" \
    --results "${RESULTS}"

  run python bench/bench_native_graph_fused_recurrent.py \
    --hf-dir "${HF_DIR}" \
    --dtype "${DTYPE}" \
    --device "${DEVICE}" \
    --attn-mode fused_recurrent \
    --fuse-norm false \
    --fast-cache true \
    --batch-size 1 \
    --prompt-tokens 64 \
    --warmup 4 \
    --steps "${NATIVE_GRAPH_FUSED_RECURRENT_STEPS}" \
    --fixed-token \
    --results "${RESULTS}"

  run python bench/bench_native_graph_fused_output.py \
    --hf-dir "${HF_DIR}" \
    --dtype "${DTYPE}" \
    --device "${DEVICE}" \
    --attn-mode fused_recurrent \
    --fuse-norm false \
    --fast-cache true \
    --batch-size 1 \
    --prompt-tokens 64 \
    --warmup 4 \
    --steps "${NATIVE_GRAPH_FUSED_OUTPUT_STEPS}" \
    --fixed-token \
    --results "${RESULTS}"

  run python bench/bench_native_quant_gemv.py \
    --hf-dir "${HF_DIR}" \
    --dtype "${DTYPE}" \
    --device "${DEVICE}" \
    --attn-mode fused_recurrent \
    --fuse-norm false \
    --batch-size 1 \
    --layers 0 1 11 \
    --modules attn.r_proj attn.k_proj attn.v_proj attn.o_proj ffn.key ffn.value \
    --warmup 8 \
    --steps "${NATIVE_QUANT_GEMV_STEPS}" \
    --results "${RESULTS}"

  run python bench/bench_native_quant_w4_gemv.py \
    --hf-dir "${HF_DIR}" \
    --dtype "${DTYPE}" \
    --device "${DEVICE}" \
    --attn-mode fused_recurrent \
    --fuse-norm false \
    --batch-size 1 \
    --layers 0 1 11 \
    --modules attn.r_proj attn.k_proj attn.v_proj attn.o_proj ffn.key ffn.value \
    --warmup 8 \
    --steps "${NATIVE_QUANT_W4_GEMV_STEPS}" \
    --results "${RESULTS}"

  run python bench/bench_native_quant_rkv.py \
    --hf-dir "${HF_DIR}" \
    --dtype "${DTYPE}" \
    --device "${DEVICE}" \
    --attn-mode fused_recurrent \
    --fuse-norm false \
    --batch-size 1 \
    --layers 0 1 11 \
    --warmup 8 \
    --steps "${NATIVE_QUANT_RKV_STEPS}" \
    --results "${RESULTS}"

  run python bench/bench_native_quant_w4_rkv.py \
    --hf-dir "${HF_DIR}" \
    --dtype "${DTYPE}" \
    --device "${DEVICE}" \
    --attn-mode fused_recurrent \
    --fuse-norm false \
    --batch-size 1 \
    --layers 0 1 11 \
    --warmup 8 \
    --steps "${NATIVE_QUANT_W4_RKV_STEPS}" \
    --results "${RESULTS}"

  if should_run_larger_smoke "${RUN_LARGER_MODEL_SMOKE}" "${LARGER_HF_DIR}" "${LARGER_PTH}"; then
    run_larger_smoke "${LARGER_HF_DIR}" "${LARGER_PTH}" "${LARGER_MODEL_SIZE_LABEL}" "${LARGER_MAX_NEW_TOKENS}" "${LARGER_FAST_TOKEN_BACKEND}"
  else
    echo "SKIP larger-model smoke: RUN_LARGER_MODEL_SMOKE=${RUN_LARGER_MODEL_SMOKE} LARGER_HF_DIR=${LARGER_HF_DIR} LARGER_PTH=${LARGER_PTH}"
  fi
  if should_run_larger_smoke "${RUN_15B_MODEL_SMOKE}" "${LARGER_15_HF_DIR}" "${LARGER_15_PTH}"; then
    run_larger_smoke "${LARGER_15_HF_DIR}" "${LARGER_15_PTH}" "1.5b" "${LARGER_15_MAX_NEW_TOKENS}" "${LARGER_15_FAST_TOKEN_BACKEND}"
  else
    echo "SKIP 1.5B larger-model smoke: RUN_15B_MODEL_SMOKE=${RUN_15B_MODEL_SMOKE} LARGER_15_HF_DIR=${LARGER_15_HF_DIR} LARGER_15_PTH=${LARGER_15_PTH}"
  fi
  if should_run_larger_smoke "${RUN_29B_MODEL_SMOKE}" "${LARGER_29_HF_DIR}" "${LARGER_29_PTH}"; then
    run_larger_smoke "${LARGER_29_HF_DIR}" "${LARGER_29_PTH}" "2.9b" "${LARGER_29_MAX_NEW_TOKENS}" "${LARGER_29_FAST_TOKEN_BACKEND}"
  else
    echo "SKIP 2.9B larger-model smoke: RUN_29B_MODEL_SMOKE=${RUN_29B_MODEL_SMOKE} LARGER_29_HF_DIR=${LARGER_29_HF_DIR} LARGER_29_PTH=${LARGER_29_PTH}"
  fi
  if should_run_larger_smoke "${RUN_72B_MODEL_SMOKE}" "${LARGER_72_HF_DIR}" "${LARGER_72_PTH}"; then
    run_larger_smoke "${LARGER_72_HF_DIR}" "${LARGER_72_PTH}" "7.2b" "${LARGER_72_MAX_NEW_TOKENS}" "${LARGER_72_FAST_TOKEN_BACKEND}"
  else
    echo "SKIP 7.2B larger-model smoke: RUN_72B_MODEL_SMOKE=${RUN_72B_MODEL_SMOKE} LARGER_72_HF_DIR=${LARGER_72_HF_DIR} LARGER_72_PTH=${LARGER_72_PTH}"
  fi
  if should_run_larger_smoke "${RUN_133B_MODEL_SMOKE}" "${LARGER_133_HF_DIR}" "${LARGER_133_PTH}"; then
    run_larger_smoke "${LARGER_133_HF_DIR}" "${LARGER_133_PTH}" "13.3b" "${LARGER_133_MAX_NEW_TOKENS}" "${LARGER_133_FAST_TOKEN_BACKEND}"
  else
    echo "SKIP 13.3B larger-model smoke: RUN_133B_MODEL_SMOKE=${RUN_133B_MODEL_SMOKE} LARGER_133_HF_DIR=${LARGER_133_HF_DIR} LARGER_133_PTH=${LARGER_133_PTH}"
  fi

  if should_run_speculative_bench "${RUN_SPECULATIVE_BENCH}" "${SPEC_TARGET_HF_DIR}" "${SPEC_DRAFT_HF_DIR}"; then
    run python bench/bench_speculative_decode.py \
      --target-model "${SPEC_TARGET_HF_DIR}" \
      --draft-model "${SPEC_DRAFT_HF_DIR}" \
      --dtype "${DTYPE}" \
      --device "${DEVICE}" \
      --attn-mode fused_recurrent \
      --max-new-tokens "${SPEC_MAX_NEW_TOKENS}" \
      --draft-tokens "${SPEC_DRAFT_TOKENS}" \
      --results "${RESULTS}"
  else
    echo "SKIP speculative decode benchmark: RUN_SPECULATIVE_BENCH=${RUN_SPECULATIVE_BENCH} SPEC_TARGET_HF_DIR=${SPEC_TARGET_HF_DIR} SPEC_DRAFT_HF_DIR=${SPEC_DRAFT_HF_DIR}"
  fi

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
