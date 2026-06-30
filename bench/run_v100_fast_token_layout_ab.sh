#!/usr/bin/env bash
set -euo pipefail

HF_DIR=${HF_DIR:-/home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf}
PTH=${PTH:-/home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-20260129-ctx8192.pth}
RESULTS=${RESULTS:-bench/results.jsonl}
DTYPE=${DTYPE:-fp16}
DEVICE=${DEVICE:-cuda}
PROMPT_TOKENS=${PROMPT_TOKENS:-512}
DECODE_TOKENS=${DECODE_TOKENS:-128}
MICRO_PROMPT_TOKENS=${MICRO_PROMPT_TOKENS:-128}
MICRO_STEPS=${MICRO_STEPS:-128}

export PYTHONNOUSERSITE=${PYTHONNOUSERSITE:-1}
export RWKV_V7_ON=${RWKV_V7_ON:-1}
export RWKV7_FAST_CACHE=${RWKV7_FAST_CACHE:-1}

for layout in 3d 2d; do
  echo "===== fast-token layout: ${layout} correctness ====="
  RWKV7_FAST_TOKEN_LAYOUT=${layout} \
  python tests/test_fast_decode_api.py \
    --model "${HF_DIR}" \
    --dtype "${DTYPE}" \
    --device "${DEVICE}" \
    --fuse-norm false \
    --batch-sizes 1 2 4 \
    --fast-token-layouts "${layout}" \
    --decode-steps 32 \
    --max-diff 0.2

  echo "===== fast-token layout: ${layout} speed_mem ====="
  RWKV7_FAST_TOKEN_LAYOUT=${layout} \
  python bench/bench_speed.py \
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
    --fuse-norm false \
    --fast-cache true \
    --hf-decode-api rwkv7_forward_token \
    --fast-token-layout "${layout}" \
    --results "${RESULTS}"

  echo "===== fast-token layout: ${layout} micro ====="
  RWKV7_FAST_TOKEN_LAYOUT=${layout} \
  python bench/bench_decode_micro.py \
    --hf-dir "${HF_DIR}" \
    --dtype "${DTYPE}" \
    --device "${DEVICE}" \
    --attn-mode fused_recurrent \
    --fuse-norm false \
    --fast-cache true \
    --fast-decode-api auto \
    --fast-token-layout "${layout}" \
    --prompt-tokens "${MICRO_PROMPT_TOKENS}" \
    --warmup 8 \
    --steps "${MICRO_STEPS}" \
    --results "${RESULTS}"
done

python bench/compare_fast_token_layouts.py --results "${RESULTS}" --device V100 --dtype "${DTYPE}" --require-candidate --min-speedup 1.0
python bench/analyze_results.py --results "${RESULTS}" --device V100 --dtype "${DTYPE}"
python bench/check_results.py --results "${RESULTS}" --device V100 --dtype "${DTYPE}"
