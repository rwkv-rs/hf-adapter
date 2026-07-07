# Apple M5 MLX chunked-prefill state-only seam

This evidence directory records the first MLX chunked-prefill cleanup after the
Qwen3.5 long-context audit identified prefill/TTFT as the main Apple gap.

The change adds `MLXRWKV7Model.prefill_state_only()` and routes all non-final
`chunked_prefill()` chunks through it.  Intermediate chunks only need to advance
RWKV recurrent state; they do not need final layer norm + `lm_head` logits.
The final chunk still uses the normal `forward(..., collect_all=False)` path so
full/chunked logits remain directly comparable.

This is a correctness/serving-shape improvement, not a claim that the Qwen3.5
speed gap is closed.  On long contexts it removes only the chunk-boundary logits
projections; the dominant cost remains per-token/per-layer WKV and projection
dispatch.

## Device and environment

- Device: Mac17,3 / Apple M5 / 16GB unified memory
- OS: macOS 26.5 arm64
- Python: 3.11.15
- MLX: 0.31.2
- RWKV env:
  - `RWKV7_MLX_STEP_EVAL_INTERVAL=8`
  - `RWKV7_MLX_GROUP_RKV_QUANT_PROJECTION=1`
  - `RWKV7_MLX_FUSED_FFN_KEY_RELU2=1`

## Commands

```bash
PY=/Users/wangyue/Documents/vllmsp/.venv-apple-torch/bin/python
OUTDIR=bench/apple_mlx_chunked_state_only_m5_20260707

PYTHONPATH=. "$PY" tests/test_apple_silicon_mlx_model_smoke.py \
  --require-mlx \
  --results "$OUTDIR/results_tiny_smoke.jsonl"

RWKV7_MLX_STEP_EVAL_INTERVAL=8 \
RWKV7_MLX_GROUP_RKV_QUANT_PROJECTION=1 \
RWKV7_MLX_FUSED_FFN_KEY_RELU2=1 \
PYTHONPATH=. "$PY" bench/run_qwen35_apple_baseline.py \
  --results "$OUTDIR/results_rwkv04_mm4_4096_128_state_only_chunked.jsonl" \
  --prompt-target-chars 4096 \
  --decode-lengths 128 \
  --repeat 1 \
  --warmup-repeats 1 \
  --qwen-models '' \
  --qwen-mlx-vlm-models '' \
  --rwkv-mlx-models /Users/wangyue/Documents/vllmsp/models/rwkv7-g1d-0.4b-hf \
  --rwkv-dtype fp16 \
  --rwkv-quantization mm4 \
  --rwkv-quant-min-params 4000000 \
  --rwkv-quant-rkv-min-params 0 \
  --rwkv-quant-backend auto \
  --rwkv-wkv-backend auto \
  --rwkv-chunk-size 512
```

## Correctness smoke

`results_tiny_smoke.jsonl` passes the tiny MLX recurrent smoke and records:

- `chunked_prefill_max_abs=0.0`
- `select_batch_decode_max_abs=0.0014168`
- `state_only_prefill_calls=1`
- `state_only_prefill_tokens=2`

## Long-context row

`results_rwkv04_mm4_4096_128_state_only_chunked.jsonl` records the same
`4096 chars / 128 tokens` RWKV 0.4B/MM4 shape used by the Qwen3.5 0.8B
long-context audit.

Key fields from the row:

| Field | Value |
|---|---:|
| `chunked_prefill_max_abs` | `0.0` |
| `chunked_state_only_prefill_calls` | `2` |
| `chunked_state_only_prefill_tokens` | `1024` |
| `chunked_quantized_linear_last_backend_counts.metal` | `126481` |
| `chunked_group_rkv_quant_projection_counts.metal` | `25296` |
| `chunked_wkv_backend_counts.metal` | `25296` |

The previous long-context row in
`bench/apple_qwen35_08b_longctx_m5_20260707` used the older chunked path and
reported `chunked_quantized_linear_last_backend_counts.metal=126483` for the
same 512-token chunking.  The new row therefore removes the two non-final
chunk-boundary logits projections while preserving exact final logits.

## Engineering implication

This confirms that chunked prefill now has the correct production shape: state
only for intermediate chunks, logits only at the final boundary.  The measured
long-context speed gap remains dominated by the recurrent token loop and fused
projection/WKV dispatch count, so the next meaningful Apple work is still:

1. fused multi-token WKV / recurrent scan for MLX or CoreML,
2. broader projection grouping/fusion beyond R/K/V and FFN key+relu²,
3. stateful CoreML decode/prefill runtime rows.
