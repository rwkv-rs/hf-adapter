# Apple M5 MLX decode synchronization cleanup and attn-mix probe

This evidence directory records a small serving-harness cleanup for the
Apple/Qwen3.5 performance lane plus one rejected opt-in fusion probe.

## What changed

`MLXRWKV7Model.forward()` already synchronizes returned logits and recurrent
state before returning from `prefill()` / `decode_step()` / `chunked_prefill()`.
The Apple baseline harness was adding another `mx.eval(logits)` immediately
after those calls, then synchronizing `next_token` again for greedy decode.

The harness now avoids the redundant logits barriers and keeps only the required
`mx.eval(next_token)` sync for streaming token production.  This better matches
the serving shape: the host waits for the next token, not for an extra explicit
logits synchronization before computing that token.

This PR also adds an opt-in `RWKV7_MLX_FUSED_ATTN_MIX=1` Metal seam that fuses
six RWKV-7 attention mix expressions (`xr/xw/xk/xv/xa/xg`) into one kernel.  The
AB row shows it is not a default win yet, so the one-command Apple acceptance
wrapper leaves it disabled by default (`RWKV_FUSED_ATTN_MIX=0`).

## Device and environment

- Device: Mac17,3 / Apple M5 / 16GB unified memory
- OS: macOS 26.5 arm64
- Python: 3.11.15
- Shape: `512 chars / 64 tokens`
- RWKV model: `/Users/wangyue/Documents/vllmsp/models/rwkv7-g1d-0.4b-hf`
- RWKV env:
  - `RWKV7_MLX_STEP_EVAL_INTERVAL=8`
  - `RWKV7_MLX_GROUP_RKV_QUANT_PROJECTION=1`
  - `RWKV7_MLX_FUSED_FFN_KEY_RELU2=1`

## Commands

```bash
PY=/Users/wangyue/Documents/vllmsp/.venv-apple-torch/bin/python
OUTDIR=bench/apple_mlx_decode_sync_m5_20260707

RWKV7_MLX_STEP_EVAL_INTERVAL=8 \
RWKV7_MLX_GROUP_RKV_QUANT_PROJECTION=1 \
RWKV7_MLX_FUSED_FFN_KEY_RELU2=1 \
RWKV7_MLX_FUSED_ATTN_MIX=0 \
PYTHONPATH=. "$PY" bench/run_qwen35_apple_baseline.py \
  --results "$OUTDIR/results_rwkv04_512_64_sync_cleanup.jsonl" \
  --prompt-target-chars 512 \
  --decode-lengths 64 \
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
  --rwkv-chunk-size 256

RWKV7_MLX_STEP_EVAL_INTERVAL=8 \
RWKV7_MLX_GROUP_RKV_QUANT_PROJECTION=1 \
RWKV7_MLX_FUSED_FFN_KEY_RELU2=1 \
RWKV7_MLX_FUSED_ATTN_MIX=1 \
PYTHONPATH=. "$PY" bench/run_qwen35_apple_baseline.py \
  --results "$OUTDIR/results_rwkv04_512_64_sync_cleanup_attn_mix.jsonl" \
  --prompt-target-chars 512 \
  --decode-lengths 64 \
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
  --rwkv-chunk-size 256
```

## Result summary

| Row | Prefill tok/s | Decode tok/s | TTFT | Notes |
|---|---:|---:|---:|---|
| sync cleanup, attn mix off | `67.170911` | `53.203176` | `1.980235s` | acceptance default shape |
| sync cleanup, attn mix on | `71.350228` | `51.868772` | `1.864285s` | `fused_attn_mix_counts.metal=4728` |

The opt-in attention-mix Metal seam is correct and observable, but the decode
regression means it should stay off by default until deeper profiling shows a
consistent win across prompt/decode shapes.

## Engineering implication

The synchronization cleanup removes a harness-side barrier and makes decode
telemetry closer to streaming serving behavior.  The attention-mix probe shows
that tiny elementwise fusion alone is not the main Apple bottleneck; the next
high-impact work remains fused multi-token WKV/recurrent scan and broader
projection fusion.
