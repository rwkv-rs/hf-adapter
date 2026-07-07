# Apple M5 Qwen3.5 0.8B long-context comparison

This evidence directory records a same-device, same-prompt long-context Apple
M5 comparison for the Qwen3.5-over-Apple/mobile goal.

It intentionally does **not** claim a Qwen3.5 win.  It fills the long-context
coverage gap in the goal audit and quantifies the remaining RWKV-7 MLX work.

## Device and runtimes

- Device: Mac17,3 / Apple M5 / 16GB unified memory
- OS: macOS 26.5 arm64
- Python: 3.11.15
- Qwen runtime: `mlx_vlm_token_only`
- RWKV runtime: repository MLX backend, Metal WKV, MM4 quantization
- RWKV env:
  - `RWKV7_MLX_STEP_EVAL_INTERVAL=8`
  - `RWKV7_MLX_GROUP_RKV_QUANT_PROJECTION=1`
  - `RWKV7_MLX_FUSED_FFN_KEY_RELU2=1`

## Shape

- Prompt case: `chars4096`
- Prompt target: 4096 characters
- Requested decode: 128 tokens
- Repeat: 1
- Warmup repeats: 1

## Inputs

- Qwen3.5 0.8B MLX 4bit local path:
  `/Users/wangyue/Documents/vllmsp/models/qwen35-0.8b-mlx-4bit`
- RWKV-7 0.4B HF local path:
  `/Users/wangyue/Documents/vllmsp/models/rwkv7-g1d-0.4b-hf`

## Commands

```bash
PY=/Users/wangyue/Documents/vllmsp/.venv-apple-torch/bin/python
OUTDIR=bench/apple_qwen35_08b_longctx_m5_20260707
mkdir -p "$OUTDIR"

PYTHONPATH=. "$PY" bench/run_qwen35_apple_baseline.py \
  --results "$OUTDIR/results_qwen35_08b_4096_128_token_only.jsonl" \
  --prompt-target-chars 4096 \
  --decode-lengths 128 \
  --repeat 1 \
  --warmup-repeats 1 \
  --qwen-models '' \
  --qwen-mlx-vlm-models /Users/wangyue/Documents/vllmsp/models/qwen35-0.8b-mlx-4bit \
  --qwen-mlx-vlm-token-only \
  --rwkv-mlx-models ''

RWKV7_MLX_STEP_EVAL_INTERVAL=8 \
RWKV7_MLX_GROUP_RKV_QUANT_PROJECTION=1 \
RWKV7_MLX_FUSED_FFN_KEY_RELU2=1 \
PYTHONPATH=. "$PY" bench/run_qwen35_apple_baseline.py \
  --results "$OUTDIR/results_rwkv04_mm4_4096_128_eval8_fused.jsonl" \
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

cat "$OUTDIR"/results_qwen35_08b_4096_128_token_only.jsonl \
    "$OUTDIR"/results_rwkv04_mm4_4096_128_eval8_fused.jsonl \
  > "$OUTDIR/results_combined_4096_128.jsonl"

PYTHONPATH=. "$PY" bench/compare_qwen35_apple_baseline.py \
  --results "$OUTDIR/results_combined_4096_128.jsonl" \
  --pair /Users/wangyue/Documents/vllmsp/models/qwen35-0.8b-mlx-4bit=rwkv7-g1d-0.4b-hf \
  --min-decode-ratio 1.0 \
  --min-prefill-ratio 1.0 \
  --max-ttft-ratio 1.1 \
  --max-memory-ratio 1.0 \
  --require-prefill \
  --require-ttft \
  --require-memory \
  --diagnostics \
  --append "$OUTDIR/results_compare_4096_128.jsonl"

PYTHONPATH=. "$PY" bench/audit_qwen35_apple_goal.py \
  --results "$OUTDIR/results_qwen35_08b_4096_128_token_only.jsonl" \
  --results "$OUTDIR/results_rwkv04_mm4_4096_128_eval8_fused.jsonl" \
  --results "$OUTDIR/results_compare_4096_128.jsonl" \
  --tier 'qwen3.5:0.8b-mlx|mlx-community/Qwen3.5-0.8B-MLX-4bit|qwen35-0.8b-mlx-4bit=rwkv7-g1d-0.4b-hf' \
  --required-shape chars4096:128 \
  --require-quality \
  --require-coreml \
  --append "$OUTDIR/results_goal_audit_4096_128.jsonl"
```

## Result summary

| Metric | Qwen3.5 0.8B MLX 4bit | RWKV-7 0.4B MM4 MLX | RWKV / Qwen |
|---|---:|---:|---:|
| Prompt tokens | 999 | 1054 | - |
| Generated tokens | 128 | 128 | - |
| Prefill tok/s | 1902.372491 | 66.495118 | 0.034954x |
| Decode tok/s | 110.858252 | 52.620777 | 0.474667x |
| TTFT | 0.525134s | 15.851017s | 30.184709x slower |
| Peak memory | 2.266975GB | 0.409776GB | 0.180759x |

## Gate status

- Memory gate: pass.
- Long-context evidence coverage: pass for this 0.8B tier and `chars4096:128`
  shape.
- Speed/latency gate: fail.
  - Decode needs about `2.11x` speedup over the current RWKV row.
  - Prefill needs about `28.61x` speedup over the current RWKV row.
  - TTFT needs the same prefill/first-token reduction work.
- Quality evidence: missing because this run used token-only Qwen generation.
- Stateful CoreML runtime evidence: missing for this tier.

## Engineering implication

This row confirms that Apple MLX memory is already competitive for 0.4B/MM4,
but long-context performance is dominated by prefill.  The next performance
work should target fused prefill/decode kernels and fewer per-token/per-layer
Metal dispatches, not wrapper-only changes.
