# RTX 5090 FP16-state Native decode acceptance

This artifact closes the measured 7.2B, FP16-state cached-decode gap against
the pinned official `rwkv7_fast_v3a.py` implementation on one RTX 5090. The
candidate remains opt-in and default-off.

## Accepted result

| Shape | Native, median of 3 x 512 | Official p50 | Native / official |
|---|---:|---:|---:|
| B1/T1 | 146.30 tok/s | 146.277184 tok/s | 1.00016x |
| B8/T1 | 892.28 tok/s | 890.210359 tok/s | 1.00232x |

All six Native rows report `ada_sparse_ffn`, `ada_lora`, and
`native_wkv_fp16` as requested and active. The three trace hashes agree within
each batch shape, and all eight identical B8 inputs retain identical 512-token
greedy traces. See `summary.json` for the fail-closed machine summary and
`repeats/` for the raw rows.

The matched official source is commit
`cc57df475465c6cacd42ecd4f2f05a588ee5473b`, WKV mode `fp16`, 20 timing
iterations, from `../5090_native_decode_fused_20260718/official_fp16_state.log`.
Native timing covers the public end-to-end token loop, including graph replay
and greedy argmax.

## Final route

The route combines an FP16 recurrent-state CUDA kernel with a deterministic
four-way B8 sparse FFN reduction. Each split preserves the official per-tile
FP16 round boundary, then the four partials are added in fixed order. This
removes cross-block half atomics while retaining enough Blackwell parallelism.

The faster official-style half-atomic probe reached approximately 927-939
tok/s at B8, but all three processes produced two different greedy traces for
identical batch rows. It is rejected evidence and is not the accepted route.
The previous FP32 scratch path remains the default-safe implementation.

## Reproduce

From an editable CUDA install of this repository:

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export RWKV7_NATIVE_GRAPH_STATE_DTYPE=fp16
export RWKV7_NATIVE_GRAPH_FP16_RECURRENT=1
export RWKV7_NATIVE_GRAPH_PRECOMPUTE_EMB_LN0=1
export RWKV7_NATIVE_GRAPH_RKV_POLICY=manual
export RWKV7_NATIVE_GRAPH_FUSED_NORM_MIX=1
export RWKV7_NATIVE_GRAPH_FUSED_NORM_MIX_NUM_WARPS=8
export RWKV7_NATIVE_GRAPH_ADA_LINEAR=1
export RWKV7_NATIVE_GRAPH_ADA_LINEAR_ROWS=1
export RWKV7_NATIVE_GRAPH_ADA_LINEAR_ROLES=hidden,ffn_up,ffn_down
export RWKV7_NATIVE_GRAPH_ADA_SPARSE_FFN=1
export RWKV7_NATIVE_GRAPH_ADA_SPARSE_FFN_MAX_ROWS=19
export RWKV7_NATIVE_GRAPH_ADA_SPARSE_FFN_UP=1
export RWKV7_NATIVE_GRAPH_ADA_SPARSE_FFN_FP32_ACCUM=0
export RWKV7_NATIVE_GRAPH_ADA_SPARSE_FFN_OFFICIAL_BOUNDARY=1
export RWKV7_NATIVE_GRAPH_ADA_SPARSE_FFN_DETERMINISTIC_SPLITS=4
export RWKV7_NATIVE_GRAPH_ADA_WAG_LORA=1
export RWKV7_NATIVE_GRAPH_ADA_WAGV_LORA=1

python bench/bench_native_model_decode.py \
  --hf-dir /absolute/path/to/rwkv7-g1h-7.2b-hf \
  --dtype fp16 --device cuda --prompt-tokens 8 \
  --decode-steps 512 --warmup 8 --batch-sizes 1 8 \
  --backends native_graph --fast-token-api --require-active-extensions \
  --results /tmp/native-decode.jsonl
```

Regenerate the strict summary with `bench/summarize_native_official_decode.py`.
A missing repetition, inactive requested extension, batch divergence, slow
matched shape, wrong device/dtype, or wrong decode length fails the command.

## Scope

This closes only the exact RTX 5090 / g1h 7.2B / FP16 / B1+B8 / T1 decode
lane. It does not promote a default, prove other checkpoints or GPUs, close
prefill, or replace the FP32 recurrent-state compatibility contract.
