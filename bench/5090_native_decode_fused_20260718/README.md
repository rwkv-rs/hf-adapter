# RTX 5090 Native decode precision-matched close

This artifact records the opt-in Native/no-FLA cached-decode close for the
official g1h 7.2B checkpoint on one RTX 5090. It is bound to implementation
commit `cf42c0e43777e4767b70c1cd0f2605c284bb8224`.

## Result

The Native candidate uses FP16 weights and FP32 recurrent-state accumulation.
Against the official Space's matching `fp32io16` state mode, the three-process
median reaches parity at both measured batches:

| Route | B1 tok/s | B8 tok/s | Native/reference |
|---|---:|---:|---:|
| Native fused, median of 3 x 512 steps | 145.06 | 845.57 | - |
| Official v3a, `fp32io16` state | 144.47 | 841.77 | `1.0041x / 1.0045x` |
| Official v3a, fp16 state | 146.28 | 890.21 | `0.9917x / 0.9499x` |

Every Native row reports both requested CUDA extensions as active. All six
512-token runs have the same greedy-trace SHA256
`d70d6c1a89e682ae573f3ecc471eb65d5d2b3c1e90709d2445968808a8339c93`.
The 64-step teacher-forced check against conservative Native graph decode has
minimum logits cosine `0.9999934435`, max absolute difference `0.0625`, and
top-1 agreement `64/64` at B1 and `512/512` at B8.

## Environment

- RTX 5090, SM120, driver `595.58.03`, NVIDIA runtime `13.2`
- PyTorch `2.11.0+cu128`, CUDA `12.8`, Triton `3.6.0`
- Transformers `5.12.1`, Python `3.10.12`
- checkpoint `rwkv7-g1h-7.2b-20260710-ctx10240.pth`, 14,400,007,869 bytes
- official Space commit `cc57df475465c6cacd42ecd4f2f05a588ee5473b`

`ninja` is part of the repository's `cuda` extra. The benchmark is run with
`--require-active-extensions`; a missing compiler dependency or failed CUDA
build terminates the run instead of recording a fallback result.

## Reproduce Native

```bash
python -m pip install -e '.[cuda]'

env \
  RWKV7_NATIVE_MODEL_BACKEND=native_graph \
  RWKV7_NATIVE_GRAPH_ADA_SPARSE_FFN=1 \
  RWKV7_NATIVE_GRAPH_ADA_SPARSE_FFN_OFFICIAL_BOUNDARY=0 \
  RWKV7_NATIVE_GRAPH_ADA_SPARSE_FFN_FP32_ACCUM=1 \
  RWKV7_NATIVE_GRAPH_ADA_SPARSE_FFN_MAX_ROWS=19 \
  RWKV7_NATIVE_GRAPH_ADA_SPARSE_FFN_UP=1 \
  RWKV7_NATIVE_GRAPH_ADA_WAG_LORA=1 \
  RWKV7_NATIVE_GRAPH_ADA_WAGV_LORA=1 \
  RWKV7_NATIVE_GRAPH_ADA_LINEAR=1 \
  RWKV7_NATIVE_GRAPH_ADA_LINEAR_ROWS=1 \
  RWKV7_NATIVE_GRAPH_ADA_LINEAR_ROLES=hidden,ffn_up,ffn_down \
  RWKV7_NATIVE_GRAPH_FUSED_NORM_MIX=1 \
  RWKV7_NATIVE_GRAPH_FUSED_NORM_MIX_NUM_WARPS=8 \
  RWKV7_NATIVE_GRAPH_FUSED_RECURRENT_RAW=1 \
  RWKV7_NATIVE_GRAPH_RKV_POLICY=manual \
  PYTHONPATH=. python bench/bench_native_model_decode.py \
    --hf-dir /absolute/path/to/rwkv7-g1h-7.2b-hf \
    --dtype fp16 --device cuda --prompt-tokens 8 \
    --decode-steps 512 --warmup 5 --batch-sizes 1 8 \
    --backends native_graph --fast-token-api \
    --require-active-extensions --results /tmp/native_decode.jsonl
```

Run the quality gate with:

```bash
PYTHONPATH=. python bench/bench_native_model_decode_alignment.py \
  --hf-dir /absolute/path/to/rwkv7-g1h-7.2b-hf \
  --dtype fp16 --device cuda --batch-sizes 1 8 \
  --prompt-tokens 8 --steps 64 --min-cosine 0.9999 \
  --results /tmp/native_decode_alignment.jsonl
```

## Reproduce the official reference

From the pinned official Space clone:

```bash
python rwkv7_fast_v3a.py \
  --model /absolute/path/to/rwkv7-g1h-7.2b-20260710-ctx10240.pth \
  --wkv fp32io16 --emb gpu --batched-rkv off \
  --cmix-sparse no-fc --lowrank-weight both \
  --cases 1x1,8x1 --warmup 5 --iters 20
```

Use `--wkv fp16` only for the separate lower-precision reference. It must not
be described as a precision-matched comparison.

## Evidence map and limits

- `native_rep*_raw.jsonl`: strict 512-step rows with extension status
- `native_repeats.jsonl`: compact three-repeat summary
- `logits_alignment.jsonl`: repository-script quality gate
- `official_fp32_state.log` / `official_fp16_state.log`: fresh same-card v3a
- `kernel_ab.jsonl`, `b1_dispatch_ab.jsonl`, `b8_quality_speed_ab.jsonl`, and
  `warps.jsonl`: route selection and rejected-boundary telemetry
- `environment.json` and `summary.json`: machine-readable environment/result

This does not promote a global default. It does not cover prefill, another
checkpoint, another card, memory parity, or training speed. The official
fp16-state route remains faster, particularly at B8. Half-atomic sparse routes
that showed batch divergence are retained only as negative evidence; the
accepted route uses FP32 scratch accumulation and keeps every new flag off by
default.
