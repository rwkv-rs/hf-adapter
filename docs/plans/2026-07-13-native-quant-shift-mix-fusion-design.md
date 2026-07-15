# Native quant FFN shift-mix fusion design

## Goal

Reduce RTX 5070 small-batch decode overhead by extending the default-off MM8
FFN-up kernel boundary from:

```text
fk = h2 + (previous - h2) * mix
hidden = relu(mm8_key(fk)) ** 2
```

to a single quantized key projection kernel that reads `h2`, `previous`, and
`mix`, computes the mixed activation in registers, dequantizes/projects the
MM8 weight, and applies ReLU-square.

## Alternatives

1. Reuse the existing attention-add + LayerNorm + shift-mix kernel. An exact
   5070 paired probe was run first. It did not close the gap: MM8 deep reached
   `0.9646x/0.9092x/1.0562x/1.0233x` fp16 at bsz1/2/4/8.
2. Fuse shift-mix into MM8 key GEMV. This removes the `fk` materialization and
   its pointwise launches without changing quantization or recurrence. This was
   implemented as the selected prototype and then rejected after end-to-end A/B.
3. Fuse both key and value projections into one persistent kernel. A full
   `4H` intermediate cannot be shared across output tiles without global
   synchronization; recomputing it is not a useful optimization. This remains
   rejected until a proven persistent-GEMM design is available.

## Runtime contract

- Add a separate `RWKV7_NATIVE_GRAPH_FUSED_QUANT_FFN_SHIFT_MIX=1` flag. It
  requires `RWKV7_NATIVE_GRAPH_FUSED_QUANT_FFN=1` and remains default-off.
- Extend scalar, batched GEMV, and Blackwell tensor-core MM8 kernels with a
  compile-time shift-mix epilogue input boundary.
- Keep `previous.copy_(h2)` after the projection kernel. Writing previous from
  one output tile while other tiles still read it would introduce a race.
- CPU, unsupported devices, bias-bearing linears, disabled flags, and dense
  paths retain portable reference behavior.
- Include the flag in the native-graph cache key.

## Validation

- CPU fallback must match `relu(mm8(h + (previous-h)*mix))**2`.
- CUDA synthetic A/B covers 2048->8192, fp16, bsz1/2/4/8, cosine, max error,
  and latency.
- End-to-end 5070 rows must use paired fp16 baselines and report footprint,
  final-logits cosine, and greedy token.
- No promotion unless every claimed cell is non-regressing; a microbenchmark
  win alone is insufficient.

## Outcome

The fused shift-mix kernel passed correctness and improved the isolated
2048->8192 key path, but did not improve the tuned full model. At the original
tile it produced bsz1/2 end-to-end ratios of `0.9702x/0.9001x` fp16. After the
MM8 tile was tuned, shift-deep was slower than deep-only at bsz1 and provided
no stable reason to retain the runtime path. The implementation was removed.

The useful result came from the adjacent layout sweep: Blackwell's previous
`128x128` scalar/small-batch GEMV tile was a poor fit for the 1.5B FFN. An
exact RTX 5070 `64x256` tile reduced isolated complete-FFN latency from about
`0.181` to `0.097 ms` at bsz1 and from `0.198` to `0.121 ms` at bsz2. Combined
with the already-existing deep MM8 epilogues, the seven-cell end-to-end matrix
reached `1.0765x-1.1548x` fp16. The production change therefore keeps the
tile override and exact-card default, not the negative shift-mix fusion.
