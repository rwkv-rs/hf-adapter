# Native quant FFN deep-fusion design

## Goal

Extend the default-off native quant FFN route from an isolated
`key projection + ReLU-square` epilogue into the complete decode FFN boundary:

1. MM8 key dequant-GEMV plus ReLU-square;
2. MM8 value dequant-GEMV plus residual add.

The first stage already exists. This change adds the second stage and routes
both stages together under `RWKV7_NATIVE_GRAPH_FUSED_QUANT_FFN=1`.

## Chosen boundary

RWKV FFN computes:

```text
hidden = relu(key(x)) ** 2
output = residual + value(hidden)
```

The selected implementation keeps two matrix-multiplication kernels but folds
each adjacent pointwise epilogue into its producer. It removes the standalone
ReLU-square and residual-add launches without materializing extra dense
weights.

A literal one-kernel key/value implementation is rejected. Output-tiled GEMV
programs cannot share the full `4H` key activation across output tiles without
a global synchronization boundary. Recomputing key activations for every
value-output tile multiplies the key projection work and is not a valid speed
optimization. A cooperative single-block implementation would expose too
little parallelism for 1.5B and larger FFNs.

## Runtime contract

- The existing environment flag remains default-off.
- Dense, MM4, A8W8, CPU, unsupported shapes, and disabled flags keep their
  current behavior.
- MM8 kernels accept an optional residual pointer and compile-time epilogue
  flag for scalar, batched GEMV, and Blackwell dot paths.
- `MM8Linear.rwkv7_forward_add()` provides the graph dispatch contract and a
  portable fallback.
- The native-graph cache key already includes the fused-quant-FFN flag.

## Validation

- CPU fallback must exactly equal `residual + module(x)`.
- Dispatch tests must prove the method is called only when the flag is enabled.
- CUDA synthetic rows cover MM8 `2048 -> 8192 -> 2048`, bsz 1/2/4/8, fp16,
  output cosine, max absolute error, separate/up-only/deep timing, and payload.
- End-to-end rows cover 1.5B first, then 2.9B and 7.2B where memory permits.
- RTX 5070 (`sm_120`) decides Blackwell telemetry; V100 (`sm_70`) separately
  decides Volta telemetry. No cross-card default promotion is allowed.

## MM4 matrix

MM4 already closes the first V100 1.5B bsz1 end-to-end row. The next matrix
expands model size, batch, prompt, and decode coverage without changing MM4's
default policy. Local 8GB runs may omit OOM fp16 baselines; the V100 32GB
matrix remains authoritative for complete paired rows.
