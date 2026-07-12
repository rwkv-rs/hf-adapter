# Native fused MM8/MM4 FFN design

## Scope

Advance the CUDA native MM8/MM4 path at the full-memory policy's hottest
boundary: the FFN key projection followed by `relu(x) ** 2`. The current
`native_graph` path launches a quantized dequant-GEMV and then a separate
pointwise kernel for ReLU-square. The new path fuses that epilogue into the
quantized projection kernel while preserving the existing module formats,
fallbacks, and HF-facing behavior.

This is deliberately narrower than a complete quantized RWKV block. Existing
R/K/V fused quant functions are isolated telemetry and do not cover the FFN
matrices selected by the default 8M-parameter memory policy on 1.5B and 2.9B
models. FFN key fusion therefore gives the next implementation a real
end-to-end integration point without changing quantization policy or defaults.

## Alternatives considered

1. Integrate the existing row-wise W8/W4 R/K/V microbench kernels. This reduces
   three launches to one, but it uses a different packing format from
   `MM8Linear`/`MM4Linear` and misses the main quantized FFN modules on smaller
   checkpoints.
2. Build a complete quantized FFN key-plus-value kernel. This has higher upside,
   but the intermediate `relu2(key)` vector is consumed by a second matrix and
   requires a substantially larger persistent/tiled design. It is too large for
   the first production integration.
3. Fuse the ReLU-square epilogue into existing MM8/MM4 kernels. This is the
   selected approach: it reuses current packing, removes one launch per layer,
   works inside CUDA graph capture, and provides a measurable base for deeper
   FFN fusion.

## Runtime contract

- Add `MM8Linear.rwkv7_forward_relu2()` and
  `MM4Linear.rwkv7_forward_relu2()`.
- CPU, non-CUDA, unsupported dtype, and biased modules preserve semantics using
  `torch.relu(module(x)) ** 2`.
- Triton MM8/MM4 kernels receive a compile-time ReLU-square epilogue flag.
- The exact-sm70 MM4 extension receives a dedicated ReLU-square entry point.
- `native_jit._native_graph_ffn_up_relu2_dispatch()` calls the fused method only
  when `RWKV7_NATIVE_GRAPH_FUSED_QUANT_FFN=1`.
- The flag is included in the native-graph runner cache key. It defaults off in
  every GPU policy until exact-card A/B rows prove non-negative end-to-end
  value.

## Validation

- CPU tests cover method fallback and opt-in native-graph routing.
- Existing MM8/MM4 CUDA correctness tests compare fused ReLU-square against the
  separate quantized projection and pointwise expression.
- A synthetic benchmark records separate/fused latency, cosine, max error,
  batch size, shape, backend, and exact GPU identity.
- V100 promotion requires bsz 1/2/4/8 correctness plus end-to-end decode A/B.
  Isolated speedup alone does not change defaults or support a quantized-speed
  claim.

