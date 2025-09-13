# RWKV-7 HF Fused Backend Goal

This phase tracks the performance work that turns the existing HF-compatible
RWKV-7 adapter into a fused native backend while keeping the public entrypoints
inside the HF wrapper.

## Scope

The fused backend is not a separate inference engine. It must be reachable from
standard HF-facing paths:

- `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`
- `model.generate(..., use_cache=True)`
- `model.rwkv7_forward_token(...)`
- `RWKV7StateCache` dynamic-batch and chunked-prefill helpers

The current wrapper/native split remains intact:

- HF wrapper owns compatibility with Transformers, PEFT, Trainer, TRL, cache
  semantics, quantization loading, and benchmark/report gates.
- `native_jit` / `native_graph` / future fused kernels are performance backends
  selected by runtime dispatch.
- `native_model` remains experimental until it reaches the same compatibility
  and benchmark surface.

## Albatross target ladder

Current committed V100 0.1B evidence shows HF native-graph decode at roughly
`0.32x`-`0.47x` Albatross and B=1,T=512 prefill at roughly `0.316x` Albatross.
The staged target is:

| Stage | Decode target | Prefill target | Meaning |
|---|---:|---:|---|
| P1 | `>=0.55x` Albatross | `>=0.60x` Albatross | fused backend is clearly working |
| P2 | `>=0.75x` Albatross | `>=0.80x` Albatross | close enough for serious bounty review |
| P3 | `>=0.90x` Albatross | follow measured bottlenecks | near-Albatross HF path |

The analyzer reports this under `fused_backend_targets` so progress is visible
from `bench/results.jsonl` instead of living only in notes.

## Quantized backend targets

Generic bitsandbytes remains a compatibility baseline, not the final fast path.
Production quantized inference needs RWKV-native packing and fused dequant GEMV:

| Mode | Footprint target | Speed target |
|---|---:|---:|
| W8 | `<=0.75x` fp16 footprint | decode `>=1.0x` fp16 reference |
| W4 | `<=0.55x` fp16 footprint | decode `>=1.0x` fp16 reference |

V100 is the first regression baseline. Newer Ada/Blackwell-class cards should be
used to validate that W8/W4 can eventually approach or beat fp16 native-graph
serving speed.

## Planned PR sequence

1. Fused-backend target/reporting gate.
2. Matrix-level projection/LoRA profiler and candidate shapes.
   - `bench/bench_projection_lora.py` emits `sample_matrix_profile`,
     `sample_matrix_profile_summary`, and `fused_kernel_plan`.
   - `bench/analyze_results.py` surfaces the first fused fp16 target in
     `projection_lora` and `next_focus`.
3. Fused fp16 projection prototype.
   - `rwkv7_hf.fused_projection.fused_rkv_projection()` provides an optional
     Triton single-launch R/K/V GEMV prototype with torch fallback.
   - `bench/bench_fused_projection.py` records correctness and speed telemetry
     as `fused_projection_proto`. The first V100 prototype is correct but still
     slower than three cuBLAS-backed linears, so it is not integrated into the
     HF fast path yet.
4. Fused fp16 attention shift-mix prototype.
   - `rwkv7_hf.fused_time_mix.fused_attn_shift_mix()` provides an optional
     Triton single-launch prototype for the six decode time-mix inputs.
   - `bench/bench_fused_shift_mix.py` records `fused_shift_mix_proto`. The
     first V100 row is exact but slower than the current torch pointwise ops,
     so shift-mix alone should stay telemetry; the next implementation should
     fuse deeper across shift-mix + projection/LoRA/state update.
5. Fused recurrent state update prototype.
   - `rwkv7_hf.fused_recurrent_update.fused_recurrent_update()` exploits the
     rank-1 structure of the RWKV-7 state transition and fuses state update plus
     readout in one Triton launch.
   - `bench/bench_fused_recurrent.py` records `fused_recurrent_proto`. The
     first V100 row is profitable, so the next implementation step is
     correctness-gated native-graph integration.
6. Native-graph integration for the recurrent fused fp16 path.
   - `RWKV7_NATIVE_GRAPH_FUSED_RECURRENT=1` makes native-graph capture use the
     recurrent prototype. The graph-runner cache key includes this flag so
     default and experimental captures cannot be reused accidentally.
   - `bench/bench_native_graph_fused_recurrent.py` records
     `native_graph_fused_recurrent` A/B rows. The first V100 integration row is
     correctness-clean but end-to-end neutral, so the flag remains opt-in while
     deeper projection/LoRA fusion is developed.
7. Native W8 pack plus fused int8 dequant-GEMV prototype.
   - `rwkv7_hf.native_quant.quantize_int8_rowwise()` packs dense weights as
     signed int8 plus row-wise fp32 scales.
   - `rwkv7_hf.native_quant.int8_rowwise_gemv()` provides an optional Triton
     fused dequant-GEMV prototype with torch fallback.
   - `bench/bench_native_quant_gemv.py` records `native_quant_gemv_proto`. The
     first V100 row proves roughly half fp16 weight footprint and good cosine,
     but it is still slower than fp16 cuBLAS, so the W8 path remains telemetry
     until the kernel is optimized.
8. Native W4 pack plus fused int4 dequant-GEMV prototype.
   - `rwkv7_hf.native_quant.quantize_int4_rowwise()` packs dense weights as
     two signed 4-bit values per byte plus row-wise fp32 scales.
   - `rwkv7_hf.native_quant.int4_rowwise_gemv()` provides an optional Triton
     fused nibble-unpack/dequant-GEMV prototype with torch fallback.
   - `bench/bench_native_quant_w4_gemv.py` records
     `native_quant_w4_gemv_proto`. The first V100 row proves roughly quarter
     fp16 sampled weight footprint, but the prototype is still slower than
     fp16 cuBLAS and needs a better packed reduction / deeper projection fusion
     before it can replace bnb or fp16.
9. V100 + 5070/newer-GPU benchmark matrix.

## Backend dispatch requirement

Fast paths must be optional and hardware-aware:

```text
if native fused quant CUDA is available and supported:
    native_quant_cuda
elif fused fp16 CUDA/Triton is available and supported:
    native_fused_fp16
elif native_graph is available:
    native_graph
elif native_jit is available:
    native_jit
else:
    FLA / PyTorch fallback
```

The project can claim broad hardware support only through this fallback stack.
It must not claim the same peak speed on every GPU generation.
