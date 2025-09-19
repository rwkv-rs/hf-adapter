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
4. Fused W/A LoRA prototype.
   - `rwkv7_hf.fused_lora.fused_wa_lora()` computes the W/A LoRA pair with
     grouped down/activation and up/bias Triton kernels.
   - `bench/bench_fused_wa_lora.py` records `fused_wa_lora_proto`. The first
     V100 row is correctness-clean but slower, proving two-kernel LoRA grouping
     alone is insufficient and should be fused deeper with R/K/V.
5. Fused W/A/G LoRA prototype.
   - `rwkv7_hf.fused_lora.fused_wag_lora()` expands LoRA grouping to W/A/G,
     including mixed ranks such as W/A rank 64 plus G rank 128.
   - `bench/bench_fused_wag_lora.py` records `fused_wag_lora_proto`. The first
     stable V100 row is correctness-clean and faster than the current W/A/G
     LoRA modules, so this is a useful sub-kernel building block for the next
     combined R/K/V + LoRA fusion target.
6. Fused R/K/V + W/A/G projection prototype.
   - `rwkv7_hf.fused_attention_projection.fused_rkv_wag_projection()` combines
     R/K/V dense projection with W/A/G LoRA down in one launch and W/A/G up in a
     second launch.
   - `bench/bench_fused_rkv_wag_projection.py` records
     `fused_rkv_wag_projection_proto`. The first V100 row is correctness-clean
     and slightly faster, but the gain is small, so the next step is full
     attention fusion or a better dense projection kernel before HF integration.
7. Fused attention output prototype.
   - `rwkv7_hf.fused_output.fused_attn_output_prepare()` fuses group norm over
     recurrent output, recurrent correction, and gate multiply while leaving the
     final `o_proj` on cuBLAS.
   - `bench/bench_fused_attn_output.py` records `fused_attn_output_proto`. The
     first V100 row is correctness-clean and faster than the current output
     prep plus cuBLAS output path, making it a useful target for full attention
     fusion after projection/LoRA and recurrent-state work.
8. Fused FFN prototype.
   - `rwkv7_hf.fused_ffn.fused_ffn()` combines FFN shift-mix, key projection,
     and relu² in one launch, then computes the value projection in a second
     launch.
   - `bench/bench_fused_ffn.py` records `fused_ffn_proto`. The first V100 row is
     correctness-clean but slower than the cuBLAS-backed FFN path, so this
     two-kernel FFN stays telemetry unless it is folded into a larger graph.
9. Fused fp16 attention shift-mix prototype.
   - `rwkv7_hf.fused_time_mix.fused_attn_shift_mix()` provides an optional
     Triton single-launch prototype for the six decode time-mix inputs.
   - `bench/bench_fused_shift_mix.py` records `fused_shift_mix_proto`. The
     first V100 row is exact but slower than the current torch pointwise ops,
     so shift-mix alone should stay telemetry; the next implementation should
     fuse deeper across shift-mix + projection/LoRA/state update.
10. Fused recurrent state update prototype.
   - `rwkv7_hf.fused_recurrent_update.fused_recurrent_update()` exploits the
     rank-1 structure of the RWKV-7 state transition and fuses state update plus
     readout in one Triton launch.
   - `bench/bench_fused_recurrent.py` records `fused_recurrent_proto`. The
     first V100 row is profitable, so the next implementation step is
     correctness-gated native-graph integration.
11. Native-graph integration for the recurrent fused fp16 path.
   - `RWKV7_NATIVE_GRAPH_FUSED_RECURRENT=1` makes native-graph capture use the
     recurrent prototype. The graph-runner cache key includes this flag so
     default and experimental captures cannot be reused accidentally.
   - `bench/bench_native_graph_fused_recurrent.py` records
     `native_graph_fused_recurrent` A/B rows. The first V100 integration row is
     correctness-clean but end-to-end neutral, so the flag remains opt-in while
     deeper projection/LoRA fusion is developed.
12. Native-graph integration for the fused output-prep fp16 path.
   - Native-graph capture now uses the fused attention output-prep kernel by
     default. `RWKV7_NATIVE_GRAPH_FUSED_OUTPUT=0` disables it for A/B or
     fallback testing. The graph-runner cache key includes both recurrent and
     output fusion flags plus the active batch size.
   - `bench/bench_native_graph_fused_output.py` records
     `native_graph_fused_output` A/B rows. The first V100 integration row is
     correctness-clean and moves full token replay latency by about 1.10x. The
     V100 bsz=1/2/4/8 matrix is also correctness-clean with minimum speedup
     above 1.03x, and an idle-V100 greedy batch sweep shows about 1.08-1.10x
     `rwkv7_forward_token` throughput over `RWKV7_NATIVE_GRAPH_FUSED_OUTPUT=0`,
     so output fusion is now the native-graph default. The env flag remains as
     a fallback/A-B switch while the 5070/newer device matrix and combined
     fusion are validated.
12a. Opt-in native-graph probe for fused output-prep plus `o_proj`.
   - `rwkv7_hf.fused_output.fused_attn_output_project()` folds group norm,
     recurrent correction, gate, and the final `o_proj` into one Triton launch.
     It is guarded by `RWKV7_NATIVE_GRAPH_FUSED_OUTPUT_PROJECT=1`; the default
     path still uses fused output-prep plus cuBLAS `o_proj`.
   - `RWKV7_NATIVE_GRAPH_FUSED_OUTPUT_PROJECT_BLOCK_M` controls the output-row
     tile and is part of the graph-runner cache key together with the project
     flag, so captures for different tiles cannot be reused accidentally.
   - `bench/bench_fused_attn_output_project.py` records isolated
     `fused_attn_output_project_proto` rows. V100 bsz=1 shows `1.5965x` over
     the old output path and `1.2931x` over fused-prep+cuBLAS with
     `max_abs_diff=0.001953125`.
   - `bench/bench_native_graph_fused_output_project.py` records full
     `native_graph_fused_output_project` A/B rows. The first V100 bsz=1/2/4/8
     matrix is greedy-exact, but end-to-end speed is only `0.95x`-`0.97x` of
     the default output-fused graph. This proves the one-launch project kernel
     is useful telemetry but not defaultable yet; next work should profile why
     graph capture loses the isolated win and only fold `o_proj` after a better
     occupancy/deeper-fusion kernel exists.
12b. Opt-in native-graph probe for W/A/G LoRA-only fusion.
   - `RWKV7_NATIVE_GRAPH_FUSED_WAG_LORA=1` makes native-graph capture use the
     existing `fused_wag_lora()` two-kernel W/A/G LoRA grouping while leaving
     R/K/V dense projections on cuBLAS. Tile envs
     `RWKV7_NATIVE_GRAPH_FUSED_WAG_LORA_BLOCK_{M,R,K}` are part of the
     graph-runner cache key.
   - `bench/bench_native_graph_fused_wag_lora.py` records
     `native_graph_fused_wag_lora` rows. V100 bsz=1/2/4/8 with `block_m=16`,
     `block_r=64`, `block_k=64` is greedy-exact, but only bsz=8 is slightly
     faster (`1.0059x`) while bsz=1/2/4 are `0.9406x`/`0.9637x`/`0.9615x`.
     This proves isolated W/A/G LoRA fusion does not survive full graph replay
     broadly enough to default; keep it opt-in and fold LoRA into a deeper
     projection/state/output kernel instead.
12c. Default native-graph fused recurrent update plus output-prep.
   - `rwkv7_hf.fused_recurrent_update.fused_recurrent_output_prepare()` fuses
     rank-1 recurrent state update/readout, group norm, recurrent correction,
     and gate multiply into one Triton launch. The final `o_proj` stays on
     cuBLAS.
   - `RWKV7_NATIVE_GRAPH_FUSED_RECURRENT_OUTPUT=1` is now the native-graph
     default for this combined path; set it to `0` for A/B or fallback testing.
     The graph-runner cache key includes the flag, so default output-fused
     runners and recurrent+output runners are isolated.
   - `bench/bench_fused_recurrent_output.py` records
     `fused_recurrent_output_proto` rows. The first V100 row is
     correctness-clean and reaches `1.7956x` versus split fused
     recurrent/output kernels (`4.1916x` versus the torch current path).
   - `bench/bench_native_graph_fused_recurrent_output.py` records
     `native_graph_fused_recurrent_output` A/B rows. V100 bsz=1/2/4/8 is
     greedy-exact and improves full native-graph decode by
     `1.2129x`/`1.1805x`/`1.2416x`/`1.2504x`. A normal batch sweep with the
     flag enabled reaches `332.2`/`589.5`/`1177.9`/`2136.7` aggregate tok/s,
     lifting Albatross decode ratios to min `0.4352x`, max `0.6474x`. This is
     the first deeper fusion that wins both isolated and end-to-end, but it
     is now the default V100 native-graph path while newer-GPU coverage and P1
     min-ratio validation continue.
   - Follow-up flag sweeps with this default path show current opt-in projection
     and W/A/G LoRA probes are still slower: W/A/G LoRA reaches only
     `0.94x`-`0.99x` of default and fused projection reaches `0.84x`-`0.91x`.
     `bench/analyze_results.py` therefore anchors Albatross decode gates to
     default native-graph batch rows and reports experimental flag rows
     separately.

13. Native-graph integration guard for the fused R/K/V + W/A/G projection path.
   - `RWKV7_NATIVE_GRAPH_FUSED_PROJECTION=1` makes native-graph capture use the
     two-kernel `fused_rkv_wag_projection()` prototype. The graph-runner cache
     key includes this flag so default runners are never reused for the probe.
   - `bench/bench_native_graph_fused_projection.py` records
     `native_graph_fused_projection` A/B rows. The first V100 bsz=1/2/4/8
     matrix is correctness-clean, but speed is only 0.86-0.93x of the default
     output-fused graph. This proves the current two-kernel projection grouping
     is not defaultable; it remains an opt-in guard while the next kernel
     attempts fewer launches / better tensor-core occupancy / deeper fusion.
14. Native W8 pack plus fused int8 dequant-GEMV prototype.
   - `rwkv7_hf.native_quant.quantize_int8_rowwise()` packs dense weights as
     signed int8 plus row-wise fp32 scales.
   - `rwkv7_hf.native_quant.int8_rowwise_gemv()` provides an optional Triton
     fused dequant-GEMV prototype with torch fallback.
   - `bench/bench_native_quant_gemv.py` records `native_quant_gemv_proto`. The
     first V100 row proves roughly half fp16 weight footprint and good cosine,
     but it is still slower than fp16 cuBLAS, so the W8 path remains telemetry
     until the kernel is optimized.
15. Native W4 pack plus fused int4 dequant-GEMV prototype.
   - `rwkv7_hf.native_quant.quantize_int4_rowwise()` packs dense weights as
     two signed 4-bit values per byte plus row-wise fp32 scales.
   - `rwkv7_hf.native_quant.int4_rowwise_gemv()` provides an optional Triton
     fused nibble-unpack/dequant-GEMV prototype with torch fallback.
   - `bench/bench_native_quant_w4_gemv.py` records
     `native_quant_w4_gemv_proto`. The first V100 row proves roughly quarter
     fp16 sampled weight footprint, but the prototype is still slower than
     fp16 cuBLAS and needs a better packed reduction / deeper projection fusion
     before it can replace bnb or fp16.
16. Native W8 fused R/K/V quant projection prototype.
   - `rwkv7_hf.native_quant.int8_fused_rkv_gemv()` computes R/K/V from packed
     row-wise W8 weights in one Triton launch.
   - `bench/bench_native_quant_rkv.py` records `native_quant_rkv_proto`. The
     first V100 row improves over three separate W8 dequant-GEMVs, but is still
     below fp16 cuBLAS, so the next quant step is deeper projection/LoRA fusion.
17. Native W4 fused R/K/V quant projection prototype.
   - `rwkv7_hf.native_quant.int4_fused_rkv_gemv()` computes R/K/V from packed
     row-wise W4 weights in one Triton launch.
   - `bench/bench_native_quant_w4_rkv.py` records `native_quant_w4_rkv_proto`.
     The first V100 row improves over three separate W4 dequant-GEMVs, but is
     still below fp16 cuBLAS, so W4 also needs deeper group fusion.
18. Single-load native W8/W4 R/K/V block sweep.
   - `bench/bench_native_quant_rkv_sweep.py` sweeps block sizes after one model
     load with one shared fp16 baseline, avoiding per-config cuBLAS drift.
   - The V100 sweep confirms best W8 (`block_m=64, block_k=128`) is still only
     `0.7873x` fp16 and best W4 (`block_m=8, block_k=64`) is `0.7675x` fp16.
     The quant path therefore needs tensor-core-aware packing or deeper fusion,
     not just block-size tuning.
19. V100 + 5070/newer-GPU benchmark matrix.

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
