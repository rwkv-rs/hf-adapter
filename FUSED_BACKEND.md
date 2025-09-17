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
   - RTX 4090 / Ada (`sm_89`) validation now uses
     `bench/run_4090_fused_backend_validation.sh`. On 0.4B fp16, the default
     fused recurrent+output path is greedy-exact across bsz=1/2/4/8 and improves
     full native-graph decode by `1.2408x`/`1.1981x`/`1.2268x`/`1.2226x`
     versus the output-only baseline. The same run confirms the current
     opt-in cuBLAS-replacement probes should stay off on Ada too:
     WAVG-LoRA is `0.9496x`/`0.9963x`/`0.9973x` for bsz=1/4/8, fused projection
     is `0.9407x` at bsz=4, and fused output-project is `0.9665x` at bsz=4.
     The Ada rule therefore matches the 5070/V100 evidence: keep fusing the
     state-update/output-prep/norm work that Triton handles well, but do not
     replace cuBLAS GEMV/GEMM subpaths unless a new kernel proves end-to-end
     speedup under `native_graph`.
   - The native-graph runner now marks a `RWKV7StateCache` as bound to the
     runner after replay. On the next token, if the cache was not mutated by
     `update`, `reset`, `detach`, `to`, `select_batch`, or `reorder_cache`,
     `copy_from_cache()` and `bind_cache()` skip the per-layer state traversal
     entirely instead of checking/rewriting 3 tensors per layer. The normal
     cache APIs invalidate the binding, so dynamic batching and HF fallback
     semantics stay conservative. On RTX 4090 0.4B fp16, the overhead rows show
     `copy_from_cache_fast_skip_rate=0.9844`,
     `bind_cache_fast_skip_rate=0.9844`, measured `copy_from_cache_ms` of
     `0.0389`/`0.0359`/`0.0358`, and measured `bind_cache_ms` of
     `0.0030`/`0.0024`/`0.0022` for bsz=1/4/8 (`copy_share_of_manual_wall`
     about 1%). Public `rwkv7_forward_token` batch sweep after the skip reaches
     `395.6`/`1143.4`/`2257.5` aggregate tok/s for bsz=1/4/8.
     `test_fast_decode_api` with native_graph bsz=1/2/4 still passes greedy,
     sequence-length, and fallback compatibility checks.
   - Dynamic batching telemetry is now part of `bench/bench_dynamic_batch.py`.
     The benchmark records state-cache select counters, native-graph LRU
     requests/hits/misses, active batch sizes, and runner copy/bind fast-skip
     rates. `select_batch()` now keeps the cache bound when the active batch is
     only reordered (same size): it reorders the captured graph buffers in place
     and falls back to tensor selection only when rows are dropped. On RTX 4090
     0.4B fp16 with initial bsz=8, `reorder_every=4`, `drop_every=32`, and final
     bsz=4, the HF forward path reaches `191.9` tok/s while
     `rwkv7_forward_token` + native_graph reaches `1671.8` tok/s. The timed
     native-graph decode has `128/128` runner-cache hits (`hit_rate=1.0`) across
     active batch sizes `[2,3,4,5,6,7,8]`, `28/32` selects preserve the bound
     graph buffers, and copy/bind fast-skip rates rise to `0.9688`. The only
     remaining full state copies are the four expected active-size drops.
     `tests/test_dynamic_batch_cache.py` now also covers an in-place same-size
     reorder under `fast_token`; the 4090 0.1B native_graph smoke remains within
     `0.09375` max-abs diff and greedy equality for the reordered/compacted
     steps.

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
   - The W4 Triton kernels now iterate over packed int4 bytes and consume both
     nibbles per load instead of loading the same packed byte once per logical
     input feature. For W4 rows the benchmark reports `block_k_unit` as
     `packed_int4_bytes`; W8 keeps `block_k_unit=input_features`.
   - RTX 4090 / Ada 0.4B fp16 sweep evidence is now recorded for the same R/K/V
     group. Best W8 reaches `0.7125x` fp16 (`0.05051ms` fused vs `0.03599ms`
     fp16, footprint ratio `0.502`) and best W4 reaches `0.6958x` fp16
     (`0.05157ms` fused vs `0.03588ms` fp16, footprint ratio `0.252`). Both
     fused quant paths are roughly `1.88x`-`1.92x` faster than three separate
     quant GEMVs, but still below fp16 cuBLAS, so 4090 quant remains a memory
     win rather than a decode-speed win until the next design uses tensor-core
     friendly activation quantization or fuses quant projection with more of the
     native_graph token path.
19. V100 + Ada/Blackwell benchmark matrix.
   - `bench/run_v100_fast_decode_validation.sh` remains the broad V100
     regression gate.
   - `bench/run_4090_fused_backend_validation.sh` is the Ada/4090 fused-backend
     gate. It validates the HF-native default path, graph overhead, the default
     fused recurrent+output A/B matrix, and a small set of negative opt-in
     probes so future changes do not accidentally default a microbench-only
     fusion.
   - `bench/run_4090_quant_validation.sh` is the Ada/4090 native-quant gate. It
     runs the single-load W8/W4 R/K/V sweep with `TORCH_CUDA_ARCH_LIST=8.9` and
     emits an analyzer report so quant work is tracked separately from generic
     bitsandbytes compatibility.
   - Blackwell/5070-specific evidence lives in `BLACKWELL_50SERIES.md`; 5070
     uses the same software stack but `sm_120`-specific kernel behavior must not
     be projected onto Ada without the 4090 gate.
20. Native fused prefill scan and bsz=1 bottleneck breakdown.
   - `bench/bench_native_prefill_scan.py` now records model-size-labeled
     end-to-end native prefill rows, and the analyzer compares
     `native_prefill_tokps_total` against Albatross for exact model-size cases
     instead of falling back to older chunked-prefill rows.
   - RTX 4090 / Ada 0.4B fp16 prompt=512 with
     `RWKV7_NATIVE_PREFILL_FUSED_SCAN=1` reaches `22025.2` tok/s at bsz=1 and
     `76787.8` tok/s at bsz=4. That is `0.3668x` and `0.6519x` of Albatross:
     bsz=4 clears prefill P1, while bsz=1 remains the prefill blocker.
   - `bench/bench_native_prefill_breakdown.py` records the next optimization
     target. On the same 4090 0.4B prompt=512 rows, bsz=1 time is dominated by
     `recurrent_scan` (`9.3071ms`, share `0.3509`) and
     `attn_lora_state_prep` (`8.9671ms`, share `0.3381`). bsz=4 is still led by
     `recurrent_scan` (`10.5853ms`, share `0.3593`), followed by FFN
     (`7.3115ms`, share `0.2482`). This means the next bsz=1 prefill work
     should tune the fused scan and LoRA/state-prep path, not cache.

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
