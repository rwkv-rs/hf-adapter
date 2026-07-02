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

## Current next target: HF-compatible native fused backend

The next phase is not another wrapper-speed pass. The target is to keep the HF
adapter contract intact while replacing the hot math with a native fused backend.
See `docs/native_fused_roadmap.md` for the agent-readable implementation map.

Required direction:

- Align math against `rwkv_v7_numpy.py` and real-model execution from
  `run_rwkv7_qwen35.py` before changing kernel layout.
- Implement around train_temp-style boundaries: `tmix_mix6`,
  `kk_pre/state_prep`, `lnx_rkvres_xg`, `cmix`, and `clampw`.
- Use Albatross-style GPU-specific layout/autotune. Promote only exact-card
  end-to-end wins with correctness rows.
- Treat DPLR/chunked prefill as the bsz=1 prefill breakthrough path, especially
  for 4090 fp16 prompt512 where bsz=1 remains the blocker.

Explicit non-goals / bans for this phase:

- No vLLM or SGLang integration work in this repository.
- No wrapper/cache micro-optimization as the main performance plan.
- No promotion of the full-head scan+output fused prefill path; current telemetry
  shows it destroys split-row scan occupancy.
- No quantized-speed claim until a native fused quant kernel beats fp16/W16
  end-to-end while preserving memory and correctness.
- No defaulting shallow projection/LoRA/shift/FFN/output-project probes from
  isolated microbench wins.

Minimum validation for promotable work: RTX 4090 fp16 bsz=1/4 prompt512 prefill,
decode, correctness, peak memory/VRAM, and `bench/analyze_results.py` output
with `fused_backend_targets` / Albatross ratios.

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
     `fused_rkv_wag_projection_proto` and now supports prefill-shaped
     `[B,T,H]` rows. The first V100 decode-shaped row was correctness-clean and
     slightly faster, but the 4090 / 0.4B / fp16 / prompt512 prefill-shaped
     rows are slower than cuBLAS/torch (`0.6823x` at bsz=1 and `0.1471x` at
     bsz=4 with `block_m/r/k=64`). Do not integrate this two-launch projection
     grouping into prefill; the next step must fuse a larger norm/shift +
     projection/LoRA + state-prep region or improve the dense projection kernel.
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
     RTX 4090 prefill-shaped rows confirm the same conclusion more strongly:
     `B*T=512` reaches only `0.1570x` of the current cuBLAS FFN path, and
     `B*T=2048` reaches `0.0785x`. Do not wire this FFN-only kernel into native
     prefill; pursue FFN only as part of a larger graph/layer fusion.
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
   - The recurrent scan kernel now has an opt-in split-row tile
     (`block_m` / `RWKV7_NATIVE_PREFILL_SCAN_BLOCK_M`) so N=64 heads no longer
     require a single Triton program to keep the full 64x64 state tile live.
     On RTX 4090 isolated scan, `block_m=8` improves T=512 latency from
     `0.32535ms` to `0.19627ms` at bsz=1 and from `0.32617ms` to `0.21120ms`
     at bsz=4, with torch-reference cosine still `1.0` on T=128 checks. In the
     full prefill path this raises the best recorded bsz=4 throughput to
     `81047.4` tok/s (`0.6881x` Albatross), but bsz=1 remains stuck around
     `0.3668x`; next work must reduce LoRA/state-prep and other full-layer
     overhead rather than assuming scan-only tuning is sufficient.
   - `rwkv7_hf.fused_prefill.fused_prefill_state_prep()` now provides an
     opt-in prefill state-prep fusion under
     `RWKV7_NATIVE_PREFILL_FUSED_STATE_PREP=1`. It fuses W decay, K adjustment,
     per-head `kk` normalization, and optional V interpolation after the
     cuBLAS-backed projection/LoRA modules. On RTX 4090 / 0.4B / fp16 /
     prompt=512 with `RWKV7_NATIVE_PREFILL_SCAN_BLOCK_M=8`, it is
     correctness-clean and raises bsz=1 native prefill from `21857.3` to
     `22358.5` tok/s (`0.3723x` Albatross) while leaving bsz=4 essentially
     neutral (`81144.8` tok/s, `0.6889x` Albatross). This is useful but not
   sufficient for P1; the next real target is deeper LoRA/projection fusion
   because `attn_lora_state_prep` still leads bsz=1 breakdown
   (`7.858ms`, share `0.3106`).
   - Raw-W `clampw` split-row scan is available as an opt-in probe under
     `RWKV7_NATIVE_PREFILL_FUSED_CLAMPW_SCAN=1` together with fused scan. It
     keeps raw `w` out of state-prep, computes
     `exp(-0.606531 * sigmoid(w_raw))` inside the split-row scan, and uses the
     no-W `fused_prefill_kv_kk_prep()` path for K/V/KK prep. On RTX 4090 /
     0.4B / fp16 / prompt=512 with `SCAN_BLOCK_M=8`,
     `SCAN_NUM_WARPS=1`, correctness/cache handoff pass, but end-to-end is
     slightly slower than the baseline state-prep row: bsz=1 `21548.3` vs
     `21742.8` tok/s (`0.991x`) and bsz=4 `80880.1` vs `81057.8` tok/s
     (`0.998x`). Fine breakdown confirms no-W state-prep shrinks
     `3.2887ms -> 2.8226ms`, but scan grows `7.4649ms -> 7.7147ms`; keep
     clampw scan opt-in telemetry-only and do not promote unless a future
     larger fused scan/state-prep kernel wins end-to-end.
   - `rwkv7_hf.dplr_prefill.dplr_chunk_scan()` is now the correctness oracle
     for the DPLR/chunked-prefill line. It accepts `[B,T,H,N]` or flat
     `[B,T,H*N]` tensors plus native `[B,H,N,N]` state and exposes the future
     chunk boundary, but V1 intentionally scans sequentially inside each chunk.
     The opt-in native prefill path
     `RWKV7_NATIVE_PREFILL_DPLR_SCAN=1` /
     `RWKV7_NATIVE_PREFILL_DPLR_CHUNK_SIZE=64` is correctness/cache clean on
     RTX 4090 / 0.4B / fp16 / prompt=128 (`greedy_match=true`,
     `decode_after_prefill_greedy_match=true`), but it remains token-loop
     speed (`~220 tok/s`) because no affine/WY chunk summary or parallel chunk
     apply kernel exists yet. Treat this as the test harness for the next
     algorithmic DPLR implementation, not a performance path.
   - Prefill W/A/G/V-gate LoRA grouping is also wired as an opt-in adaptive
     probe (`RWKV7_NATIVE_PREFILL_FUSED_WAVG_LORA=1`,
     `RWKV7_NATIVE_PREFILL_FUSED_WAVG_LORA_MAX_M`, default `1024` rows). The
     isolated 4090 rows show `B*T=512` faster (`1.2762x`) but `B*T=2048`
     slower (`0.6626x`), and the end-to-end bsz=1 prefill row regresses to
     `21773.4` tok/s when enabled. Keep it telemetry-only; do not default it
     until a deeper projection+LoRA design improves full prefill.
   - `bench/bench_native_prefill_breakdown.py --fine-attn` now splits the
     remaining attention prep bucket into LoRA, dense R/K/V projection, fused
     state-prep, and scan components. With state-prep enabled on 4090 / 0.4B /
     fp16 / prompt=512, the latest bsz=1 row is led by recurrent scan
     (`7.6627ms`), LoRA sum (`6.2419ms`), norm/shift/mix (`3.8281ms`), FFN
     (`3.9687ms`), fused state-prep (`3.1581ms`), and dense R/K/V projection
     sum (`2.2056ms`; R `0.8844ms`, K `0.6769ms`, V `0.6443ms`). A follow-up
     full-prefill block sweep confirms `RWKV7_NATIVE_PREFILL_SCAN_BLOCK_M=8`
     remains the best recorded Ada tile; `4`, `16`, and `32` are slower for
     both bsz=1 and bsz=4. Next prefill work should therefore be a deeper
     cross-bucket fusion (scan + norm/shift/projection/state prep), not cache
     work and not shallow WAVG LoRA alone.
   - Native prefill can also reuse the decode output-prep kernel under
     `RWKV7_NATIVE_PREFILL_FUSED_OUTPUT=1`. The 4090 / 0.4B / fp16 /
     prompt=512 A/B is correctness-clean, and fine breakdown shows output prep
     itself shrinking to `0.1855ms` (bsz=1), but end-to-end prefill regresses
     versus the state-prep-only row (`21691.5` vs `22358.5` tok/s at bsz=1;
     `80948.1` vs `81144.8` tok/s at bsz=4). Keep this path opt-in telemetry;
     it is too small a bucket to offset extra overhead without deeper fusion.
   - `RWKV7_NATIVE_PREFILL_SCAN_NUM_WARPS` is now recorded for both the
     recurrent-scan microbench and full native-prefill rows. On RTX 4090 /
     0.4B / fp16 / prompt=512 with `SCAN_BLOCK_M=8`, isolated scan likes
     different warp counts by batch (`num_warps=8` is fastest at bsz=1,
     `0.15578ms`; `num_warps=1` is fastest at bsz=4, `0.19655ms`), while
     full prefill does not repeat the microbench win (`22018.8` tok/s at
     bsz=1 with warps=1, `21733.0` with warps=8, both below the existing
     state-prep-only `22358.5` tok/s row). Keep the default heuristic
     unchanged and treat this as a per-card telemetry override, not a promoted
     Ada default.
   - `RWKV7_NATIVE_PREFILL_FUSED_SHIFT_MIX=1` reuses the Triton attention
     shift-mix kernel for full `[B,T,H]` prefill tensors. It is
     correctness-clean, but the standalone prefill-shaped shift-mix microbench
     is slower than torch addcmul on 4090 (`0.6876x` at bsz=1/T=512 and
     `0.6940x` at bsz=4/T=512). End-to-end with state-prep and
     `SCAN_BLOCK_M=8`, `SCAN_NUM_WARPS=4` gives `21965.9` tok/s at bsz=1
     and `81303.7` tok/s at bsz=4: a small win over the explicit warps=4
     rows, but bsz=1 is still below the best state-prep-only `22358.5` tok/s
     row. Keep this opt-in telemetry-only; shallow shift-mix is not the
     missing P1 fix unless folded into a larger norm/shift/projection/state
     kernel.
   - `rwkv7_hf.fused_norm_mix.fused_attn_norm_shift_mix()` is available as a
     pure-torch correctness oracle for the larger attention pre-norm /
     attention-norm / shift / mix6 boundary. It is deliberately not wired into
     `native_jit` and makes no speed claim; use it to test a future fused
     `tmix_mix6` kernel that includes the layernorm boundary instead of the
     already-negative shallow shift-mix-only probe.
   - `RWKV7_NATIVE_PREFILL_FUSED_SCAN_OUTPUT=1` tests a larger full-head
     kernel that fuses the recurrent scan with group-norm, recurrent
     correction, and gate multiply before the final cuBLAS `o_proj`. It is
     correctness-clean on RTX 4090 / 0.4B / fp16 / prompt=512, but forcing the
     scan to own all 64 rows of a head in one Triton program destroys the
     split-row scan benefit: bsz=1 falls to `224.1` tok/s and bsz=4 to
     `764.1` tok/s, only `0.009x`-`0.010x` of the best split scan rows. Keep
     this as negative telemetry only. The next prefill kernel should preserve
     split-row scan occupancy and attack the larger norm/shift/projection/LoRA
     and state-prep buckets, not fuse output prep into a full-head scan.
   - `bench/bench_native_prefill_breakdown.py --layer-breakdown` now records
     per-layer component timings without adding extra timed passes. The 4090 /
     0.4B / fp16 / prompt=512 bsz=1 row shows the bsz=1 bottleneck is broad
     rather than isolated to one pathological layer: top layer totals are
     close (`L17=1.2175ms`, `L0=1.2012ms`, `L1=1.1727ms`, `L2=1.1724ms`,
     `L4=1.1684ms`). The hottest layer is still led by recurrent scan
     (`0.3052ms`) plus state-prep/FFN/norm-shift work, so the next optimization
     should be a repeated per-layer fusion pattern that benefits all 24 layers,
     not a layer-specific special case.

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

## DPLR/chunked prefill prototype benchmark

`bench/bench_dplr_prefill_scan.py` is the standalone validation harness for the
DPLR/chunked prefill prototype. It is synthetic-only: it creates post-projection
`r/w/k/v/kk/a` tensors plus native `[B,H,N,N]` state, does not load HF model
weights, and intentionally does not import or call `native_jit`. JSONL rows use
`axis="dplr_prefill_scan_proto"` and compare:

- `algorithm="torch_recurrent_scan"`: the pure torch recurrent reference.
- `algorithm="sequential"`: `dplr_chunk_scan()` with the current sequential
  within-chunk implementation.
- `algorithm="affine"`: the dense torch affine prototype. It explicitly
  constructs each per-token transform
  `A_t = diag(w_t) + (-kk_t)(kk_t*a_t)^T` and `B_t = v_t k_t^T`, then composes
  chunk prefix/suffix transforms. This is correctness scaffolding for the
  future WY/Triton path and is intentionally O(T*N^3), not a promoted speed
  implementation.

4090 validation for the dense affine prototype:

- `tests/test_dplr_prefill_scan.py`: PASS.
- Synthetic fp32 `B=1,T=32/64,H=2,N=8,chunk=8/16`: affine max diff
  `<=3e-8` vs `torch_recurrent_scan`; affine is slower than sequential as
  expected (`~5.1k`-`5.3k tok/s` vs `~8.6k`-`8.7k tok/s`).
- Synthetic fp16 `B=1,T=32,H=2,N=8`: affine output max diff
  `1.22e-4`, state max diff `2.87e-5`, min cosine `0.99999994`.
- HF native prefill smoke with `RWKV7_DPLR_PREFILL_ALGORITHM=affine`,
  prompt32, chunk16: `status=pass`, `greedy_match=true`,
  `decode_after_prefill_greedy_match=true`.

Safe server validation command for the dense affine prototype:

```bash
PYTHONPATH=. python bench/bench_dplr_prefill_scan.py \
  --device cuda --dtype fp32 \
  --batch-sizes 1 --tokens 32 64 \
  --heads 2 --head-dim 8 --chunk-sizes 8 16 \
  --algorithms sequential affine \
  --warmup 1 --steps 2 \
  --results bench/results.jsonl

PYTHONPATH=. python bench/analyze_results.py --results bench/results.jsonl --dtype fp32
```

Do not run the dense affine prototype at `H=16,N=64,T=512` except deliberately
as a stress test; it materializes dense chunk transforms and is expected to be
slower than the sequential reference until the WY/low-rank chunk composer lands.

### WY/lowrank prototype validation commands

`bench/bench_dplr_prefill_scan.py` accepts `--algorithms sequential affine wy
lowrank` and records both `requested_algorithm` and `effective_algorithm`.
If the checked-out `dplr_chunk_scan()` does not yet support `wy` or `lowrank`,
the row is emitted as `status="skip_unsupported_algorithm"`; if one alias is
implemented and the other is requested, the benchmark may use the implemented
alias and record `status="fallback_algorithm_alias"`.

Start with small correctness-focused matrices, where the dense affine scaffold
is still safe enough to compare against sequential and the future WY/lowrank
path:

```bash
PYTHONPATH=. python bench/bench_dplr_prefill_scan.py \
  --device cuda --dtype fp32 \
  --batch-sizes 1 --tokens 32 64 \
  --heads 2 --head-dim 8 --chunk-sizes 8 16 \
  --algorithms sequential affine wy lowrank \
  --warmup 1 --steps 2 \
  --results bench/results.jsonl

PYTHONPATH=. python bench/analyze_results.py --results bench/results.jsonl --dtype fp32
```

Then measure the target prefill-shaped case without dense affine.  For
`H=16,N=64` do **not** include `affine`; it materializes dense O(N^3) chunk
transforms and is not the intended large-matrix path.

```bash
PYTHONPATH=. python bench/bench_dplr_prefill_scan.py \
  --device cuda --dtype fp16 \
  --batch-sizes 1 --tokens 512 \
  --heads 16 --head-dim 64 --chunk-sizes 64 128 \
  --algorithms sequential wy \
  --warmup 3 --steps 10 \
  --results bench/results.jsonl

PYTHONPATH=. python bench/analyze_results.py --results bench/results.jsonl
```

If the implementation exposes the low-rank path as `lowrank` rather than `wy`,
replace the second command with `--algorithms sequential lowrank`, or list both
aliases and let unsupported rows skip explicitly in JSONL.

4090 validation for the current pure-torch WY/lowrank prototype:

- `tests/test_dplr_prefill_scan.py`: PASS.
- Synthetic fp32 `B=1,T=32/64,H=2,N=8,chunk=8/16`: `wy`/`lowrank`
  rows PASS with output/state max diff `<=2.3e-8` vs `torch_recurrent_scan`.
  The implementation is correctness-only in torch: `wy`/`lowrank` run at
  roughly `2.5k`-`2.6k tok/s`, slower than sequential (`~8.4k`-`8.5k tok/s`).
- Synthetic fp16 `B=1,T=128,H=4,N=16,chunk=16/32`: `wy`/`lowrank`
  rows PASS with output max diff `1.22e-4`, state max diff `6.65e-5`, and
  min cosine `0.99999988`.  They run at roughly `2.8k tok/s`, slower than
  sequential (`~9.3k tok/s`).
- HF native prefill smoke on 0.4B / prompt32 with
  `RWKV7_NATIVE_PREFILL_DPLR_SCAN=1` and
  `RWKV7_DPLR_PREFILL_ALGORITHM=wy`: `status=pass`, `greedy_match=true`,
  `decode_after_prefill_greedy_match=true`, but only `~188 tok/s`.  This
  confirms cache/generate correctness and also confirms that Python WY is not a
  performance path.

Next step: stop micro-optimizing the pure torch loops and move the same
contract into compiled kernels: (1) chunk summary for diagonal-plus-low-rank
metadata, (2) chunk-level prefix combine, (3) chunk apply/output.  Keep
`algorithm_family="lowrank_wy"` and `is_dense_affine=false` in JSON so we can
tell the real WY path apart from dense affine and sequential fallbacks.

### Triton DPLR/WY P0 compiled prototype

`rwkv7_hf/dplr_prefill_triton.py` adds the first opt-in compiled backend hook:

- `dplr_chunk_scan_triton(...)`
- `dplr_chunk_scan_triton_available()`

The synthetic benchmark can request it with `--algorithms triton_wy` (or the
alias `cuda_wy`).  JSON rows use `algorithm_family="triton_wy"` plus
`triton_wy_available` and `triton_wy_block_m`.

Important limitation: this P0 is a compiled DPLR scan bridge, not the final
three-stage WY factor implementation.  It currently delegates the recurrence to
the existing Triton `fused_recurrent_scan` kernel while preserving the future
`dplr_chunk_scan(..., chunk_size=...)` API.  It proves the synthetic opt-in
compiled path and benchmark plumbing; the next kernel step is still explicit
chunk summary + prefix combine + chunk apply.

4090 validation for the P0 compiled backend:

- `tests/test_dplr_prefill_scan.py`: PASS.
- Synthetic fp32 `B=1,T=32,H=2,N=8,chunk=8`: `triton_wy` PASS with
  output/state max diff `1.49e-8`, about `0.116ms` / `276k tok/s`.
- Target synthetic fp16 `B=1,T=512,H=16,N=64,chunk=64`: `triton_wy` PASS with
  output max diff `4.88e-4`, state max diff `1.26e-4`, min cosine
  `0.99999988`; `0.234ms` / `2.19M tok/s` versus pure torch sequential
  `64.63ms` / `7.9k tok/s`.

Target-shape command:

```bash
PYTHONPATH=. python bench/bench_dplr_prefill_scan.py \
  --device cuda --dtype fp16 \
  --batch-sizes 1 --tokens 512 \
  --heads 16 --head-dim 64 --chunk-sizes 64 \
  --algorithms sequential triton_wy \
  --warmup 1 --steps 3 \
  --results bench/results.jsonl

PYTHONPATH=. python bench/analyze_results.py --results bench/results.jsonl
```

For HF end-to-end validation, ensure the model directory loaded by
`trust_remote_code=True` contains the same updated `native_jit.py`,
`dplr_prefill.py`, and `dplr_prefill_triton.py`; otherwise the benchmark row may
only reflect the checkpoint-local remote code rather than this repository's
new backend.

`bench/bench_native_prefill_scan.py --code-source repo` creates a temporary
checkpoint directory that symlinks the model weights/tokenizer files and copies
the current repo's `rwkv7_hf/*.py` files into the HF remote-code root.  Rows now
record `code_source`, `effective_model_path`, and `native_jit_module` so DPLR
experiments can distinguish checkpoint-local code from the current worktree.

4090 HF smoke using current repo code:

```bash
env RWKV7_NATIVE_PREFILL_DPLR_SCAN=1 \
    RWKV7_NATIVE_PREFILL_DPLR_CHUNK_SIZE=64 \
    RWKV7_DPLR_PREFILL_ALGORITHM=triton_wy \
    RWKV7_DPLR_TRITON_BLOCK_M=8 \
    RWKV7_DPLR_TRITON_STRICT=1 \
    RWKV7_NATIVE_PREFILL_FUSED_SCAN=0 \
    RWKV7_NATIVE_PREFILL_FUSED_SCAN_OUTPUT=0 \
    RWKV7_NATIVE_PREFILL_FUSED_CLAMPW_SCAN=0 \
    RWKV7_NATIVE_PREFILL_FUSED_STATE_PREP=1 \
    PYTHONPATH=. \
    python bench/bench_native_prefill_scan.py \
      --model /workspace/models/rwkv7/rwkv7-g1d-0.4b-hf \
      --code-source repo \
      --device cuda --dtype fp16 \
      --batch-sizes 1 --prompt-tokens 512 \
      --fused-scan false \
      --warmup 2 --steps 5 \
      --results bench/results.jsonl
```

Result: `status=pass`, `prefill_dplr_scan_effective=true`,
`greedy_match=true`, `decode_after_prefill_greedy_match=true`, and
`native_prefill_tokps_total=21567.9` on 4090 / 0.4B / prompt512.  This proves
the repo-code HF native path can dispatch through
`RWKV7_DPLR_PREFILL_ALGORITHM=triton_wy`.  It is roughly the existing fused-scan
performance envelope because P0 delegates to `fused_recurrent_scan`; exceeding
the current mainline still requires the real three-stage WY kernels.

### Triton chunk-summary kernel boundary

The first explicit chunk-summary kernel boundary now exists in
`dplr_prefill_triton.py`:

- `dplr_dense_chunk_summary_torch(...)`
- `dplr_dense_chunk_summary_triton(...)`
- `dplr_dense_chunk_summary_triton_available()`

This is not the final compact WY summary yet: it returns dense chunk affine
summaries `transition/additive` shaped `[B, chunks, H, N, N]` for
`S_end = S_start @ transition + additive`.  It is still useful because it pins
down stage 1 of the target backend and gives a correctness oracle for the next
two stages:

1. chunk summary kernel — **present as dense DPLR summary**
2. chunk-level prefix combine — next
3. chunk apply/output — next

Run the summary probe with:

```bash
PYTHONPATH=. python bench/bench_dplr_prefill_scan.py \
  --device cuda --dtype fp16 \
  --batch-sizes 1 --tokens 512 \
  --heads 16 --head-dim 64 --chunk-sizes 64 \
  --algorithms triton_wy \
  --summary-probe \
  --warmup 1 --steps 3 \
  --results bench/results.jsonl
```

4090 target-shape result for the dense summary probe:

- `axis="dplr_chunk_summary_proto"`
- `status=pass`
- `summary_shape=[1,8,16,64,64]`
- `transition_max_abs_diff=4.97e-14`
- `additive_max_abs_diff=5.96e-8`
- `state_max_abs_diff=1.26e-4`
- `ms=0.31705`, `tokps=1.61M`

Next implementation step: use those per-chunk summaries to compute chunk start
states, then add a chunk-apply kernel that emits recurrent outputs from each
chunk start state.  After that is correct, replace dense `transition/additive`
with compact WY factors to reduce memory and make the design closer to the
Albatross/DPLR line.
