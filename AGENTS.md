# AGENTS.md

## Project Mission

This repository is now scoped to the **RWKV-7 Hugging Face / Transformers adapter only**.

Historical upstream context mentioned three independent tracks: Hugging Face,
vLLM, and SGLang. For this repository and active work, do **not** build or gate
native vLLM/SGLang integrations. Any vLLM/SGLang work is a separate future
project and must not block HF deliverables.

The active reward target is the HF/Transformers track: make RWKV-7 usable from
standard HF APIs with near-production correctness, performance, memory behavior,
training compatibility, quantized inference, and reproducible benchmarks.

## Current Agent Contract: Native Fused HF Backend

This is the active contract for the next workers on this branch. Treat
`FUSED_BACKEND.md` and `docs/native_fused_roadmap.md` as the performance
roadmap.

- Scope is **HF adapter only**. Do not implement or gate native vLLM/SGLang
  integrations in this repository.
- Keep HF compatibility as the invariant: `AutoModelForCausalLM`, `generate`,
  PEFT, Trainer, TRL, `RWKV7StateCache`, dynamic batching, chunked prefill,
  save/load, and quantized loading must keep working.
- Move the speed core to native fused backends: fused fp16 first, then fused
  quant. The wrapper is the compatibility shell, not the place for the next
  performance breakthrough.
- Use official math alignment from `rwkv_v7_numpy.py` and
  `run_rwkv7_qwen35.py`; preserve exact RWKV-7 recurrence, clamp, state, and
  output semantics before optimizing layout.
- Follow train_temp-style fused boundaries: `tmix_mix6`, `kk_pre/state_prep`,
  `lnx_rkvres_xg`, `cmix`, and `clampw`.
- Use Albatross-style GPU-specific layout/autotune. Exact-card rows decide
  defaults; V100, 4090, A100/H100, and Blackwell must not blindly share tile
  choices.
- Treat DPLR/chunked prefill as the bsz=1 prompt-prefill breakthrough line.
  Do not spend the next phase on wrapper/cache micro-optimizations.
- Forbidden directions: wrapper micro-optimization as the main plan, native
  vLLM/SGLang work, defaulting the full-head scan+output fused prefill path,
  and full-memory quantized-speed claims before a native fused quant kernel
  beats fp16 end-to-end. Speed-oriented quantization may be claimed separately
  only when W8/W4 rows show lower model footprint, fp16-or-better decode on the
  exact card, and logits/greedy-token alignment vs fp16.
- Required next validation loop: RTX 4090 fp16, bsz=1/4, prompt512 prefill,
  decode, correctness, peak memory/VRAM, and `bench/analyze_results.py`
  reporting.

## Current RTX 4090 Milestone (2026-07-10)

- 0.4B dense fp16 native-graph decode now reaches
  `795.7/1469.5/2585.7/3185.3 tok/s` for bsz1/2/4/8, or
  `1.007x/1.016x/1.008x/1.418x` the matching recorded Albatross rows. All four
  graph runners coexist and pass 32-step greedy plus standard-HF fallback.
  Exact-4090 sparse FFN is restricted to rows 1/2; batch-keyed packed weights
  and graph-safe in-place residual accumulation are required to prevent
  cross-graph state corruption.
- The optional TorchAO group-128 W4 lane is a real speed lane, not only a
  memory smoke. With bf16 activations and the Ada bf16 W/A/G/V fusion it reaches
  `927/1713/3093/3407 tok/s`, or `1.17x-1.52x` Albatross, for 0.4B
  bsz1/2/4/8. Payload is `0.399x`, logit cosine is `>=0.999239`, and next-token
  equality passes. 1.5B bsz1/2 is `2.17x/2.37x` its bf16 baseline with
  `0.355x` payload.
- Do not generalize the W4 decode result to all quantization: W4 prefill is only
  `0.819x-0.831x` bf16 for the measured B1/B4 T64/T256 rows, and W8 remains
  below fp16. The next quant priority is W8 tensor-core/grouped projection,
  followed by quantized prefill.
- Evidence and reproduce commands are in the RTX 4090 section of
  `BENCHMARK.md`; the runtime integration is
  `rwkv7_hf/native_quant_torchao.py` plus quant-aware native-graph operand
  extraction in `rwkv7_hf/native_jit.py`.

## Current V100 Decode Milestone

The 2026-07-10 sm70 pass adds decode norm/mix fusion, grouped shape-routed
projection/FFN kernels, and raw recurrent-output preparation. Same-host
0.1B/0.4B/1.5B × bsz1/2/4/8 measurements now span `0.629x-1.185x`
Albatross: all 12 rows pass P1, all bsz8 rows pass P3, and 0.4B/1.5B bsz8
exceed Albatross. The raw recurrent A/B is 32-step greedy-exact at 0.4B and
1.5B bsz2. Evidence is under
`bench/v100_sm70_decode_gap_20260710/`.

This closes the V100 decode P1 floor, not the final mission. The next workers
must pursue, in order:

1. universal V100 P2/P3 for the remaining bsz1/2/4 rows;
2. native fused W8/W4 with lower footprint and end-to-end speed >= fp16;
3. exact-card reproduction on 4090/A100/H100/Blackwell and AMD fallback;
4. continued prefill/DPLR work without regressing the promoted decode routes.

## Parallel Prefill Goal: DPLR/WY Compiled Prototype

Active branch work is now the opt-in DPLR/WY compiled prefill backend, not
wrapper micro-optimization. Keep the default HF behavior unchanged unless a
benchmark explicitly opts in.

Goal:

- Move `dplr_chunk_scan(algorithm="wy"/"lowrank")` from pure torch
  correctness prototype toward a real Triton/CUDA performance prototype.
- Maintain native VxK state layout `[B,H,N,N]`, fp16/bf16/fp32 token inputs,
  and fp32 state accumulation.
- Synthetic first: support the critical target
  `B=1,H=16,N=64,T=512,chunk_size=64,fp16` on RTX 4090.
- Correctness gates: match `torch_recurrent_scan`; for fp16 target require
  `out_min_cosine >= 0.9999` and keep greedy/cache smoke passing when routed
  through HF repo-code loading.

Current implementation state:

- `rwkv7_hf/dplr_prefill_triton.py` exposes:
  - `dplr_chunk_scan_triton(...)` / `dplr_chunk_scan_triton_available()`
  - dense chunk summary: `dplr_dense_chunk_summary_*`
  - dense prefix combine: `dplr_dense_prefix_combine_*`
  - dense chunk apply/output: `dplr_dense_chunk_apply_*`
  - dense three-stage scaffold: `dplr_dense_three_stage_triton(...)`
- `algorithm="triton_wy"` is the P0 compiled bridge using the existing fused
  recurrent scan. It is fast and correctness-passing, but it is not yet compact
  WY.
- `algorithm="triton_dense3"` is the explicit dense three-stage scaffold
  (summary -> prefix -> apply/output). It proves the mathematical kernel
  boundaries, but it materializes dense `[N,N]` summaries and is expected to be
  slower than the P0 fused scan until replaced with compact WY factors.

Latest RTX 4090 target evidence:

- Synthetic `B=1,T=512,H=16,N=64,chunk=64,fp16`:
  - `sequential`: pass, about `55.63 ms`, `9.2k tok/s`
  - `triton_wy`: pass, about `0.233 ms`, `2.20M tok/s`,
    `out_min_cosine ~= 0.9999999`
  - `triton_dense3`: pass; latest stage-probe full row is about
    `0.264-0.269 ms`, `~1.9M tok/s`, `out_min_cosine = 1.0`
  - dense stage split from `--stage-probe`: summary `~0.144 ms`, prefix
    `~0.092 ms`, apply/output `~0.065 ms`, summary shape
    `[1,8,16,64,64]`. Dense summary/prefix `[N,N]` traffic is the first
    compact-WY target.
- HF repo-code smoke on 4090 / 0.4B / prompt512 / bsz1:
  - Sweep path: `/tmp/native_4090_todo_sweep_20260702_103919.jsonl`.
  - Albatross reference for this shape remains `52,148.52 tok/s`; `0.45x`
    is `23,467 tok/s`.
  - DPLR repo-code rows: `triton_wy` pass at `20,421.7 tok/s` (`0.3916x`),
    `triton_dense3` pass at `18,546.0 tok/s` (`0.3556x`),
    `triton_wy_compact` pass at `17,970.5 tok/s` (`0.3446x`).
  - Fastest short sweep row was the fused recurrent scan path, not DPLR:
    `RWKV7_NATIVE_PREFILL_FUSED_SCAN=1`,
    `RWKV7_NATIVE_PREFILL_SCAN_BLOCK_M=8`,
    `RWKV7_NATIVE_PREFILL_SCAN_NUM_WARPS=1`,
    `RWKV7_NATIVE_PREFILL_FUSED_STATE_PREP=1`; pass at `22,777.0 tok/s`
    (`0.4368x`) and `991.2 MiB`.
  - Confirmation for that older split setting:
    `/tmp/native_4090_todo_confirm_20260702_104202.jsonl`, pass at
    `22,292.0 tok/s` (`0.4275x`), below the `0.45x` target.
  - Breakdown path: `/tmp/native_4090_todo_breakdown_20260702_104126.jsonl`.
    Top components for the best fused-scan setting are recurrent scan
    `7.4571 ms` / `26.34%`, FFN `4.0836 ms` / `14.42%`,
    attention norm+shift-mix `3.8040 ms` / `13.44%`, and fused state prep
    `3.2982 ms` / `11.65%`.
  - New opt-in fused state-prep + recurrent scan row:
    `RWKV7_NATIVE_PREFILL_FUSED_STATE_SCAN=1` and
    `RWKV7_NATIVE_PREFILL_FUSED_OUTPUT=1` with path
    `/tmp/native_4090_fused_state_scan_confirm_20260702_111924.jsonl` passes
    greedy/cache smoke at `25,663.2 tok/s`, `19.9507 ms`, `0.4921x`
    Albatross, and `989.2 MiB` peak VRAM. This closes the 0.45x checkpoint but
    remains below the `0.60x` stretch target (`31,289 tok/s`).
  - DPLR compact retest after the launch-count reduction is still slower:
    `/tmp/native_4090_dplr_compact_retest_20260702_111924.jsonl`,
    `16,863.4 tok/s`, `30.3616 ms`, `0.3234x` Albatross, greedy/cache smoke
    passing. Keep DPLR compact as high-upside work, but its next useful step is
    DPLR-specific apply/output fusion rather than wrapper-level changes.
  - Prefill fused-output-project is now an opt-in experiment
    (`RWKV7_NATIVE_PREFILL_FUSED_OUTPUT_PROJECT=1`) for evidence only. The
    first 4090 row `/tmp/native_4090_output_project_20260702_104430.jsonl`
    passed correctness but was slower (`18,228.8 tok/s`), so it must remain
    disabled by default.

Remaining before this goal is complete:

- Compact WY torch reference, dense reconstruction oracle, and first Triton
  compact summary kernel now exist via
  `dplr_compact_wy_chunk_summary_torch`,
  `dplr_compact_wy_chunk_summary_triton`,
  `dplr_compact_wy_summary_to_dense`, and
  `dplr_compact_wy_apply_summaries_torch`. The first Triton compact kernel is
  target-constrained to `N<=64, chunk_size<=64`; 4090 target factor diff is
  `<=5.96e-08`, final state diff is `~1.13e-04`, and summary time is
  `~0.155 ms`. Compact prefix combine now exists via
  `dplr_compact_wy_prefix_combine_torch` and
  `dplr_compact_wy_prefix_combine_triton`; 4090 target prefix time is
  `~0.067 ms` with starts diff vs dense `~5.96e-08`. The next required
  step is reusing current chunk apply/output to make a compact three-stage
  route. That route now exists as `dplr_compact_wy_three_stage_triton`; 4090
  target correctness passes (`out_min_cosine=1.0`, state diff
  `~1.26e-04` in the benchmark row). It is exposed as
  `RWKV7_DPLR_PREFILL_ALGORITHM=triton_wy_compact`; latest 4090 synthetic
  target benchmark is `~0.241 ms`, `~2.12M tok/s`, close to P0 `triton_wy`
  (`~0.228 ms`). HF repo-code smoke 0.4B/prompt512/bsz1 passes greedy/cache
  at `~17.5k tok/s`.
- Replace dense `[N,N]` transition/additive runtime summaries with compact
  WY/low-rank factors to reduce memory traffic and close the Albatross gap.
- Make the explicit three-stage path at least competitive with the P0 fused
  recurrent scan; current dense3 is correctness-first and slower than P0.
- The exact-4090 fixed-shape prefill graph now runs 0.4B prompt512 at
  `64511.2 tok/s` for bsz1 and `107870.1 tok/s` for bsz4. Bsz4 is `1.007x` a
  same-session Albatross rerun and `0.916x` the older strongest recorded row.
  The gain comes from no-`cat` sequence shift/state kernels plus fused ReLU²,
  not wrapper work. Continue deeper scan/projection/layout work against the
  historical high-water mark. Keep DPLR/WY as the cross-card/variable-shape
  algorithmic route rather than deleting it because one fixed shape passes.
- Do not call the DPLR/WY goal finished until compact WY or an equivalent
  compiled path is verified end-to-end against the original acceptance target.

## Current Apple Branch Checkpoint: MLX DPLR/WY Stage 1

The Apple sibling path now has `rwkv7_hf/mlx_dplr_prefill.py` and
`scripts/mlx_dplr_prefill_bench.py`. It ports the same compact factor contract
to MLX and implements custom Metal kernels for chunk summary and chunk
apply/output; prefix combine is still high-level MLX. This is a synthetic math
and kernel checkpoint, not yet a model-level prefill route.

- M5 production-shaped target: `B=1,T=512,H=16,N=64,chunk=64,fp16`,
  warmup 2, repeat 5.
- Recurrent reference median: `77.207 ms`, `6,631.48 effective tok/s`.
- High-level three stage: `207.038 ms`, `2,472.98 tok/s`.
- Metal summary: `58.801 ms`, factor max-abs `3.73e-08`.
- Metal chunk apply/output: `0.863 ms`; full output max-abs `1.53e-04`.
- Metal summary + MLX prefix + Metal apply: `60.249 ms`, `8,498.11 tok/s`,
  about `1.28x` the synthetic recurrent reference and `3.44x` high-level
  three-stage; final-state max-abs `1.14e-04`.
- Raw evidence: `bench/results_mlx_dplr_stage1_target_m5_20260710.jsonl`;
  the smaller kernel bring-up row remains in
  `bench/results_mlx_dplr_stage1_m5_20260710.jsonl`.

Next Apple critical path: parallelize/tile the still-dominant summary kernel,
support partial final chunks, then refactor MLX model prefill to layer-major
sequence execution and route attention WKV through this three-stage backend.
Do not claim Qwen/Albatross-level model performance from the synthetic row.

## Active Goal: Finish the Current HF Adapter First

Current priority: finish the RWKV-7 Hugging Face / Transformers adapter with
the hardware and evidence available now. Do not wait for H100/4090/5090/A100
access before completing the current repository deliverables. V100 remains the
active development and regression baseline; newer GPUs are follow-up validation
targets once available.

The current delivery strategy is:

- Keep the HF wrapper as the production-facing compatibility layer for
  `AutoModelForCausalLM`, `generate`, PEFT, Trainer, TRL, state cache,
  dynamic batching, chunked prefill, quantization, speculative decoding, and
  benchmark gates.
- Stop treating wrapper micro-optimization as the performance plan. The
  wrapper may be changed for HF compatibility, correctness, telemetry, and
  dispatch, but new speed wins should come from native fused fp16 kernels and
  later fused native W8/W4 kernels. `native_jit`, `native_graph`, cache reuse,
  and reduced launch count are fallback/baseline layers, not the next
  breakthrough by themselves.
- Keep `native_model` explicitly experimental. It is the long-term base for
  removing the mandatory FLA runtime, upstream Transformers work, AMD/CPU
  fallback, and future kernels. It must not be described as replacing the
  wrapper until it proves the same HF compatibility, batching, cache semantics,
  and benchmark coverage.
- Do not merge older native branches wholesale when they would remove current
  HF training, quantization, cache, benchmark, or telemetry work. Audit those
  branches and port only the useful implementation ideas.

Near-term completion, without waiting for extra GPUs:

1. Done: preserve V100 training telemetry for HF Trainer, TRL SFT, TRL DPO,
   and TRL GRPO in the benchmark/report pipeline.
2. Done: add Albatross A/B benchmark ingestion on the same checkpoint, V100,
   dtype, batch size, prompt length, decode length, and cache policy.
3. Done: harden the experimental native/no-FLA HF path with smoke tests for
   Trainer, TRL SFT, TRL DPO, TRL GRPO, PEFT adapter save/load/merge, Trainer
   checkpoint resume, and bnb W8/W4 functional quantized inference. These are
   compatibility gates only; they do not close the Albatross or quantized-speed
   gaps.
4. Current: finish W8/W4 reporting and gates so the repository clearly records
   both the memory-target bnb rows and the fastest passing hybrid variants.
5. Current: keep code/tests/docs green locally without CUDA, then merge only
   changes that preserve existing HF training, cache, quantization, benchmark,
   and telemetry behavior.
6. Current no-GPU task: finish executable DeepSpeed ZeRO-2/ZeRO-3 HF Trainer
   smoke harness, analyzer/report ingestion, docs, and local tests. Real pass
   rows can wait for live GPU/DeepSpeed access, but the repository should be
   ready to run them with one command.
7. Current performance phase: follow `docs/performance/FUSED_BACKEND.md` for the native fused
   fp16 -> native W8/W4 backend. The analyzer must track Albatross ratio
   ladders and quantized speed/footprint gates under `fused_backend_targets`.
8. Next when GPUs return: expand V100 evidence for large-model smoke,
   speed/precision sweeps, chunked prefill, dynamic batching, state-cache reuse,
   speculative decoding, and ZeRO-2/ZeRO-3 multi-GPU smoke.
9. Later validation: run the prepared benchmark matrix on H100/4090/5090/A100.
   These newer cards are validation targets, not blockers for current progress.

Current no-GPU work mode:

- Finish everything that does not require live CUDA access first: HF API
  compatibility code, analyzers, benchmark ingestion, result gates, docs, unit
  tests, and PR hygiene.
- Treat existing V100 evidence as the active baseline until GPUs return. Do not
  block merges on new H100/4090/5090/A100 numbers.
- Keep GPU-only work as explicit follow-up rows in `BENCHMARK.md` /
  `docs/archive/NEXT_STEPS.md`: fresh speed sweeps, large-model runs, fused W8/W4 kernels,
  ZeRO-2/3 multi-GPU validation, and cross-card validation.
- The immediate finish line for this repository is a clean HF adapter that can
  be reviewed, installed, tested, and benchmarked reproducibly; vLLM/SGLang and
  DFlash stay outside the current merge gate.
- Do not start vLLM/SGLang work in this repository while the HF adapter still
  has open local tasks. First finish the HF adapter evidence, gates, and docs.

## Target Acceptance Criteria

Use this HF-only checklist as the authoritative target for the active
deliverable:

1. Match or approach the current RWKV-LM and Albatross training/inference
   performance, speed, precision, and memory use through HF-compatible paths
   across common batch sizes.
2. HF adaptation must work with common Transformer-based PEFT, RL, and training
   libraries, including PEFT, TRL, SFT/DPO/GRPO-style workflows, Trainer-style
   loops, gradient accumulation, and real multi-batch training smoke tests.
3. HF serving helpers must expose RWKV recurrent state cache semantics, dynamic
   batch select/reorder/drop, chunked prefill, state-cache allocation/reuse, and
   cache-reuse metrics that can later be reused by serving integrations.
4. Hardware support should cover common professional and consumer GPUs:
   NVIDIA from Pascal onward where feasible, newer NVIDIA generations, and AMD
   GPUs. HF inference should keep a path toward PP/TP, and HF training should
   support DeepSpeed ZeRO-2 and ZeRO-3 where feasible.
5. Quantized inference must support common W8 and W4 modes, reduce memory
   accordingly, and be faster than W16 on common cards. Older cards may need
   dedicated optimization. Quality should get as close as possible to
   llama.cpp-style Q*_K_M levels.
6. Add initial HF-compatible speculative decoding support, such as using a
   smaller RWKV model as the draft model. DFlash, native vLLM/SGLang adapters,
   and deeper standalone serving-engine work stay as follow-up projects.

Benchmark comparisons must separate engine performance from model quality:

- Albatross is the high-performance RWKV inference-engine reference. Compare it
  against this repository on the same checkpoint, hardware, dtype, batch size,
  prompt length, decode length, and cache policy. Track prefill tok/s, decode
  tok/s, aggregate tok/s, latency percentiles, memory footprint, peak VRAM,
  state-cache reuse/hit rate, and dynamic-batch behavior.
- Qwen3.5 is the model-quality target. The overall model-level goal is to
  exceed comparable Qwen3.5 baselines on instruction quality, reasoning, math,
  code, multilingual/Chinese, long-context, and RL/PEFT training workflows. Do
  not treat an inference-engine speed win as proof of beating Qwen3.5 quality;
  require explicit evaluation rows and reproducible prompts/datasets.

The final implementation should approach the performance, speed, precision, and memory usage of the official RWKV-LM path and Albatross path across different batch sizes.

### HF Transformers Track

Required goals:

- Convert official RWKV-7 `.pth` checkpoints to Hugging Face format.
- Provide `RWKV7Config`, `RWKV7Model`, and `RWKV7ForCausalLM`.
- Provide RWKV tokenizer support.
- Support `AutoConfig.from_pretrained`, `AutoTokenizer.from_pretrained`, and
  `AutoModelForCausalLM.from_pretrained`.
- Support `generate(..., use_cache=True)` with RWKV recurrent state cache.
- Support HF-style recurrent-state utilities for serving-like usage:
  state-cache allocation/reuse, dynamic batch select/reorder/drop/compact,
  chunked prefill, offload/restore, and cache telemetry.
- Support PEFT LoRA workflows and common HF training / RL libraries, especially
  PEFT, Trainer, TRL `SFTTrainer`, `DPOTrainer`, and `GRPOTrainer`-style flows.
- Support DeepSpeed ZeRO-2/ZeRO-3 presets where feasible through HF training
  entrypoints.
- Support 8-bit and 4-bit HF inference paths that reduce memory, preserve
  quality as much as possible, and target speed no slower than W16 on common
  cards.
- Add initial HF-compatible speculative decoding support, such as a smaller
  RWKV draft model verified by a larger HF RWKV target model.
- Keep a migration path toward an upstreamable native Transformers
  implementation without a mandatory FLA runtime dependency.

### Hardware Support

Required goals:

- Support common professional GPUs.
- Support common consumer GPUs.
- Current development server has 2 x Tesla V100-PCIE-32GB.
- V100 is acceptable for smoke tests and development, but final performance work
  should also be validated on newer cards such as A100/H100/4090/5090 where
  available.
- AMD GPU support remains a compatibility target for the HF path, preferably via
  pure PyTorch/reference paths first and optional kernels later.

### GPU-Specific Kernel Policy Registry

This is a **live per-GPU adaptation contract**, not a historical notes section.
Every time this project touches a new card, add or update the card/family rule
here and the machine-readable rule in `rwkv7_hf/kernel_policy.py`. Do not leave
new hardware as an implicit "works on my GPU" case.

For every card that is developed, rented, borrowed, or used for validation,
record the following before claiming support:

1. Exact identity: GPU name, SM/ROCm target, driver, CUDA/ROCm, PyTorch, Triton,
   model checkpoint, dtype, and batch/prompt/decode matrix.
2. Runtime policy: `rwkv7_hf/kernel_policy.py` classification plus default-on
   and default-off kernels for that card/family.
3. AGENTS contract: this section must say which kernels are allowed by default,
   which are opt-in only, and which benchmark rows are mandatory.
4. Evidence rows: append `bench/results.jsonl` rows for functional smoke,
   decode, prefill, cache, and quant axes that are being claimed.
5. Analyzer support: update `bench/analyze_results.py` / summaries whenever a
   new axis, gate, or card-local metric is added.

Environment variables always override the policy. The policy is only the safe
default selected when the user does not set explicit flags. Policy coverage and
validation are separate: a family can have a conservative routing rule before it
has production evidence, but it is not a validated production target until the
required exact-card rows exist in `bench/results.jsonl`.

Current exact-card evidence status:

- V100 (`sm_70`): active regression baseline; preserve training/PEFT/TRL/cache
  and decode greedy-match rows before changing defaults.
- RTX 4090 (`sm_89`): active Ada consumer validation card; native fused prefill
  scan plus state-prep fusion is promising under explicit A/B flags, while
  WAVG/projection fusion is still opt-in because shallow LoRA grouping regresses
  end-to-end even when isolated rows improve. Full-head scan+output-prep fusion
  is also negative telemetry on Ada because it gives up the split-row scan tile
  that currently keeps prefill occupancy acceptable. Per-layer bsz=1 breakdown
  shows the prefill gap is broad across layers, so pursue reusable per-layer
  fusion patterns rather than layer-specific patches.
- RTX 5070 Laptop / RTX 5090 / 50-series (`sm_120` observed): touched
  Blackwell path. RTX 5090 now has HF load/generate, HF API, native-prefill,
  dynamic batching, W8/W4 functional quant, native mm8/mm4 benchmark rows,
  native/no-FLA Trainer + PEFT LoRA, bsz sweep, and Blackwell
  Triton/torch.compile compatibility evidence under
  `bench/5090_blackwell_hf_matrix_20260704/` and
  `bench/5090_blackwell_native_quant_20260704/`; keep adding exact-card
  rows when new 50-series kernels are claimed. Native no-FLA compatibility is
  important because some FLA training kernels can be architecture-limited;
  fusion wins must be re-proven end-to-end on each 50-card.
- A800 (`sm_80`): touched Ampere server validation card; 0.4B / 1.5B /
  2.9B bsz=1/2/4 batch sweep and W8/W4 memory-policy quantization rows exist
  on `NVIDIA A800-SXM4-80GB`, plus 0.1B generate/API/PEFT/alignment/Trainer/
  SFT/DPO/GRPO rows, 7.2B HF `larger_model_smoke`, 0.4B single-GPU and
  2-GPU ZeRO-2/3 base/checkpoint-resume rows, and native mm8/mm4 rows for
  0.4B / 1.5B / 2.9B / 7.2B / 13.3B.
  Keep the Ampere defaults conservative: output fusions remain allowed,
  prefill-scan, projection/LoRA, and quantized-speed fusions stay opt-in until
  exact-card sweeps prove end-to-end value.
- RTX A6000 (`sm_86`): touched Ampere workstation validation card; 2026-07-04
  rows on `NVIDIA RTX A6000` cover 0.1B core smoke, 0.4B / 1.5B / 2.9B /
  7.2B fp16+bf16 load/generate and batch sweep, bnb W8/W4 functional/footprint
  rows with slower decode telemetry, native mm8/mm4 decode telemetry, 0.4B /
  1.5B / 2.9B Trainer/SFT/DPO and HF checkpoint resume, plus 2x A6000
  ZeRO-2/ZeRO-3 base and resume to 2.9B.
  These rows validate the conservative Ampere defaults on `sm_86`; they do not
  promote prefill-scan, projection/LoRA, or quantized-speed kernels.
- GTX 1080 Ti (`sm_61`): Pascal smoke evidence exists for 0.1B / fp16 /
  default policy on one card. The safe default is native/no-FLA compatibility
  because FLA/Triton RWKV-7 kernels can emit `sm_70` PTX features on Pascal.
  Bnb 8/4-bit loading and decode speed rows exist but are slower than fp16, so
  bnb remains a memory/compatibility fallback. Repository-native mm8/mm4 rows
  exist for 0.1B with `lm_head` quantized and near-fp16 decode.
- Turing/Hopper/AMD: registry rules exist, but support remains TODO
  until exact-card rows are added.

#### Per-GPU adaptation checklist

Run this checklist for every new GPU before marking it as supported:

- Functional: import/from_pretrained, `generate(use_cache=True)`,
  `rwkv7_forward_token`, batch cache, dynamic batch select/reorder/drop,
  chunked prefill, save/reload, and greedy-match decode.
- Decode performance: `bench_batch_sweep.py` for `bsz=1/2/4/8`,
  `bench_native_graph_overhead.py`, native_graph cache hit/miss telemetry, and
  per-step tok/s/latency.
- Prefill performance: `bench_native_prefill_scan.py` when fast prefill is
  claimed; compare HF/FLA prefill against native prefill and record cache handoff
  correctness.
- Fused kernels: fused recurrent-output and fused output integration smokes;
  projection/LoRA/layout sweeps before any projection default; full native_graph
  end-to-end rows before promotion.
- Quantization: W8/W4 footprint, speed, greedy/quality rows. Microbench wins are
  never enough for a speed claim; require end-to-end decode evidence.
- Training, if claimed: HF Trainer, PEFT LoRA, TRL SFT/DPO/GRPO, checkpoint
  resume, and ZeRO-2/ZeRO-3 smoke on the relevant card or multi-GPU setup.
- Promotion rule: do not enable a fused/quant kernel by default unless exact-card
  rows show correctness plus non-negative end-to-end value across the claimed
  batch sizes. If bsz=1 regresses, keep the kernel opt-in even if bsz=4/8 wins.

#### Pascal / GTX 10 / P100 (`sm_60`/`sm_61`)

- Policy family: `pascal`.
- Default stance: compatibility-first; Pascal lacks the newer tensor-core path.
- Default-on: `fast_cache` only.
- Default-off: fused recurrent/output/projection/LoRA/prefill-scan kernels.
- Required validation: common functional checklist plus default native/no-FLA
  decode smoke on the exact card. Native graph / fused-kernel smokes are opt-in
  only and do not promote defaults without Pascal rows.
- GTX 1080 Ti evidence: 2026-07-03, 0.1B, fp16, one `sm_61` GPU, driver
  `550.127.05`, `nvidia-smi` CUDA `12.4`, PyTorch `2.7.1+cu118`,
  Transformers `5.12.1`, bitsandbytes `0.49.2`, FLA `0.5.1`;
  `smoke_hf_generate`, `test_hf_api_contract`, bnb W8/W4 quantized inference,
  bnb W8/W4 speed, native mm8/mm4 speed, `bench_speed`, and bsz 1/2/4
  `bench_batch_sweep` pass under the default native/no-FLA route. Optional 0.4B
  fp16 `bench_speed` also passes. Training was not run.
- Quant rule: current bnb W8/W4 rows are slower than fp16. Native mm8/mm4
  `speed` policy (`lm_head` only) can be used for the "footprint drops while
  decode is not slower" acceptance lane when exact-card rows also include
  logits/greedy-token parity. Native `memory` policy remains a footprint lane,
  not a speed claim, until fused quantized block kernels beat fp16.
- Promotion rule: never inherit V100/4090/5070 fused defaults without Pascal rows.

#### Volta / V100 (`sm_70`)

- Policy family: `volta`.
- Role: current regression baseline and conservative production-smoke target.
- Default-on: `fast_cache`, `fused_recurrent_output`, `fused_output`.
- Default-off: `fused_recurrent`, `fused_prefill_scan`, `fused_output_project`,
  `fused_projection`, `fused_wag_lora`, `fused_wavg_lora`.
- Required validation: functional checklist plus HF Trainer, PEFT LoRA, TRL
  SFT/DPO/GRPO, checkpoint resume, decode greedy-match, cache telemetry, and
  Albatross A/B rows when available.
- Quant rule: W8/W4 memory rows are valid footprint evidence. Treat bnb and
  native `memory` rows as non-speed paths unless they beat fp16. Native
  `speed` policy may be reported as the speed-acceptance lane only with
  card-local footprint, speed, and logits/greedy-token parity rows.
- Promotion rule: any default change must preserve V100 training and decode rows.

#### Turing / RTX 20 / T4 (`sm_75`)

- Policy family: `turing`.
- Default stance: Volta-safe output fusions can be attempted, but performance is
  not claimed without Turing rows.
- Default-on: `fast_cache`, `fused_recurrent_output`, `fused_output`.
- Default-off: prefill-scan, output-project, projection, WAG/WAVG LoRA fusions.
- Required validation: common functional checklist, bsz sweep, native_graph
  overhead, quant footprint/speed, and cache hit-rate rows.
- Promotion rule: projection/LoRA fusions stay opt-in until exact-card
  native_graph end-to-end speedup is measured.

#### Ampere / A100 / A800 / RTX 30 (`sm_80`/`sm_86`)

- Policy family: `ampere`.
- Default stance: stable output/recurrent-output fusions may be enabled; larger
  batch, training, and quant behavior must be tuned per exact card.
- Default-on: `fast_cache`, `fused_recurrent_output`, `fused_output`.
- Default-off: prefill-scan by default, output-project, projection, WAG/WAVG LoRA
  fusions.
- A800 adaptation rule:
  - `NVIDIA A800-SXM4-80GB` has 0.4B / 1.5B / 2.9B fp16 HF adapter evidence
    for bsz=1/2/4 native_graph decode and W8/W4 memory-policy quantization,
    plus 0.1B generate/API/PEFT/alignment/Trainer/SFT/DPO/GRPO smokes, 7.2B
    standard loading/generation, 13.3B bnb W8/W4 quantized smoke, 0.4B
    single-GPU and 2-GPU ZeRO-2/3 base/checkpoint resume, and native mm8/mm4
    rows for 0.4B / 1.5B / 2.9B / 7.2B / 13.3B. These rows validate the
    conservative Ampere defaults only; they do not promote prefill-scan,
    output-project, projection, LoRA, or quantized-speed kernels.
  - Latest 2.9B prompt128/decode8 native_graph decode rows are `93.6`,
    `199.1`, and `388.5` tok/s for bsz=1/2/4, with peak VRAM `6428.9`,
    `7262.5`, and `8906.6` MiB. W8/W4 reduce 2.9B model footprint from
    `5622.4` MB to `3222.4`/`2022.4` MB but remain slower than fp16. Native
    mm8/mm4 reduce 2.9B model footprint from `5622.4` MB to `3865.7`/`2985.7`
    MB, but decode falls from `110.7` tok/s fp16 to `20.5`/`19.5` tok/s. The
    default `8_000_000` native-mm gate replaces every per-layer FFN
    `key`/`value` matrix plus `lm_head`; A800 microbench rows show the current
    Triton dequant-GEMV kernels do not beat fp16 cuBLAS on those shapes. A
    `50_000_000` gate leaves only `lm_head` quantized and is roughly neutral for
    1.5B/2.9B decode, but saves much less footprint. Larger native mm8/mm4 rows
    also reduce footprint but remain slower than fp16: 7.2B falls from `36.1`
    tok/s fp16 to `17.0`/`15.9` tok/s, and 13.3B falls from `10.2` tok/s fp16
    to `7.7`/`8.6` tok/s. Quant speed is still unsolved on A800.
- RTX A6000 adaptation rule:
  - `NVIDIA RTX A6000` (`sm_86`, 48GB) has 2026-07-04 HF adapter rows in
    `bench/results.jsonl` for 0.4B / 1.5B / 2.9B / 7.2B fp16+bf16 smoke,
    native_graph batch sweep, bnb W8/W4, and native mm8/mm4 decode. Single-GPU
    Trainer/SFT/DPO/resume rows pass for 0.4B / 1.5B / 2.9B; 2x A6000
    ZeRO-2/ZeRO-3 base and resume rows pass to 2.9B.
  - Latest fp16 native_graph decode rows are: 0.4B bsz1/8 `286.3`/`1750.2`
    tok/s, 1.5B bsz1/4 `149.9`/`504.1` tok/s, 2.9B bsz1/2
    `81.7`/`148.1` tok/s, and 7.2B bsz1/2 `41.4`/`78.7` tok/s. 7.2B fp16 and
    bf16 load/generate fit within 48GB with `13997.8` MiB peak in the smoke
    row. Bnb W8/W4 reduce footprint but are slower than fp16; native mm8/mm4
    rows are real decode telemetry, not a production quant-speed win on larger
    models.
- Required validation: common functional checklist, larger-batch prefill, state
  cache reuse/hit-rate rows, W8/W4 rows, and ZeRO-2/ZeRO-3 smoke when training is
  claimed.
- Promotion rule: do not assume V100/4090 block sizes; run Ampere block/layout
  sweeps before changing defaults.

#### Ada / RTX 40 / 4090 (`sm_89`)

- Policy family: `ada`.
- Role: high-end consumer validation target.
- Exact-4090 default-on: `fast_cache`, `fast_prefill`, fixed-shape prefill
  graph, split prefill scan, fused prefill state prep/output/shift,
  `fused_recurrent_output`, and decode `fused_output`.
- Other Ada default-off: prefill graph/scan and unmeasured prefill fusions;
  all cards keep the compatible dense fallback. `fused_output_project` and
  projection/LoRA experiments remain opt-in everywhere.
- 4090 adaptation rule:
  - cuBLAS/torch remains the baseline for shallow R/K/V projection; split-K/layout
    prototype rows were slower and must stay telemetry-only.
  - Fixed-shape CUDA Graph replay is the promoted serving path. It captures the
    complete native prefill layer sequence and removes the launch gaps that made
    isolated positive kernels regress end-to-end. Use
    `rwkv7_warmup_fast_prefill()` before serving and clear the graph cache after
    changing capture-affecting environment settings.
  - For 0.4B/fp16/prompt512, exact-4090 defaults use scan tile 4 at bsz1 and
    tile 8 at bsz>=2, both with four warps. Public API rows after sequence
    shift/state and ReLU² fusion are `64511.2 tok/s` at bsz1 and `107870.1
    tok/s` at bsz4. Bsz4 is `1.007x` the current same-session Albatross row and
    `0.916x` the historical strongest row. 1.5B bsz1 previously reached
    `32357.8 tok/s`. Greedy/cache handoff, HF generate, dynamic-cache, and
    full-vs-chunked tests pass.
  - Keep separate fused state prep + split scan inside the graph. The combined
    state-prep+scan kernel is slower under graph replay. Fused output prep and
    fused shift mix are positive only as part of the whole captured sequence.
  - Do not generalize the 4090 tile/defaults to 4070/4080 or another GPU family
    without card-local rows. Do not claim universal parity: same-session B4 is
    closed, but it remains below the historical strongest Albatross row.
  - Latest fine prefill breakdown splits dense R/K/V projection separately:
    bsz=1 prompt512 has scan `7.6627ms`, LoRA sum `6.2419ms`, norm/shift/mix
    `3.8281ms`, state-prep `3.1581ms`, and dense R/K/V sum `2.2056ms`. The
    next Ada prefill experiment should fuse across these buckets rather than
    optimizing cache or a single shallow pointwise kernel.
  - Prefill output-prep fusion remains negative on the uncaptured direct path,
    but is positive together with shift mix inside whole-prefill graph replay;
    this promotion is exact-4090-only and does not reverse the cross-card rule.
  - Prefill WAVG LoRA grouping must stay telemetry-only:
    `RWKV7_NATIVE_PREFILL_FUSED_WAVG_LORA=1` improves isolated `B*T=512`
    microbench but regresses end-to-end bsz=1 prefill (`21773.4` tok/s) and is
    disabled for larger flattened rows by default
    (`RWKV7_NATIVE_PREFILL_FUSED_WAVG_LORA_MAX_M=1024`).
  - Deeper R/K/V + W/A/G/V-gate projection fusion is opt-in: current 4090
    decode rows show greedy correctness, bsz=4 speedup, and bsz=1 regression,
    so default promotion is blocked by the min-batch gate. Prefill-shaped
    R/K/V+W/A/G rows are worse (`0.6823x` at bsz=1/T512 and `0.1471x` at
    bsz=4/T512), so do not wire the current two-launch projection prototype
    into prefill.
- Required validation: common functional checklist, bsz=1/2/4/8 decode matrix,
  prefill scan A/B if fast prefill is claimed, quant end-to-end rows, and
  Albatross-ratio reporting.
- Promotion rule: the minimum speedup across claimed bsz values must be >= 1.0x
  with greedy match before enabling a new fusion by default.

#### Hopper / H100 (`sm_90`)

- Policy family: `hopper`.
- Default stance: expected fast server path, but H100 is not tuned until H100 rows
  exist.
- Default-on: `fast_cache`, `fused_recurrent_output`, `fused_output`.
- Default-off: prefill-scan by default, output-project, projection, WAG/WAVG LoRA
  fusions.
- Required validation: common functional checklist, larger model rows, large
  batch/chunked prefill, W8/W4/FP8-like precision and speed rows, PP/TP serving
  smoke if claimed, and ZeRO-2/ZeRO-3 smoke if training is claimed.
- Promotion rule: do not reuse 4090 or Blackwell block sizes without H100 sweeps.

#### Blackwell / RTX 50 / 5070-5090 (`sm_100+`, observed `sm_120`)

- Policy family: `blackwell`.
- Role: next consumer-generation compatibility target.
- Default-on: `fast_cache`, `fused_recurrent_output`, `fused_output`.
- Default-off: `fused_output_project`, projection/LoRA fusions, and prefill-scan
  as a default.
- 50-series adaptation rule:
  - Always include native/no-FLA smokes because FLA kernels may fail or regress on
    new architectures even when inference forward works.
  - Keep projection/LoRA/quant fusions opt-in until exact 50-card end-to-end rows
    prove both correctness and speed. Isolated kernel wins do not promote.
  - Quantization must include footprint, long/short decode speed, and greedy or
    quality rows. Treat bnb as a compatibility/memory baseline, not a fast path.
- Mandatory before claiming support: import/generate, fast decode, dynamic batch,
  chunked prefill, bnb W8/W4 functional inference, `triton_compat` remote-code
  import on early sm_120 stacks, native_model no-FLA fallback/training smoke,
  and exact-card fused-kernel A/B rows. For RTX 5090 specifically, keep
  `bench/run_5090_hf_validation.sh` as the one-command smoke matrix and store
  its dated output under `bench/5090_blackwell_*`.
- Promotion rule: promote only kernels with exact-card greedy match and min bsz
  speedup >= 1.0x; otherwise leave them opt-in/telemetry.

#### AMD / ROCm / HIP

- Policy family: `amd_hip`.
- Default stance: compatibility-first; CUDA/Triton-only kernels are off.
- Default-on: `fast_cache` only.
- Default-off: CUDA native_graph fused kernels and CUDA-only quant speed paths.
- Required path: pure PyTorch/native_model or ROCm-supported fallback first, then
  HIP-specific kernels only after evidence.
- Required validation: ROCm import/generate, pure PyTorch/native_model
  forward/backward, cache smokes, and HIP-specific speed rows before parity
  claims.
- Quant rule: no AMD quant performance claim until HIP-specific W8/W4 rows exist.

#### Apple Silicon / MPS / MLX / CoreML

- Policy family: `apple_mps` in `rwkv7_hf/kernel_policy.py`.
- Default stance: native/no-FLA HF compatibility; all CUDA/Triton fused kernels
  off. MLX/Metal and CoreML are explicit sibling backends, not CUDA-policy
  fallbacks.
- Default-on: `fast_cache` and automatic native-model fallback when MPS is
  available. Default-off: CUDA native-graph kernels and CUDA/bitsandbytes quant.
- Current exact-device evidence: MacBook Air / Apple M5 / 16GB / macOS 26.5.
  MPS HF/PEFT/Trainer/TRL and MLX recurrent/session/quant rows exist. CoreMLTools
  9.0 now has live 0.1B/0.4B `stateful-multifunction` rows: MLState transfer,
  alternate chunk split, and HF fp32 greedy tokens all match exactly. Initial
  CoreML INT8 reduces package size to about `0.45x`/`0.36x` and preserves the
  short greedy gates, but decode remains about `0.95x`/`0.98x` fp32; 0.1B
  INT4/LUT4 reduce package size further while failing the current HF greedy gate.
- First live same-device Qwen gate: M5/16GB, Ollama 0.31.1
  `qwen3.5:0.8b-mlx` vs RWKV-7 0.4B MLX, 128/512 prompt chars and decode32.
  retained fp16 decode is about `0.82x/0.92x` Qwen but prefill only `0.090x/0.049x`;
  W4 lowers RWKV peak to about `0.568x` fp16 while decode drops to about
  `0.60x` Qwen. Treat Qwen peak memory and quality as open gates; `/api/ps`
  loaded memory is telemetry, not a peak substitute.
- MLX prompt graph-evaluation batching is now an explicit, parity-gated seam.
  The model-level conservative default remains interval `1`; the Apple/Qwen
  acceptance wrapper uses interval `2`. On M5, 512-character interleaved rows
  keep logits, every recurrent/cache tensor, seen-token count, and next token
  exact while interval `2` changes median prefill by `1.05x/1.28x/1.09x` for
  0.1B/0.4B/1.5B fp16 and `1.38x/1.32x` for 0.4B/1.5B W4. This removes host
  synchronization overhead only; it does not replace the required MLX/Metal
  DPLR/WY chunk-summary, prefix-combine, and chunk-apply/output implementation.
- CoreML state contract: state is fp16-only, so WKV uses fp16 high + fp16
  residual tensors; attention/FFN previous inputs and `v_first` are separate
  states. `--coreml-compute-precision auto` must resolve to fp32 for stateful
  exports until the fp16 greedy mismatch is fixed.
- Required validation before Apple production claims: exact M-series identity,
  MPS/MLX/CoreML versions, prompt/decode matrix, peak memory, chunk/state/HF
  parity, W8/W4 footprint and speed, and confirmed runtime placement. Never
  treat `CPU_AND_NE` eligibility as proof that ANE executed the graph.
- Promotion rule: keep fp16 stateful CoreML and quantized CoreML opt-in until
  exact-device long-context greedy/quality gates pass; do not generalize M5 Air
  numbers to M-series Pro/Max/Ultra or iPhone/iPad.

### Quantized Inference

Required goals:

- Support 8-bit inference.
- Support 4-bit inference.
- Quantization must reduce memory usage.
- Quantized speed should be no slower than fp16 as much as possible.
- V100 may not be ideal for final int4/int8 speed validation because it lacks
  newer tensor core features.
- Card-validation PRs must report native `mm8`/`mm4` decode tok/s + footprint
  (PR #85/#88), not just bnb W8/W4. bnb is the generic fallback; `mm8`/`mm4`
  (fused Triton dequant-GEMV) is this repo's quant path and the one that must
  be validated per card. If `mm8`/`mm4` cannot run on a card (e.g. Pascal
  sm_61 Triton `.evict_last` limits), record that as the conclusion instead.

## Current State

Completed first-stage HF wrapper adaptation:

- Downloaded official RWKV-7 0.1B checkpoint.
- Verified official `rwkv` package can load and generate on V100.
- Converted 0.1B checkpoint to Hugging Face-style `model.safetensors`.
- Added remote-code wrappers for config/model/tokenizer.
- Verified `AutoTokenizer` loading.
- Verified `AutoModelForCausalLM` loading.
- Verified `generate(use_cache=True)`.
- Verified PEFT LoRA forward/loss/backward smoke test.
- Compared HF logits with official RWKV path:
  - top-5 token IDs match
  - fp16 cosine similarity around `0.999996`
  - max absolute difference around `0.047`

The default production-facing wrapper uses FLA (`flash-linear-attention`) as
backend. The opt-in `RWKV7_NATIVE_MODEL=1` path loads the experimental
pure-PyTorch `NativeRWKV7ForCausalLM` backend for FLA-free compatibility work;
it is not yet the final performance backend.

Recent completed evidence:

- V100 training telemetry is recorded for HF Trainer, TRL SFT, TRL DPO, and TRL
  GRPO-style smoke paths.
- The experimental native/no-FLA backend has explicit HF ecosystem smokes for
  HF Trainer, TRL SFT, TRL DPO, TRL GRPO, PEFT adapter save/load/merge,
  Trainer checkpoint resume, and bnb W8/W4 functional inference. These prove
  compatibility and regression coverage, not Albatross-level speed.
- Albatross A/B ingestion exists and analyzer output reports HF-vs-Albatross
  prefill/decode ratios.
- W8/W4 quantization rows record both canonical memory-target bnb behavior and
  `decode_hot` hybrid variants. The hybrid variants improve decode over generic
  bnb on V100 while remaining below fp16/native-graph speed. For native
  mm8/mm4, distinguish `memory` policy (maximum footprint reduction, may be
  slower) from `speed` policy (small but real footprint reduction, speed gate).
  Fused/native quantized projection kernels remain the main path for combining
  large footprint reductions with fp16-or-better speed.

## Important Paths

Local GitHub checkout:

```bash
/Users/wangyue/Documents/vllmsp/rwkv7-hf-adapter
```

Server project checkout:

```bash
/home/data/wangyue/projects/rwkv7-hf-adapter
```

Server model files:

```bash
/home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-20260129-ctx8192.pth
/home/data/wangyue/models/rwkv7/rwkv_vocab_v20230424.txt
/home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf
```

Server environments:

```bash
/home/data/wangyue/envs/rwkv7
/home/data/wangyue/envs/rwkv7-cu118
```

Reference repos on server:

```bash
/home/data/wangyue/projects/RWKV-LM
/home/data/wangyue/projects/Albatross
/home/data/wangyue/projects/flash-linear-attention
```

## Development Environment

Use this for the current HF wrapper work:

```bash
source /home/wzu/anaconda3/etc/profile.d/conda.sh
conda activate /home/data/wangyue/envs/rwkv7
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/home/data/wangyue/projects/flash-linear-attention:/home/data/wangyue/projects/rwkv7-hf-adapter:$PYTHONPATH
```

For official RWKV / CUDA extension smoke tests:

```bash
source /home/wzu/anaconda3/etc/profile.d/conda.sh
conda activate /home/data/wangyue/envs/rwkv7-cu118
export RWKV_V7_ON=1
export CUDA_VISIBLE_DEVICES=0
```

## Common Commands

### Convert checkpoint to HF format

```bash
python /home/data/wangyue/projects/rwkv7-hf-adapter/scripts/convert_rwkv7_to_hf.py \
  --input /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-20260129-ctx8192.pth \
  --output /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf \
  --vocab-file /home/data/wangyue/models/rwkv7/rwkv_vocab_v20230424.txt \
  --precision fp16 \
  --attn-mode chunk
```

### HF generate smoke test

```bash
python /home/data/wangyue/projects/rwkv7-hf-adapter/tests/smoke_hf_generate.py \
  --model /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf
```

### PEFT LoRA smoke test

```bash
export TORCHDYNAMO_DISABLE=1
python /home/data/wangyue/projects/rwkv7-hf-adapter/tests/test_peft_lora.py \
  --model /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf \
  --attn-mode fused_recurrent
```

## Engineering Rules

- Do not commit model weights, `.pth`, `.safetensors`, `.bin`, `.gguf`, checkpoints, or generated large artifacts.
- Keep conversion scripts reproducible.
- Keep smoke tests small and runnable on V100.
- Always compare against official RWKV-LM or `rwkv` package outputs when changing math or weight mapping.
- Treat FLA backend as a temporary first-stage dependency until native Transformers implementation is ready.
- Be explicit about state cache behavior: RWKV recurrent state is not Transformer KV cache.
- For PEFT/TRL compatibility, prefer standard HF model signatures and return types.
- For HF serving-style helpers, design state cache allocator/gather/scatter/reorder/release explicitly.

## Next Milestones

1. Convert and validate larger RWKV-7 checkpoints, including the 13.3B gate.
2. Keep official RWKV vs HF logits/generation alignment tests green.
3. Keep `save_pretrained` / reload roundtrip tests green.
4. Expand PEFT / Trainer / TRL SFT/DPO/GRPO smoke tests into multi-batch and gradient-accumulation checks.
5. Move HF performance work into the native fused backend: train_temp-style
   fp16 kernel boundaries, GPU-specific layout/autotune, and DPLR/chunked
   prefill for bsz=1 prompt-prefill. Keep wrapper/cache work to compatibility
   and telemetry fixes.
6. Finish HF quantized W8/W4 inference as two explicit lanes: `speed` policy
   for the acceptance target (footprint lower than fp16, W8/W4 decode not
   slower, logits aligned), and `memory` policy for maximum footprint reduction
   that still needs fused quant prefill before it can claim fp16-or-better
   speed. Exact-4090 0.4B `speed` rows now pass both prefill and decode: W8
   payload `0.926x`, W4 payload `0.891x`; continue the same all-phase gate on
   larger models/cards. The W4 `memory` lane keeps `0.399x` payload and faster
   decode but does not yet pass prefill.
7. Validate on more GPUs and larger batch sizes.
8. Start native Transformers implementation under `src/transformers/models/rwkv7/` style layout.
9. Remove mandatory FLA dependency from the final HF implementation.
