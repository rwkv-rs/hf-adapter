# Native Fused Backend Roadmap

This roadmap is the next-agent contract for the RWKV-7 HF adapter performance
phase. It is intentionally scoped to the HF/Transformers adapter in this repo.

## 1. Scope and compatibility contract

- This repository is **HF adapter only**. Do not start native vLLM, SGLang, or
  other serving-engine integrations here; they are separate future projects.
- Keep standard HF behavior working: `AutoModelForCausalLM`, `generate`,
  `RWKV7StateCache`, PEFT, Trainer, TRL, save/load, quantized loading, dynamic
  batching, and chunked prefill remain the public contract.
- The performance target is to keep HF compatibility while moving the hot path
  to a **native fused backend**. The wrapper is the API shell; fused fp16 and
  fused quant kernels are the speed path.
- `native_jit` / `native_graph` / cache fast paths are baselines and fallback
  layers. New performance work should not be another round of Python-wrapper or
  cache-traversal micro-optimization.

## 2. Math alignment sources

Use the official/reference material as the math oracle before changing kernels:

- `rwkv_v7_numpy.py`: scalar/NumPy reference for RWKV-7 equations. Preserve the
  operation order, decay/clamp behavior, normalization, recurrent state update,
  and output equations when translating into fused kernels.
- `run_rwkv7_qwen35.py`: official end-to-end checkpoint/tokenization execution
  path for Qwen3.5-aligned RWKV-7 work. Use it to keep real-model logits,
  generation, and state handoff aligned rather than optimizing only synthetic
  tensors.
- Existing HF adapter tests and official RWKV/Albatross comparisons remain the
  guardrails. A fused kernel is not useful until it passes both local tensor
  correctness and end-to-end HF generation/cache correctness.

Minimum alignment evidence for a new fused boundary:

1. Local tensor check against the unfused HF/reference operation.
2. Prompt prefill state handoff into decode with greedy equality or documented
   fp16 tolerance.
3. Logit cosine/max-abs tolerance on real model weights.
4. Analyzer row in `bench/results*.jsonl` so the result is visible to future
   agents.

## 3. train_temp-style kernel boundary map

Use these boundaries as the next implementation map. Keep the names in design
notes and benchmark axes so agents can connect HF code, official math, and
training-kernel references.

### `tmix_mix6`

- Purpose: attention time-mix input preparation for the six mixed streams used
  downstream by recurrent attention and LoRA/projection work.
- Boundary: consume current hidden state, previous hidden state, and mix
  parameters; emit the six mixed tensors without changing math or hidden-state
  semantics.
- Rule: do not default a standalone shift/mix kernel just because the isolated
  op passes. Existing 4090 evidence says shallow shift-mix is not the missing
  bsz=1 prefill fix unless folded into a larger norm/shift/projection/state
  kernel.

### `kk_pre` / `state_prep`

- Purpose: prepare the recurrent-state inputs after projection/LoRA: decay `w`,
  key adjustment, `kk` normalization, value interpolation, and tensors consumed
  by the recurrent scan/update.
- Boundary: keep cuBLAS-backed projection valid while fusing the pointwise and
  per-head state-prep work; then use it as the bridge to deeper projection/LoRA
  fusion.
- Rule: keep this boundary available for variable-shape/cross-card kernels.
  Exact-4090 fixed-shape graph replay has closed bsz1 prompt512, while bsz4 and
  uncaptured paths still need cross-bucket projection/LoRA/scan-aware work.

### `lnx_rkvres_xg`

- Purpose: recurrent output preparation: normalize recurrent output, apply the
  R/K/V residual correction, and multiply by the gate stream before final
  output projection.
- Boundary: preserve the current profitable fused recurrent+output path for
  decode, while avoiding prefill designs that force a full head into one scan
  program.
- Rule: do **not** promote the full-head scan+output fused prefill path. It is
  negative telemetry because it destroys split-row scan occupancy on Ada.

### `cmix`

- Purpose: channel-mix / FFN path: channel shift/mix, key/value projections,
  activation, and residual add.
- Boundary: treat FFN-only fusions as telemetry unless they are folded into a
  larger per-layer fused pattern. Existing rows show shallow FFN-only kernels
  are slower than cuBLAS-backed paths.
- Rule: optimize only with end-to-end bsz=1/4 prompt512 evidence; microbench
  wins are insufficient.

### `clampw`

- Purpose: exact decay clamp / stabilization used by the RWKV-7 recurrence.
- Boundary: keep clamp placement, dtype behavior, and numerical tolerances
  aligned with `rwkv_v7_numpy.py` and official execution.
- Rule: never move, approximate, or remove clamp behavior for speed unless a
  reference-aligned correctness row proves equivalence on real weights.

## 4. Albatross-style GPU-specific layout and autotune

Use Albatross as the performance reference and copy its engineering principle:
layout and block choices are GPU-specific, not universal constants.

- Maintain exact-card policy. A V100, 4090, 5070/5090, A100, and H100 may need
  different scan tiles, split-row layouts, warp counts, projection grouping, and
  quant packing.
- Record the winning layout/autotune choice with the benchmark row: GPU name,
  compute capability, dtype, model size, batch size, prompt length, flags,
  `block_m`, `num_warps`, chunk size, and peak memory.
- Preserve split-row scan occupancy for 64-row heads unless a new end-to-end row
  proves a better layout. Do not replace split-row scan with a full-head fused
  scan+output path.
- Promotion rule: a kernel can become default only after exact-card end-to-end
  correctness plus non-negative speed across the claimed bsz set. Isolated
  microbench speedups remain telemetry.

## 5. DPLR / chunked prefill as the variable-shape breakthrough line

Exact-4090 fixed-shape CUDA Graph replay reaches Albatross parity at bsz1 /
prompt512, but it does not solve variable prompt shapes, cross-card portability,
or the remaining bsz4 gap. Treat DPLR/chunked prefill as that algorithmic path,
not another wrapper optimization.

Roadmap:

1. Build a reference-equivalent chunked/DPLR prefill formulation that produces
   the same final recurrent state and logits as sequential HF prefill.
2. Use chunk-local fused kernels for scan/state-prep work and an inter-chunk
   combine step so bsz=1 has enough parallelism.
3. Tune chunk sizes on 4090 fp16 prompt512 first, then carry the layout matrix to
   V100/A100/H100/Blackwell instead of assuming the same winner.
4. Keep `RWKV7StateCache` handoff exact: prefill output state must decode with
   greedy equality against the unfused path.
5. Track Albatross ratios for bsz=1 and bsz=4 separately. bsz=4 passing P1 does
   not close the bsz=1 prefill gap.

## 6. Explicitly forbidden directions

- Do not continue wrapper micro-optimization as the main performance plan. Fix
  wrapper bugs and compatibility regressions only.
- Do not start or gate native vLLM/SGLang work in this repository.
- Do not promote the full-head scan+output fused prefill path. It is documented
  negative telemetry and should not be generalized.
- Do not make quantized-speed claims from bitsandbytes or generic dequant rows.
  Quant performance waits for a native fused quant kernel that beats fp16
  end-to-end while preserving footprint and correctness.
- Do not default shallow projection, LoRA, shift-mix, FFN, or output-project
  probes based only on isolated microbenchmarks.
- Do not hide regressions by reporting only bsz=4/8 wins. bsz=1 remains a first
  class target.

## 7. Required validation loop

Primary next-card loop: **RTX 4090, fp16, bsz=1/4, prompt512**.

1. Correctness:
   - tensor-level fused-vs-unfused max-abs/cosine checks;
   - HF `generate(use_cache=True)` greedy equality where applicable;
   - prefill state handoff into decode;
   - no cache semantic regressions for dynamic batch/reorder/drop.
2. Prefill:
   - run prompt512 bsz=1 and bsz=4 rows through `bench/bench_native_prefill_scan.py`;
   - use `bench/bench_native_prefill_breakdown.py --fine-attn --layer-breakdown`
     when a prefill change claims a bottleneck shift;
   - compare against Albatross ratios in the analyzer.
3. Decode:
   - run 4090 fp16 bsz=1/4 decode rows through the existing native-graph
     validation path (`bench/run_4090_fused_backend_validation.sh` or the
     equivalent individual benches);
   - include correctness and cache hit/skip telemetry, not only tok/s.
4. Memory:
   - record peak allocated/reserved VRAM or script-reported peak memory for
     prefill, decode, and quant rows;
   - quant rows must report footprint ratio plus speed ratio.
5. Analysis:
   - run `python bench/analyze_results.py --results <jsonl> --device "NVIDIA GeForce RTX 4090" --dtype fp16`;
   - inspect `fused_backend_targets`, Albatross prefill/decode ratios,
     correctness rows, missing axes, memory rows, and `next_focus` before
     choosing the next kernel.

A change is promotable only when the validation rows show correctness, memory is
not worse than the claimed mode, and end-to-end bsz=1 and bsz=4 behavior matches
the stated target.
