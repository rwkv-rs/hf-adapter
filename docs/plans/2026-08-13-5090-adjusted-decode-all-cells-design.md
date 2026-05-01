# RTX 5090 parameter-adjusted Decode all-cell design

## Goal

Produce a reproducible single-card RTX 5090 comparison in which RWKV-7 beats
the already validated strong Qwen3.5 reference in parameter-adjusted Decode in
every one of the 48 matched cells.

The hard acceptance condition is strict:

```text
adjusted_decode_ratio =
    (rwkv_decode_tokps_total / qwen_decode_tokps_total)
    * (rwkv_active_parameters / qwen_active_parameters)

min(adjusted_decode_ratio over all 48 cells) > 1.00
```

`>= 1.00`, a rounded `1.00`, or a median-only win does not close the goal. A
target margin of at least `1.05` is preferred so normal measurement variance
does not turn the formal rerun red.

## Immutable reference

Use the 48-row RTX 5090 Qwen3.5 artifact in
`bench/5090_qwen35_best_optimized_hf_v1_20260813/` without weakening it or
substituting an older module-call baseline. Its contract is:

- Dense FP16, one RTX 5090, batch 1/8, prompt 128/512/2048, decode 128/512,
  prefill chunk 512, warmup 3, runs 7, per-cell median.
- Official FLA plus official `causal_conv1d`, with all fast-path gates passing.
- Qwen 0.8B/2B Decode uses StaticCache + Inductor max-autotune + CUDAGraph.
- Qwen 4B/9B Decode uses StaticCache + raw CUDA Graph after the same-cache
  numerical gate rejected the less stable Inductor route.
- All rows share one locked runtime signature and repository commit.

The reference remains a Qwen-only source artifact. The final paired artifact
must cite and validate it; rows must not be relabelled or hand-edited to make
the protocols appear identical.

## Candidate checkpoints and runtime

Use the exact converted FP16 checkpoints already hashed by the 3090/4090/5090
evidence:

- RWKV-7 g1d 0.4B, source `rwkv7-g1d-0.4b-20260210-ctx8192.pth`.
- RWKV-7 g1i 1.5B, source `rwkv7-g1i-1.5b-20260805-ctx16384.pth`.
- RWKV-7 g1i 2.9B, source `rwkv7-g1i-2.9b-20260805-ctx16384.pth`.
- RWKV-7 g1i 7.2B, source `rwkv7-g1i-7.2b-20260805-ctx16384.pth`.

The converted config and safetensors hashes must match the promoted historical
manifests before a row is admitted. Run the candidates in the exact Qwen
Python/PyTorch/CUDA/Transformers environment and record the same runtime-lock
fields in every row.

## Optimization policy

Start with the current best HF-compatible RWKV route: repo code, recurrent
state cache, native fused kernels, and CUDA Graph Decode. Preserve standard HF
loading and cache semantics.

Tuning is allowed at the exact-card model-and-batch policy level. A selected
route for a `(model_size, batch_size)` lane must be declared in a route manifest
and applied to all six prompt/decode cells in that lane. Do not select different
kernels after observing individual prompt/decode cell results. Environment
overrides, tile sizes, accumulation choices, grouped projection paths, graph
scope, and fallback behavior must all be recorded as requested/effective
fields.

Priority order:

1. Establish the 48-cell same-runtime native-graph baseline.
2. Tune the known weak B8 lanes first: 0.4B/0.8B and 1.5B/2B.
3. Preserve already passing 2.9B/4B and 7.2B/9B lanes while checking all six
   cells per batch, not just their medians.
4. Promote only a route that passes correctness and improves the complete
   declared lane. Keep negative probes as diagnostics outside the formal table.

Qwen parameters, route, samples, and timing scope are immutable during this
work.

## Correctness and evidence gates

Every candidate row must be fail-closed before its throughput is eligible:

- `status=pass`, Dense FP16, exact RTX 5090, expected model pair and shape.
- Logits finite; greedy tokens and recurrent cache handoff match the declared
  eager/native reference through warmup plus the full decode horizon.
- CUDA Graph requested/effective fields agree, graph launch is verified, and
  no silent eager fallback is accepted.
- Cache identity, batch shape, state progression, and final seen-token count
  are correct; B8 correctness prompts are distinct.
- Prefill and Decode each contain seven positive finite timing samples; stored
  medians and tok/s must reproduce arithmetically from those samples.
- Peak VRAM fits the card without reducing batch, prompt, decode, precision, or
  model size.
- Model hashes, source commit, runtime lock, environment, system identity, and
  route manifest are captured in the artifact.

## Final paired artifact

The final validator joins 48 RWKV candidate rows to the 48 immutable Qwen
reference rows by model pair, batch, prompt, and decode. It must reject duplicate
cells, runtime mismatches, unapproved reference matrices, route-manifest
mismatches, failed correctness fields, and any adjusted Decode ratio `<= 1.00`.

Output order is model size, GPU, B1/B8, prompt, Decode. Human-readable values use
zero decimal places for values at least 100 and one decimal place below 100;
machine-readable rows retain raw timing samples and unrounded recomputed ratios.

The report must distinguish:

- raw Prefill and raw Decode throughput;
- parameter-adjusted Prefill and Decode ratios;
- the independent best Prefill/Decode envelope from a continuous end-to-end
  cache-handoff route;
- diagnostic probes from the single promoted formal route per lane.

The work is complete only when the validator reports 48/48 strict adjusted
Decode wins and no correctness, runtime, route, coverage, or memory errors.
