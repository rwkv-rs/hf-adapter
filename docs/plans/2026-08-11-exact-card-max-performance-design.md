# Exact-Card Maximum Performance Design

## Scope

Drive the RWKV-7 Hugging Face adapter to the highest reproducible end-to-end
inference performance available on one exact RTX 5070 Laptop GPU and the
project's V100 baseline, without weakening the public HF contract or allowing
one card to inherit another card's unverified schedule. Training performance is
a second phase; the first phase optimizes prompt prefill, cached decode, total
latency, model footprint, and peak VRAM.

The work has two lanes. The `production` lane may become a default only after
the full correctness and regression matrix passes. The opt-in `max_perf` lane
may use narrower model, dtype, batch, and sequence gates, but must still pass
logit, recurrent-state, cache-handoff, and long greedy checks. Microbenchmarks
select candidates; only full-model A/B rows can establish a speed result.

## Exact Hardware Boundaries

- RTX 5070 Laptop: exact device name, `sm_120`, 8GB, Windows/WDDM. Initial
  matrix: 0.4B, 1.5B, and 2.9B FP16; B1/B2/B4/B8 where the checkpoint fits;
  prompt 128/512 and decode 128. Power limit, clocks, temperature, driver,
  PyTorch, CUDA, Triton, and background-memory pressure are recorded with each
  accepted artifact.
- Tesla V100-PCIE-32GB: exact V100 identity and software stack must be captured
  on the server. Initial matrix: 1.5B, 2.9B, and 7.2B FP16 plus already
  productionized MM4 profiles; B1/B2/B4/B8 where memory permits.
- No RTX 4080, RTX 5090, generic Blackwell, or generic sm70 policy is widened
  from these results. Environment variables remain explicit benchmark
  overrides.

## Optimization Loop

1. Freeze a current-main baseline with three independent model loads and
   synchronized CUDA timing.
2. Attribute prefill and decode time to projections, recurrent update,
   normalization/mix, FFN, output preparation, graph capture/replay, and
   quantization/dequantization.
3. Test the narrowest reusable candidate against the unchanged baseline.
4. Reject any candidate that fails cache, state, logits, greedy, memory, or
   cross-card isolation gates.
5. Promote only repeated end-to-end wins, then rerun the full exact-card matrix.
6. Repeat on the new baseline until the plateau rule is satisfied.

The first 5070 candidate is the grouped B8 W/A/V tensor-core BMM path recently
accepted for desktop RTX 4080. It will be enabled only through the existing
override and benchmarked on 5070 before any policy change. Subsequent work is
chosen from the measured top hotspot, with priority given to deeper native
fusion and reduced state/activation traffic rather than wrapper changes.

## Correctness and Promotion Gates

- First-step and multi-step logits cosine at least `0.9999`.
- Recurrent-state comparison in FP32 plus dtype-appropriate absolute-error
  reporting; state dtype changes require a dedicated long-trace gate.
- Exact greedy token match for at least 128 decode steps per accepted cell,
  extended to 512 steps for a new default.
- HF `generate(use_cache=True)`, `rwkv7_forward_token`, cache handoff, dynamic
  batch select/reorder/drop, and chunked-prefill smoke remain passing.
- Three independent-process A/B repetitions; candidate median speedup must be
  at least `1.01x`, with no repeated run below `1.00x` for a production default.
- Peak VRAM must fit the target matrix and any material memory regression must
  be disclosed. Quantized speed claims additionally require lower physical
  model footprint than FP16.

## Plateau Rule

An exact-card/profile optimization cycle is considered squeezed only when the
profiler accounts for the dominant runtime, every feasible top-hotspot
candidate has either failed a correctness/memory gate or produced less than a
1% repeated end-to-end gain, and a final baseline rerun reproduces within 3%.
The artifact records the rejected candidates and limiting hardware resource so
future CUDA, Triton, driver, or power-envelope changes can reopen the profile.

