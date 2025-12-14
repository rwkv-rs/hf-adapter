# RWKV-7 HF adapter benchmark summary

This is the canonical **promoted-results summary**. It intentionally excludes
exploratory tuning chronology. Raw rows, logs and negative experiments remain
in [`bench/`](bench/); platform interpretation lives in
[`docs/PERFORMANCE.md`](docs/PERFORMANCE.md).

Last updated: **2026-07-18**.

## Benchmark contract

A result may be promoted here only when it records hardware/runtime versions,
model/checkpoint and dtype, shape/batch, exact command, raw output, correctness
checks and memory/throughput telemetry. Comparison claims must use compatible
shapes and retain their reference source.

Status vocabulary:

- **PASS:** the named fail-closed gate passed.
- **PARTIAL:** functionality or selected shapes passed; the full target did not.
- **SMOKE:** execution proof only, not a production performance claim.

## Production-close overview

| Platform | Scope | Correctness / quality | Performance | Result |
|---|---|---|---|---|
| V100 32GB | dense/Qwen lanes plus 1.5B/2.9B/7.2B packed-MM4 decode | greedy/cache gates; MM4 complete-sequence and repeat hashes pass 21/21 | Albatross P1; full-FLA Qwen gates; MM4 decode minima `1.0255x/1.0111x/1.0810x` | **PASS measured lanes** |
| RTX 3090 | RWKV-7 7.2B vs Qwen3.5-9B, prompt2048, bsz1/2 | finite logits, greedy equality and cosine `>=0.999995`; Qwen fast bindings verified | self-fused dense prefill `1.0519x–1.0846x`; decode `1.9258x–2.1441x` | **PASS measured cells** |
| RTX 3090 | g1h 7.2B vs Qwen3.5-9B, bsz8, dense/W8/W4 | finite logits, fail-closed Qwen FLA and route contracts; quality is a separate axis | dense prefill/decode min `1.0589x/1.7884x`; decode active work min `1.4379x`; W8/W4 total latency and memory gates pass | **PASS 18/18** |
| RTX 3090 | 1.5B/2B and 2.9B/4B, bsz8, dense/W8/W4 | finite logits, fail-closed native/Qwen FLA contracts; quality is a separate axis | dense prefill min `1.0306x/1.3559x`, decode min `3.3828x/2.9213x`; W8/W4 total latency and physical-memory gates pass | **PASS 36/36** |
| RTX 4090 | g1h 7.2B vs Qwen3.5-9B, bsz8, dense/W8/W4 | finite logits, fail-closed Qwen FLA routes, BNB8/MM4 same-quant probes; task quality is separate | dense prefill/decode min `1.0240x/2.2101x`; decode active work min `1.7770x`; W8/W4 total-latency and quant-local memory gates pass | **PASS 18/18** |
| RTX 4090 | 0.4B/0.8B, 1.5B/2B and 2.9B/4B, bsz8, dense/W8/W4 | finite logits, fail-closed native/full-FLA/route contracts; quality is a separate axis | dense prefill min `1.3704x/1.0420x/1.3051x`, decode min `12.1018x/5.6368x/4.2144x`; W8/W4 total latency and physical-memory gates pass | **PASS 54/54** |
| RTX 4090 | Historical 0.4B dense and W8/W4 speed lanes | 32-step greedy and cache handoff pass | decode `1.007x–1.418x` matching Albatross; bsz4 prefill `1.007x` current-session / `0.916x` historical high-water | **PASS measured lanes** |
| RTX 5090 | 0.4B MATH500; 1.5B/2.9B/7.2B quant; 13.3B inference | pass@64 `0.38`; compression ratio `1.0`; all quant same-next | MATH summary/decode `4.336x/4.871x` committed Albatross reference; 2.9B/7.2B quant `>=0.99x` paired fp16 | **PASS artifact** |
| RTX 5090 | 0.4B/0.8B through 7.2B/9B, B1/B8, dense/W8/W4 | 144/144 Qwen references verify full FLA plus Triton conv; 32/32 greedy checks pass; task quality is separate | raw dense prefill/decode minima `1.0226x/2.8130x`; per-active-B speed leads in all cells; W8/W4 total-latency and footprint gates pass | **PASS 8/8 batch-pairs** |
| RTX 5090 | g1h 1.5B/2.9B/7.2B/13.3B BF16 versus W4, B1/B8, prompt128/decode128 | prompt/final cosine `>=0.9995`, same-next 8/8; group-128 grid 280/280 | prefill/decode minima `1.0010x/1.1854x`; footprint `0.5298x–0.6250x` with automatic exact-model profiles | **PASS 8/8 all-phase cells** |
| RTX 5090 | official train_temp vs opt-in HF train_temp CUDA, 12x768 BF16, B1/T512 | backward 400/400 and FusedAdam step 800 tensors/deltas exactly match; 3-seed x 1,000-step cohort passes | median runtime 48.4061 s official vs 43.5184 s HF; candidate 10.10% lower in this synthetic cohort | **PASS exact lane** |
| RTX 5090 | official train_temp vs Native/no-FLA train_temp CUDA, 12x768 BF16, B16/T512, real MiniPile | 399/399 gradients and 399/399 parameter deltas exact; paired 3-seed x 1,000-step, continuous 5,000-step and 2,500+2,500 resume gates pass | paired-seed median `1.00049x`; 5,000-step Native `410.414s` vs official `411.462s`, or `1.00255x` | **PASS exact measured lane** |
| RTX 5090 | official RWKV-Gradio-3 v3a vs Native HF, g1h 7.2B FP16 weights + FP32 state, B1/B8 | extensions active; three 512-token repeats share one trace hash; min logits cosine `0.99999344`, top-1 `64/64` and `512/512` | Native median `145.06/845.57 tok/s` vs precision-matched v3a `144.47/841.77`, or `1.0041x/1.0045x` | **PASS precision-matched exact lane** |
| RTX 5090 | official RWKV-Gradio-3 v3a vs Native HF, FP16 weights/state/I/O | 7.2B B1/B8 16-step logits/state/xpa/xpf and greedy pass; 2.9B/13.3B B1/B8 prompt128/512/2048 prefill passes all tensor/state/greedy gates | 7.2B decode `1.0010x/1.0104x`; selected prefill 12/12 at `1.0029x–1.5690x` | **PASS exact fp16-state profiles** |
| Apple M5 | 0.4B/1.5B selected MLX vs Qwen3.5 pairs | state/session/greedy and speculative target oracle pass | selected conservative decode/prefill/TTFT/memory gates pass | **PASS measured pairs** |

## RTX 5090 official train_temp alignment

The opt-in HF backend vendors the exact Apache-2.0 CUDA operator sources from
RWKV-LM commit `e6f74b63a06e08606d130043599d218209628bad`. On one RTX 5090,
the 191,084,544-parameter 12x768 BF16 model at B1/T512 matches the official
reference exactly for backward loss and all 400 gradients. The DeepSpeed
FusedAdam step also matches parameter grouping/order, 401 gated tensors, 399
parameter deltas and post-step loss with cosine `1.0`, relative L2 `0` and max
absolute difference `0`.

Long CUDA training is nondeterministic in both the official and candidate
paths, including repeated runs with the same seed. The promoted long-run gate
therefore uses three complete 1,000-step runs per backend. Official and HF each
reach minimum validation loss `<=1.0` in `2/3` runs and `<=0.1` in `2/3` runs;
median train/validation AUC relative differences are `0.8531%/5.1030%`, and
the median maximum-gradient ratio is `1.3281x`. The cohort is `pass` under the
predeclared fail-closed thresholds. Evidence:
[`bench/5090_train_temp_alignment_20260717/`](bench/5090_train_temp_alignment_20260717/README.md).

The Native/no-FLA route now also has the official shell training shape rather
than a B1 substitute. At B16/T512 with gradient checkpointing, backward loss,
all 399 gradients, FusedAdam grouping/order, all 399 parameter deltas and
post-step loss match exactly. The final paired real-MiniPile cohort runs the
official backend immediately before Native for seeds 131/232/333. All six runs
are finite and reach final validation loss `<=4.8`; median train/validation AUC
relative differences are `0.0784%/0.1029%`, and Native median throughput is
`1.00049x` official. The p99.9 gradient ratio is `1.4745x`, below the `2x`
gate; isolated finite pre-clip maxima remain telemetry rather than replacing
the clipped-update, curve, and validation gates.

The continuous 5,000-step pair records Native `410.414s` versus official
`411.462s` (`1.00255x`), train/validation AUC differences of
`0.0953%/0.2200%`, and final validation losses `3.80373/3.81245`. A separate
Native run stops at 2,500, restores model/optimizer/Python/NumPy/torch
CPU/CUDA RNG digests, and finishes at 5,000 with `0.99822x` uninterrupted
Native throughput. This is a single RTX 5090, 12x768, BF16, B16/T512 result,
not a larger-model, multi-day, or distributed-training claim. Evidence:
[`bench/5090_native_train_temp_real_minipile_20260718/`](bench/5090_native_train_temp_real_minipile_20260718/README.md).

The unchanged official `demo-training-prepare.sh` and
`demo-training-run.sh` entry points also pass a bounded B16/T512/ZeRO-2 run on
the same card. The Native equivalent reports finite `399/399` ZeRO gradients
and a changed model hash. The same artifact records the original real Gradio
Native HF page and official shell validation. Its direct-decode numbers are
historical; the newer precision-matched fused close is recorded separately:
[`bench/5090_native_hf_gradio_train_temp_20260718/`](bench/5090_native_hf_gradio_train_temp_20260718/README.md).

## RTX 5090 Native cached-decode parity

Implementation commit `cf42c0e` adds a public Native fast-token path that can
borrow the CUDA-graph logits buffer for immediate consumption, FP32 sparse-FFN
accumulation, SM120 W/A/G low-rank grouping, and fail-closed extension checks.
The exact-card candidate keeps every new route opt-in.

On official g1h 7.2B with FP16 weights and FP32 recurrent state, three fresh
512-token process repeats produce B1/B8 medians of `145.06/845.57 tok/s`.
The official Space commit `cc57df4` in matching `fp32io16` mode measures
`144.47/841.77 tok/s`, so the ratios are `1.0041x/1.0045x`. Both requested
CUDA extensions are active in all six rows. All rows share the same complete
greedy-trace SHA256. The repository alignment script passes 64 teacher-forced
steps with minimum logits cosine `0.9999934435`, max absolute difference
`0.0625`, and exact top-1 at B1 and B8.

A fresh real-browser run through the current official Gradio page also exercises
the committed bridge rather than the standalone benchmark. Its stable 100-token
labels are Native/official `138.5/137.7 tok/s` at B1 and `831.8/837.7 tok/s`
at B8. A separate readable B1 prompt ends at 54 tokens with byte-identical
official and Native text. The post-B1/B8 process allocations are 15,520 MiB
Native and 14,886 MiB official, so this browser smoke proves output parity and
near speed parity, not memory parity or an all-shape lead. Screenshots, full
text, environment and logs are in
[`bench/5090_gradio_native_hf_frontend_ab_20260719/`](bench/5090_gradio_native_hf_frontend_ab_20260719/README.md).

The follow-up fp16-state profile closes that former gap. Three deterministic
512-token runs produce Native B1/B8 medians `146.42/899.51 tok/s` against the
pinned official `146.277/890.21`, or `1.00098x/1.01045x`. A separate 16-step
same-input comparison passes logits cosine `>=0.99999966`, state cosine
`>=0.99999919`, xpa/xpf gates, exact greedy `16/16` at B1 and `128/128` at B8.

Same-precision sequence-prefill evidence covers official g1h 2.9B and 13.3B,
B1/B8, prompt128/512/2048. All 12 selected cells pass prefill logits, layer
outputs, recurrent state, xpa/xpf, first-token and first-decode-token gates,
while Native throughput is `1.0029x–1.5690x` official. The measured profiles
are exact RTX 5090 defaults in `kernel_policy.py`; explicit environment
overrides still win, and unlisted cards/models/shapes remain conservative.
Evidence:
[`bench/5090_native_official_fp16_production_20260718/`](bench/5090_native_official_fp16_production_20260718/README.md).

## V100 production-close

Canonical matrix: 0.1B/0.4B/1.5B × bsz1/2/4/8.

| Lane | Range | Gate |
|---|---:|---|
| Dense decode / Albatross | `0.908x–1.248x` | P1 PASS |
| Prompt-512 prefill / Albatross | `0.930x–1.047x` | P1 PASS |
| Native W8/W4 payload / fp16 | `0.803x–0.956x` | footprint PASS |
| Native W8/W4 decode / fp16 | `1.006x–1.128x` | speed PASS |
| Native W8/W4 paired prefill / fp16 | `0.996x–1.007x` | 1% equivalence PASS |

Evidence: [`bench/v100_production_close_20260711/README.md`](bench/v100_production_close_20260711/README.md).

### V100 packed-MM4 BN/TN decode matrix

The exact-sm70 packed-W4 path adds card-local BN/TN tables and groupwise head
scales. Each promoted row is one load-time configuration across seven paired
fp16 cells: B1/B2/B4/B8 at prompt128/decode128, plus prompt512, prompt2048 and
decode512 at B1.

| Model | Config | Decode vs fp16 | Footprint | Min cosine | Complete gate |
|---|---|---:|---:|---:|---:|
| 1.5B | memory + group128 head + fused epilogue | `1.0255x-1.1837x` | `0.5395x` | `0.99828702` | `7/7` |
| 2.9B | speed + group256 head | `1.0111x-1.0346x` | `0.9573x` | `0.99965668` | `7/7` |
| 7.2B | memory + group128 head | `1.0810x-1.8422x` | `0.3013x` | `0.99903870` | `7/7` |

All 21 current-main cells match the complete fp16 greedy sequence and repeat
SHA256. The weakest 1.5B B4, 2.9B B8 and 7.2B B8 cells use five repeats. The
old unfused 1.5B B4 row (`0.9997x`), a 2.9B group128 B8 row (`0.9984x`) and
2.9B full-memory B4 (`0.9888x`) are retained as rejected boundaries; the gate
was not relaxed.

This is primarily a cached-decode result. Full-memory prefill is
`0.1276x-0.3192x` fp16 for 1.5B and `0.0716x-0.1516x` for 7.2B, so no universal
W4/prefill claim is made. The head-only 2.9B speed profile separately passes
paired prefill at `1.0006x-1.0603x`. Evidence:
[`bench/v100_sm70_mm4_bntn_20260716/README.md`](bench/v100_sm70_mm4_bntn_20260716/README.md).

### V100 B1/B8 active-parameter comparison against full-FLA Qwen3.5

The current optimized-reference artifact compares RWKV-7 1.5B with the
official Qwen3.5-2B checkpoint at prompt 512, decode 64 and batch 1/8. Qwen is
fail-closed on FLA chunk prefill, fused-recurrent decode, fused gated norm and
the repository Triton causal-convolution kernels; both rows report the full
operator contract and effective backend
`qwen_fla_gated_delta_rule_fla_triton_conv`.

RWKV/Qwen active parameters are 1,527,404,544/1,881,825,088 (`0.811661x`).
The explicit normalized gate uses `aggregate tok/s * active parameters`, so
RWKV needs at least `1.232041x` raw Qwen throughput to tie. Both phases pass in
both cells:

| Bsz | Prefill RWKV/Qwen | Prefill active work | Decode RWKV/Qwen | Decode active work | Peak VRAM RWKV/Qwen |
|---:|---:|---:|---:|---:|---:|
| 1 | `2.815921x` | `2.285574x` | `5.913307x` | `4.799514x` | `1.024885x` |
| 8 | `5.407762x` | `4.389270x` | `5.270432x` | `4.277804x` | `0.837248x` |

Qwen full-FLA/Triton-conv versus its oracle and RWKV native graph versus its
FLA-backed HF route each preserve all 32 greedy tokens and pass their cosine
gates. This is target-only inference with no draft/speculative path. The B1
peak-VRAM loss is retained explicitly; memory was not a gate for this dense
speed artifact.

Evidence: [`bench/v100_active_b1b8_20260715/README.md`](bench/v100_active_b1b8_20260715/README.md).

### Historical V100 RWKV-7 vs Qwen3.5 Torch-fallback matrix

The historical official text-only matrix covers three model pairs, fp16/bnb8/
bnb4, prompt 128/512/2048, decode 128/512, and bsz1/2/4/8: `432/432` raw rows
pass and all `216/216` comparison cells join.

| Metric | Minimum | Median | Maximum | Strict 1.05x pass |
|---|---:|---:|---:|---:|
| Prefill RWKV/Qwen | `1.246x` | `1.936x` | `8.141x` | 216/216 |
| Decode RWKV/Qwen | `0.947x` | `1.317x` | `10.832x` | 207/216 |

All nine strict-gate misses are bnb4 decode rows; only three are below `1.0x`.
Static model footprint is lower in `216/216` cells (`0.629x-0.812x`), and peak
allocated VRAM is lower in `192/216` cells (`0.390x-1.068x`).

Important boundary: all recorded Qwen rows use the official
Transformers/PyTorch fallback (`qwen3_5_text`, forced torch, FLA not
importable). This artifact is now a historical diagnostic and does not satisfy
the optimized-Qwen acceptance gate. The replacement matrix defaults to
`--qwen-backend fla` and fails closed unless every Qwen Gated DeltaNet layer
binds FLA prefill/decode, causal-convolution, and fused-normalization operators.
The completed RTX 5070 artifact is reported below. The current V100 B1/B8
artifact above additionally closes the optimized reference on `sm_70`,
including the Triton causal-convolution contract. Neither result retroactively
upgrades these historical Torch-fallback rows.

Evidence: [`bench/qwen35_v100_hf_matrix_20260712/README.md`](bench/qwen35_v100_hf_matrix_20260712/README.md).
Design: [`docs/plans/2026-07-13-qwen35-5070-fla-design.md`](docs/plans/2026-07-13-qwen35-5070-fla-design.md).

## RTX 3090 self-fused long-prefill rows

The vendored sequence-mode DPLR kernel is now the measured production route
for RWKV-7 7.2B long prefill on RTX 3090.  It computes the RWKV-specific DPLR
A/B terms in-register, directly consumes/emits the native recurrent-state
layout and removes standalone gate/residual work.  The effective HF backend is
`native_graph`, not FLA.

| Bsz | RWKV/Qwen prefill tok/s | Prefill ratio | RWKV/Qwen decode tok/s | Decode ratio | Result |
|---:|---:|---:|---:|---:|---|
| 1 | 4,536.404 / 4,182.369 | `1.0846x` | 49.922 / 23.283 | `2.1441x` | PASS |
| 2 | 4,579.237 / 4,353.260 | `1.0519x` | 89.369 / 46.406 | `1.9258x` | PASS |

Both rows use fp16, prompt 2048, decode 128 and three warmups/three measured
runs.  All 24 Qwen GatedDeltaNet layers are fail-closed verified to bind FLA
chunk/recurrent kernels and causal-conv1d; the recorded Qwen backend is
`fla+causal_conv1d`.  RWKV also has the lower model footprint and peak VRAM in
both rows.  This closes these two dense cells only; the full 3090 216-cell
dense/W8/W4 matrix remains open.

Evidence: [`bench/3090_self_fused_20260713/README.md`](bench/3090_self_fused_20260713/README.md).

### RTX 3090 native quant production batch

The 7.2B/9B broad matrix now also has a fail-closed `72/72` artifact with zero
red or missing cells at the historical dense `1.05x` floor. The current
acceptance policy is stricter and bsz8-only: dense RWKV is compared with dense
Qwen using pair-specific active-parameter-normalized targets, while RWKV W8/W4
is gated only against the same RWKV fp16 row. Quantized Qwen is not a quant
acceptance dependency.

At bsz8, dense decode is `>=1.8924x` Qwen and passes the new 7.2B/9B `1.50x`
target. Dense prefill is `>=1.0537x`, so the normalized prefill target remains
open. W8 and W4 are respectively `>=1.7970x/1.0900x` and
`>=1.0018x/1.0179x` their matching RWKV dense prefill/decode rows, with both
footprint and peak VRAM lower. External-token quality ratios are `1.001475`
(W8), `1.001564` (BnB W4) and `1.004745` (TorchAO W4), all within the `1.01`
gate.

Evidence and exact reproduction:
[`bench/3090_native_quant_20260713/README.md`](bench/3090_native_quant_20260713/README.md).

### RTX 3090 g1h 7.2B bsz8 acceptance

The latest g1h 7.2B checkpoint is now measured against official Qwen3.5-9B at
bsz8, prompt 128/512/2048, decode 128/512, and fp16/W8/W4. All 18 joined cells
pass with zero red or missing rows. The six dense Qwen rows verify all 24 FLA
Gated DeltaNet and causal-conv1d bindings.

| Family | RWKV/Qwen prefill min | RWKV/Qwen decode min | Quant/fp16 prefill min | Quant/fp16 decode min | Quant/fp16 total min | Footprint max | Peak max |
|---|---:|---:|---:|---:|---:|---:|---:|
| fp16 | `1.058907x` | `1.788418x` | — | — | — | — | — |
| W8 / BNB8 | `1.901037x` | `1.941481x` | `1.697094x` | `1.084305x` | `1.098658x` | `0.552555x` | `0.702992x` |
| W4 / native MM4 | `1.052379x` | `1.838668x` | `0.988822x` | `1.025666x` | `1.014527x` | `0.972049x` | `0.981434x` |

The dense RWKV/Qwen active-parameter ratio is `0.804032x`. Dense decode also
passes the explicit active-parameter work gate in 6/6 cells, with a minimum
`1.437946x` work rate. Prefill active work (`0.851395x–0.905145x`) remains
disclosed telemetry; direct prefill token throughput is the acceptance gate.
W4 is not faster in every prefill phase, so its exact-cell total-latency
fallback is stated explicitly rather than hidden. Total peak VRAM is lower in
18/18 cells, although runtime working set excluding model weights is larger.
This is an inference speed/memory result, not a task-quality claim.

Evidence and reproduction:
[`bench/3090_g1h_7p2_bsz8_20260714/README.md`](bench/3090_g1h_7p2_bsz8_20260714/README.md).

### RTX 3090 small-pair bsz8 acceptance

The 1.5B/2B and 2.9B/4B matrices cover prompt 128/512/2048, decode 128/512,
bsz8 and dense/W8/W4. Both pass `18/18`; all `36/36` Qwen reference rows
satisfy the fail-closed FLA Gated DeltaNet operator contract and there are no
red or missing cells.

| Pair | Dense prefill min vs Qwen | Dense decode min vs Qwen | W8 total min vs fp16 | W4 prefill/decode/total min vs fp16 |
|---|---:|---:|---:|---:|
| RWKV 1.5B / Qwen 2B | `1.0306x` | `3.3828x` | `1.1929x` | `0.9835x / 1.0279x / 1.0107x` |
| RWKV 2.9B / Qwen 4B | `1.3559x` | `2.9213x` | `1.1809x` | `0.9863x / 1.0198x / 1.0068x` |

W4 is not described as faster in every phase: its minimum prefill ratios are
shown above. The explicitly enabled non-inferiority gate uses the exact-cell
`prefill + decode` latency, while retaining phase telemetry. Every W4 cell has
lower total latency, model footprint and peak VRAM than its matching RWKV fp16
cell. This is a scoped speed/memory result, not a model-quality claim.

Evidence and reproduction:
[`bench/3090_small_bsz8_20260714/README.md`](bench/3090_small_bsz8_20260714/README.md).

## RTX 5070 Laptop RWKV-7 vs verified Qwen3.5 FLA

The promoted exact-card bsz8 matrix compares 1.5B RWKV with official 2B Qwen
across prompt128/512/2048, decode128/512, and fp16/bnb8/bnb4. All 36 raw rows
and 18 joined cells pass. Every Qwen performance row binds all 18 Gated
DeltaNet layers to FLA chunk prefill, FLA fused-recurrent decode, FLA fused
gated norm, and FLA Triton causal-convolution prefill/update. The effective
backend is `qwen_fla_gated_delta_rule_fla_triton_conv`; there is no Qwen Torch
fallback in the performance matrix.

| Metric | Minimum | Median | Maximum | Strict pass |
|---|---:|---:|---:|---:|
| Prefill RWKV/Qwen | `1.082707x` | `1.375135x` | `1.688725x` | 18/18 at >=1.05x |
| Decode RWKV/Qwen | `1.795119x` | `2.544989x` | `3.456505x` | 18/18 at >=1.05x |
| Model footprint RWKV/Qwen | `0.729146x` | `0.811662x` | `0.856635x` | 18/18 no larger |
| Peak VRAM RWKV/Qwen | `0.605574x` | `0.845321x` | `0.955585x` | 18/18 no larger |
| Prefill tok/s per active-B | `1.333940x` | `1.694224x` | `2.080579x` | 18/18 at >=1.0x |
| Decode tok/s per active-B | `2.211641x` | `3.135530x` | `4.258556x` | 18/18 at >=1.0x |

RWKV/Qwen active parameters are 1,527,404,544/1,881,825,088 (`0.811661x`).
The model-efficiency gate uses `tok/s / active-B`. Hardware logical work rate
(`tok/s * active parameters`) remains separate telemetry; its minimum prefill
ratio is `0.878791x` and is not used to penalize the smaller model. Runtime
working set is lower in 8/18 cells, while total peak VRAM, the fit constraint,
is lower in all 18.

Fp16 and BNB4 decode use native graph; BNB4 prefill also uses the opt-in
external-quant graph. BNB8 uses the conservative `decode_rk` policy. The Qwen
full-FLA numerical oracle passes 8/8 greedy at prompt/final cosine
`0.99999022`/`0.99999237`. RWKV bsz8 native-prefill probes pass 8/8 greedy and
the `0.9999` cosine gate for fp16, BNB8, and BNB4.

This supersedes the broader 72-cell 5070 artifact for the strict bsz8
optimized-Qwen claim. That older artifact remains useful bsz1/2/4 coverage,
but its Qwen convolution is a Transformers Torch fallback. The new result is
an exact-card performance and memory close, not a model-quality claim.

Final full-FLA evidence: [`bench/5070_qwen35_full_fla_bsz8_20260714/README.md`](bench/5070_qwen35_full_fla_bsz8_20260714/README.md).
Historical FLA-core-only evidence: [`bench/5070_qwen35_fla_native_prefill_20260714/README.md`](bench/5070_qwen35_fla_native_prefill_20260714/README.md).
Historical baseline: [`bench/5070_qwen35_fla_matrix_20260713/README.md`](bench/5070_qwen35_fla_matrix_20260713/README.md).

## RTX 4090 promoted rows

The latest g1h 7.2B checkpoint is measured against official Qwen3.5-9B at
bsz8, prompt 128/512/2048, decode 128/512, shared prefill chunk 512, and
fp16/W8/W4. All 18 joined cells pass with zero red or missing rows. The six
dense Qwen rows verify all 24 FLA Gated DeltaNet, fused-gated-norm and
causal-conv1d operator bindings.

| Family | RWKV/Qwen prefill min | RWKV/Qwen decode min | Quant/fp16 prefill min | Quant/fp16 decode min | Quant/fp16 total min | Footprint max | Peak max |
|---|---:|---:|---:|---:|---:|---:|---:|
| fp16 | `1.023951x` | `2.210065x` | — | — | — | — | — |
| W8 / BNB8+A8W8 head | `1.508672x` | `3.002438x` | `1.472988x` | `1.356914x` | `1.360072x` | `0.533926x` | `0.455834x` |
| W4 / MM4 or TorchAO | `1.000256x` | `2.260570x` | `0.976859x` | `1.022724x` | `1.013273x` | `0.972617x` | `0.983054x` |

Dense decode passes the explicit active-parameter work gate in 6/6 cells,
with a minimum `1.776961x` work rate. W4 is not faster in every prefill phase;
its exact-cell total-latency fallback is stated explicitly. The dense RWKV
footprint is `0.804034x` Qwen, but dense peak allocated VRAM is
`1.156353x–1.209017x` Qwen under shared chunk-512, so cross-model peak memory is
not claimed as a win. Both selected quant families lower footprint and peak
VRAM versus matching RWKV fp16 in every cell. This is a scoped inference
speed/memory result, not a model-quality claim.

Evidence and reproduction:
[`bench/4090_g1h_7p2_bsz8_20260715/README.md`](bench/4090_g1h_7p2_bsz8_20260715/README.md).

### RTX 4090 small-model bsz8 acceptance

The 0.4B/0.8B, 1.5B/2B and 2.9B/4B pairs use the same prompt
128/512/2048, decode 128/512, batch-8 and dense/W8/W4 contract as the latest
7.2B lane. All three pair gates pass: `54/54` joined cells, `54/54` verified
Qwen FLA references, zero red cells and zero missing rows.

| Pair | Dense prefill/decode min vs Qwen | Dense decode active work min | W8 total min vs fp16 | W4 prefill/decode/total min vs fp16 | W8 footprint/peak max | W4 footprint/peak max |
|---|---:|---:|---:|---:|---:|---:|
| RWKV 0.4B / Qwen 0.8B | `1.370369x / 12.101818x` | `7.250339x` | `1.011441x` | `0.999344x / 1.041423x / 1.029994x` | `0.925797x / 0.963266x` | `0.890672x / 0.945793x` |
| RWKV 1.5B / Qwen 2B | `1.041959x / 5.636846x` | `4.575207x` | `1.131672x` | `0.930925x / 1.038061x / 1.027211x` | `0.560704x / 0.625465x` | `0.935468x / 0.968566x` |
| RWKV 2.9B / Qwen 4B | `1.305103x / 4.214362x` | `2.953767x` | `1.176050x` | `0.986393x / 1.024407x / 1.014959x` | `0.544714x / 0.509156x` | `0.961227x / 0.977123x` |

The W4 prefill deficits are explicitly reported; every complete W4 cell is
faster than fp16 under the declared exact-cell total-latency gate and uses less
physical memory. The 1.5B prompt-512 close promotes an exact
hidden/batch/prompt scan tile (`2048x8x512 -> block_m=32`) without changing the
7.2B row-8 route. A no-environment-override probe verifies that policy, and the
focused suite passes 74 tests.

Evidence and reproduction:
[`bench/4090_small_bsz8_20260715/README.md`](bench/4090_small_bsz8_20260715/README.md).

### Historical RTX 4090 0.4B rows

0.4B dense fp16 native-graph decode:

| bsz | HF tok/s | HF / matching Albatross |
|---:|---:|---:|
| 1 | 795.7 | `1.007x` |
| 2 | 1,469.5 | `1.016x` |
| 4 | 2,585.7 | `1.008x` |
| 8 | 3,185.3 | `1.418x` |

Prompt512 fixed-shape prefill reaches `64.51k tok/s` at bsz1 and `107.87k
tok/s` at bsz4. The bsz4 row is `1.007x` the same-session Albatross rerun and
`0.916x` the retained historical `117.79k tok/s` reference.

Quant speed lanes:

- W8 payload `0.926x`, prefill `1.011x` fp16, decode bsz1/2/4/8
  `1.001x–1.020x`.
- W4 payload `0.891x`, prefill `1.010x` bf16, measured decode bsz1/4
  `1.043x/1.058x`.
- W4 memory policy can reach payload `0.399x`, but is not promoted as a universal
  fp16-or-faster lane.

## RTX 5090 production-close

Environment and full evidence:
[`bench/5090_blackwell_production_close_20260712/README.md`](bench/5090_blackwell_production_close_20260712/README.md).

### Native versus Albatross/v3a

The default Native/no-FLA path passes the pinned `rwkv7_fast_v3a` comparison
on official g1h checkpoints. With FP16 weights, recurrent state and I/O, g1h
7.2B cached decode reaches `146.42/899.51 tok/s` at B1/B8 versus
`146.28/890.21 tok/s`, or `1.0010x/1.0104x`. The paired tensor oracle passes
logits cosine `0.99999967/0.99999979`, exact top-1 `17/17` and `136/136`, exact
greedy `16/16` and `128/128`, recurrent state, xpa and xpf.

Sequence prefill covers g1h 2.9B and 13.3B at B1/B8 and prompt
128/512/2048. All 12 cells pass tensor, state, token and throughput gates at
`1.0029x–1.5690x` pinned v3a throughput. The real RWKV-Gradio-3 browser A/B
also records byte-identical Native/official output and B1/B8 page rates of
`138.5/831.8 tok/s` Native versus `137.7/837.7 tok/s` official.

Evidence:
[`bench/5090_native_official_fp16_production_20260718/`](bench/5090_native_official_fp16_production_20260718/README.md)
and
[`bench/5090_gradio_native_hf_frontend_ab_20260719/`](bench/5090_gradio_native_hf_frontend_ab_20260719/README.md).

### Full-FLA Qwen3.5 B1/B8 matrix

The current-main artifact at
[`bench/5090_g1h_qwen35_b1_b8_20260715/`](bench/5090_g1h_qwen35_b1_b8_20260715/README.md)
covers 0.4B/0.8B, 1.5B/2B, 2.9B/4B, and 7.2B/9B at B1 and B8, prompt
128/512/2048 and decode 128/512. The strict summary passes 8/8 batch-pairs,
144 candidate rows, 144 joined Qwen rows, and 144/144 full-FLA contracts. Raw
dense prefill/decode minima across pair minima are `1.0226x/2.8130x`; tokens/s
per active billion parameters also lead in every cell. W8 and W4 pass
paired-fp16 total-latency and footprint gates. For RWKV-7 7.2B versus
Qwen3.5-9B, B1/B8 minima are `1.1739x/1.0309x` prefill,
`2.8934x/2.8130x` decode and `2.3263x/2.2618x` active-work decode. Dense model
footprint is `13,731.3 MiB` versus `17,078.0 MiB`, or `0.8040x`.

### Quant pressure matrix

Shape: 1.5B/2.9B/7.2B × fp16/MM8/MM4 × prompt128/2048 × decode128/512 × bsz8.

| Model | Quant | Minimum speed ratio | Minimum footprint ratio | Same next |
|---|---|---:|---:|---:|
| 1.5B | MM8 | `0.9841x` | `0.9562x` | 4/4 |
| 1.5B | MM4 | `0.9932x` | `0.9342x` | 4/4 |
| 2.9B | MM8 | `0.9925x` | `0.9716x` | 4/4 |
| 2.9B | MM4 | `0.9967x` | `0.9573x` | 4/4 |
| 7.2B | MM8 | `0.9913x` | `0.9814x` | 4/4 |
| 7.2B | MM4 | `0.9919x` | `0.9720x` | 4/4 |

All 24 quant rows lower footprint and preserve the fp16 next token. The
2.9B/7.2B strict 1% gate and the combined 2% gate both pass.

### Official g1h 13.3B

The latest official `rwkv7-g1h-13.3b-20260710-ctx10240.pth` checkpoint is
26,540,868,485 bytes with SHA256
`5bd705d13497d23530e544d5afb45bdf542b5f67dffee31e3e2b35e4042cfcfb`.
The low-memory conversion produced six safetensors shards with 2,016 indexed
tensor keys. Load/forward/generate passes on RTX 5090 with a 25,309.1 MiB model
footprint and 25,448.3 MiB smoke peak. At B8, prompt128/decode128, selected
speed-policy MM8/MM4 measure `1.0013x/0.9845x` paired-fp16 decode speed and
`0.9899x/0.9848x` footprint, with cosine above `0.99985` and matching next
tokens. Full evidence:
[`bench/5090_g1h_13p3_20260715/`](bench/5090_g1h_13p3_20260715/README.md).

### Full MATH500

| Metric | HF adapter | Albatross reference | Result |
|---|---:|---:|---:|
| Tasks × rollout | `500 × 64` | `500 × 64` | compatible |
| pass@64 | `0.38` | `0.37` | PASS |
| Rollout accuracy | `0.142469` | `0.145937` | delta `-0.003469` |
| Summary token/s | `16,925.6` | `3,903.6` | `4.336x`, PASS |
| Steady decode token/s | `19,339.5` | `3,970.1` | `4.871x`, PASS |
| Compression bits/token | `1.9241015` | `1.9241015` | ratio `1.0`, PASS |

The RTX 5090 HF run passes every configured MATH500 and committed Albatross
reference gate.

## Apple M5 production-close

The promoted Apple artifact covers selected RWKV-7 0.4B vs Qwen3.5 0.8B and
RWKV-7 1.5B vs Qwen3.5 2B pairs using MLX groupwise W4, tiled DPLR prefill and
guarded compiled/speculative decode. Conservative speed, TTFT and memory gates
pass for the documented shapes, and speculative output passes the target-greedy
oracle. This is an M5 claim, not an all-Apple-family claim.

Evidence: [`docs/hardware/APPLE_PRODUCTION_CLOSE.md`](docs/hardware/APPLE_PRODUCTION_CLOSE.md).

## Training and compatibility evidence

| Platform | Current promoted compatibility evidence |
|---|---|
| V100 | Trainer/TRL/PEFT real-model rows; dual-card ZeRO base/resume |
| A100 40GB | 0.4B–7.2B Trainer/SFT/DPO/resume; dual-card ZeRO-2/3 base |
| A800 80GB | 0.1B–13.3B mixed inference/quant plus single/dual-card ZeRO |
| RTX A6000 48GB | 0.4B–7.2B training/resume; dual-card ZeRO through 2.9B |
| RTX 5090 | Exact 12x768 BF16 train_temp backward/FusedAdam step, paired real-MiniPile 3-seed cohort, continuous 5,000-step run and 2,500+2,500 recovery |
| Apple M5 | Tiny and real-model PEFT/Trainer/SFT/DPO/GRPO compatibility smoke |

See [`docs/TRAINING.md`](docs/TRAINING.md) and the validation documents.

## Reproduction and evidence rules

- Benchmark inventory: [`bench/INDEX.md`](bench/INDEX.md)
- Evidence format and promotion rules: [`bench/README.md`](bench/README.md)
- Acceptance mapping: [`docs/ACCEPTANCE.md`](docs/ACCEPTANCE.md)
- Hardware matrix: [`docs/HARDWARE_MATRIX.md`](docs/HARDWARE_MATRIX.md)
- Quantization: [`docs/QUANTIZATION.md`](docs/QUANTIZATION.md)
- Performance methodology: [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md)

Historical exploratory numbers remain available through Git history and dated
`bench/` artifacts; they are deliberately not mixed into this current summary.
