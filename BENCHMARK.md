# RWKV-7 HF adapter benchmark summary

This is the canonical **promoted-results summary**. It intentionally excludes
exploratory tuning chronology. Raw rows, logs and negative experiments remain
in [`bench/`](bench/); platform interpretation lives in
[`docs/PERFORMANCE.md`](docs/PERFORMANCE.md).

Last updated: **2026-07-15**.

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
| V100 32GB | 0.1B/0.4B/1.5B × bsz1/2/4/8 | greedy, cache handoff and focused regressions pass | dense decode `0.908x–1.248x`; prompt512 prefill `0.930x–1.047x` same-host Albatross | **PASS P1** |
| RTX 3090 | RWKV-7 7.2B vs Qwen3.5-9B, prompt2048, bsz1/2 | finite logits, greedy equality and cosine `>=0.999995`; Qwen fast bindings verified | self-fused dense prefill `1.0519x–1.0846x`; decode `1.9258x–2.1441x` | **PASS measured cells** |
| RTX 3090 | g1h 7.2B vs Qwen3.5-9B, bsz8, dense/W8/W4 | finite logits, fail-closed Qwen FLA and route contracts; quality is a separate axis | dense prefill/decode min `1.0589x/1.7884x`; decode active work min `1.4379x`; W8/W4 total latency and memory gates pass | **PASS 18/18** |
| RTX 3090 | 1.5B/2B and 2.9B/4B, bsz8, dense/W8/W4 | finite logits, fail-closed native/Qwen FLA contracts; quality is a separate axis | dense prefill min `1.0306x/1.3559x`, decode min `3.3828x/2.9213x`; W8/W4 total latency and physical-memory gates pass | **PASS 36/36** |
| RTX 4090 | g1h 7.2B vs Qwen3.5-9B, bsz8, dense/W8/W4 | finite logits, fail-closed Qwen FLA routes, BNB8/MM4 same-quant probes; task quality is separate | dense prefill/decode min `1.0240x/2.2101x`; decode active work min `1.7770x`; W8/W4 total-latency and quant-local memory gates pass | **PASS 18/18** |
| RTX 4090 | 0.4B/0.8B, 1.5B/2B and 2.9B/4B, bsz8, dense/W8/W4 | finite logits, fail-closed native/full-FLA/route contracts; quality is a separate axis | dense prefill min `1.3704x/1.0420x/1.3051x`, decode min `12.1018x/5.6368x/4.2144x`; W8/W4 total latency and physical-memory gates pass | **PASS 54/54** |
| RTX 4090 | Historical 0.4B dense and W8/W4 speed lanes | 32-step greedy and cache handoff pass | decode `1.007x–1.418x` matching Albatross; bsz4 prefill `1.007x` current-session / `0.916x` historical high-water | **PASS measured lanes** |
| RTX 5090 | 0.4B MATH500; 1.5B/2.9B/7.2B quant; 13.3B inference | pass@64 `0.38`; compression ratio `1.0`; all quant same-next | MATH summary/decode `4.336x/4.871x` committed Albatross reference; 2.9B/7.2B quant `>=0.99x` paired fp16 | **PASS artifact** |
| Apple M5 | 0.4B/1.5B selected MLX vs Qwen3.5 pairs | state/session/greedy and speculative target oracle pass | selected conservative decode/prefill/TTFT/memory gates pass | **PASS measured pairs** |

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

### V100 RWKV-7 vs Qwen3.5 HF matrix

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
The replacement exact-card run targets the RTX 5070 Laptop. Its required FLA
core contract covers Gated DeltaNet prefill/decode and fused gated norm;
`causal-conv1d` is reported separately on Windows. The completed 5070 artifact
is reported below; it does not retroactively upgrade the historical V100 rows.

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
2.9B/7.2B strict 1% gate passes; the combined matrix passes a 2% gate. The one
1.5B W8 `0.9841x` row prevents a universal strict `>=0.99x` claim.

### Official 13.3B

The official 26,540,868,485-byte checkpoint was converted on a 48GB/no-swap
host with the mmap/meta-template low-memory path into six safetensors shards.
Load/forward/generate on RTX 5090 uses 25,309.1 MiB model footprint and 25,536.6
MiB peak. Speed-policy boundaries are W8 `0.9912x` and W4 `0.9889x`; these are
selected-module speed-policy rows, not full-memory quant claims.

### Full MATH500

| Metric | HF adapter | Albatross reference | Result |
|---|---:|---:|---:|
| Tasks × rollout | `500 × 64` | `500 × 64` | compatible |
| pass@64 | `0.38` | `0.37` | PASS |
| Rollout accuracy | `0.142469` | `0.145937` | delta `-0.003469` |
| Summary token/s | `16,925.6` | `3,903.6` | `4.336x`, PASS |
| Steady decode token/s | `19,339.5` | `3,970.1` | `4.871x`, PASS |
| Compression bits/token | `1.9241015` | `1.9241015` | ratio `1.0`, PASS |

The HF run is live RTX 5090 evidence. The Albatross side is the committed
full-run reference, not a fresh same-card/same-session rerun.

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
| Apple M5 | Tiny and real-model PEFT/Trainer/SFT/DPO/GRPO compatibility smoke |

See [`docs/TRAINING.md`](docs/TRAINING.md) and the validation documents.

## Known gaps

1. Full-memory W8/W4 is not yet universally fp16-or-faster.
2. H100, AMD/ROCm and Turing lack promoted matrices.
3. Albatross P2/P3 is not closed for every model/card/batch.
4. RTX 5090 final comparison needs a fresh same-card Albatross rerun.
5. Apple needs cross-M-series, CoreML INT4/ANE and broader Qwen quality evidence.
6. Longer production training and larger ZeRO-3 resume matrices remain open.

## Reproduction and evidence rules

- Benchmark inventory: [`bench/INDEX.md`](bench/INDEX.md)
- Evidence format and promotion rules: [`bench/README.md`](bench/README.md)
- Acceptance mapping: [`docs/ACCEPTANCE.md`](docs/ACCEPTANCE.md)
- Hardware matrix: [`docs/HARDWARE_MATRIX.md`](docs/HARDWARE_MATRIX.md)
- Quantization: [`docs/QUANTIZATION.md`](docs/QUANTIZATION.md)
- Performance methodology: [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md)

Historical exploratory numbers remain available through Git history and dated
`bench/` artifacts; they are deliberately not mixed into this current summary.
