# RWKV-7 HF adapter benchmark summary

This is the canonical **promoted-results summary**. It intentionally excludes
exploratory tuning chronology. Raw rows, logs and negative experiments remain
in [`bench/`](bench/); platform interpretation lives in
[`docs/PERFORMANCE.md`](docs/PERFORMANCE.md).

Last updated: **2026-07-13**.

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
| RTX 4090 | 0.4B dense and W8/W4 speed lanes | 32-step greedy and cache handoff pass | decode `1.007x–1.418x` matching Albatross; bsz4 prefill `1.007x` current-session / `0.916x` historical high-water | **PASS measured lanes** |
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

### V100 full-memory fused quant FFN probe

The 1.5B/bsz1/prompt128/decode128 paired-baseline probe validates the
default-off fused MM8/MM4 FFN key projection plus ReLU-square epilogue.

| Quant | Fused FFN | Decode / fp16 | Footprint / fp16 | Same next |
|---|---|---:|---:|---|
| MM8 | off | `0.4110x` | 0.6932 | yes |
| MM8 | on | `0.4145x` | 0.6932 | yes |
| MM4 | off | `1.1462x` | 0.5389 | yes |
| MM4 | on | `1.1867x` | 0.5389 | yes |

The isolated `2048 -> 8192` FFN epilogue rows pass for bsz1/2/4/8: MM8
speedup is `1.0075x-1.0989x` and MM4 is `1.1541x-1.7242x`, with minimum
cosine `0.99999988`. MM4 closes the speed/footprint gate for this exact
end-to-end shape, but MM8 remains slower than fp16. The flag therefore stays
opt-in pending broader model/batch/card evidence.

Evidence: [`bench/v100_native_fused_quant_ffn_20260712/README.md`](bench/v100_native_fused_quant_ffn_20260712/README.md).

The expanded V100 full-memory matrix completes `126/126` rows across 1.5B,
2.9B, and 7.2B. MM4 off/up beat fp16 in every `21/21` model/cell pair; fused
MM4 ranges are `1.0553x-1.1951x`, `1.0415x-1.2564x`, and
`1.2238x-1.9110x`, with footprint ratios `0.5389x/0.5306x/0.3010x`.
Strict acceptance remains open because greedy is only 6/7, 6/7, and 4/7.
MM8 passes zero of 21 speed cells for every off/up/deep lane and reaches only
`0.1123x-0.4394x` fp16. Execution completeness must not be reported as quant
acceptance.

Evidence: [`bench/v100_native_quant_full_matrix_20260713/README.md`](bench/v100_native_quant_full_matrix_20260713/README.md).

### RTX 5070 Laptop full-memory fused quant FFN matrix

The 1.5B expanded matrix covers seven batch/context/decode cells and six paths
per cell: fp16, MM8 off/up/deep, and MM4 off/up. All `42/42` rows pass and all
quantized rows preserve the fp16 greedy token.

| Path | Median decode / fp16 | Range | Footprint / fp16 | Greedy |
|---|---:|---:|---:|---:|
| MM8 off | `0.9551x` | `0.9413x-1.0820x` | `0.6932x` | 7/7 |
| MM8 up | `0.9620x` | `0.9472x-1.0852x` | `0.6932x` | 7/7 |
| MM8 deep | `0.9671x` | `0.9471x-1.0893x` | `0.6932x` | 7/7 |
| MM4 off | `0.8171x` | `0.8025x-0.9868x` | `0.5394x` | 7/7 |
| MM4 up | `0.8171x` | `0.7870x-0.9911x` | `0.5394x` | 7/7 |

MM8 deep is a small Blackwell-local improvement over up-only at the median,
but it regresses one paired cell and remains below fp16 at the median. MM4 is
a memory path on this card. Both fusions remain opt-in.

Evidence: [`bench/5070_native_fused_quant_ffn_20260713/README.md`](bench/5070_native_fused_quant_ffn_20260713/README.md).

The exact-card follow-up changes RTX 5070 MM8 decode tiles from `128x128` to
`64x256`. With the opt-in deep FFN epilogues, all seven 1.5B cells now beat
same-process fp16: minimum/median/maximum `1.0765x/1.1036x/1.1548x`, footprint
`0.6932x`, minimum final cosine `0.9999553`, and greedy 7/7. The MM4 follow-up
below closes the matching exact-card matrix separately.

Evidence: [`bench/5070_native_mm8_tuned_deep_20260713/README.md`](bench/5070_native_mm8_tuned_deep_20260713/README.md).

The matching MM4 follow-up fuses the FFN-down residual epilogue and adds an
exact-card, output-aware tensor-core dot route from bsz2. All seven 1.5B cells
beat same-process fp16: minimum/median/maximum `1.0580x/1.1360x/1.2525x`,
footprint `0.5394x`, minimum final cosine `0.99809039`, and greedy 7/7. The
measured 5070 tiles are selected automatically, while both FFN fusion flags
remain opt-in and other cards/models remain open gates.

Evidence: [`bench/5070_native_mm4_tuned_deep_20260713/README.md`](bench/5070_native_mm4_tuned_deep_20260713/README.md).

### RTX 5070 Laptop 2.9B strict matrix and 7.2B feasibility

The 2.9B expanded matrix completes `42/42` fresh-process rows. All three MM8
lanes independently pass all seven exact-shape speed, footprint, and greedy
gates. MM4 is faster and smaller in every cell but fails every greedy check.

| Path | Speed >= fp16 | Decode / fp16 range | Footprint / fp16 | Greedy | Accepted |
|---|---:|---:|---:|---:|---:|
| MM8 off | 7/7 | `1.0870x-1.1887x` | `0.6876x` | 7/7 | yes |
| MM8 up | 7/7 | `1.0567x-1.1906x` | `0.6876x` | 7/7 | yes |
| MM8 deep | 7/7 | `1.1019x-1.1918x` | `0.6876x` | 7/7 | yes |
| MM4 off | 7/7 | `1.1012x-1.3737x` | `0.5310x` | 0/7 | no |
| MM4 up | 7/7 | `1.1518x-1.3834x` | `0.5310x` | 0/7 | no |

Dense 7.2B fp16 has a `13731.3 MiB` model footprint and cannot fit in the
card's `8151 MiB`. CPU-first quantization makes bsz1 prompt128/decode128
feasible: MM4 up records `4140.5 MiB` model, `4769.9 MiB` peak, and `40.1
tok/s`; MM8 deep records `7340.5 MiB`, `7700.4 MiB`, and `32.7 tok/s`. These
rows have no same-card fp16 timing or logits baseline, so they are footprint and
execution evidence only. Their token `31261` matches the exact-shape V100 fp16
reference, which is corroboration rather than a same-card acceptance gate.

Evidence: [`bench/5070_native_quant_large_models_20260713/README.md`](bench/5070_native_quant_large_models_20260713/README.md).

### V100 RWKV-7 vs Qwen3.5 HF matrix

The complete official text-only matrix covers three model pairs, fp16/bnb8/
bnb4, prompt 128/512/2048, decode 128/512, and bsz1/2/4/8: `432/432` raw rows
pass and all `216/216` comparison cells join.

| Metric | Minimum | Median | Maximum | Strict 1.05x pass |
|---|---:|---:|---:|---:|
| Prefill RWKV/Qwen | `1.246x` | `1.936x` | `8.141x` | 216/216 |
| Decode RWKV/Qwen | `0.947x` | `1.317x` | `10.832x` | 207/216 |

All nine strict-gate misses are bnb4 decode rows; only three are below `1.0x`.
Static model footprint is lower in `216/216` cells (`0.629x-0.812x`), and peak
allocated VRAM is lower in `192/216` cells (`0.390x-1.068x`).

Important boundary: all Qwen rows use the official Transformers/PyTorch
fallback (`qwen3_5_text`, forced torch, FLA not importable), not an optimized
Qwen FLA/Triton backend. This is an exact V100 HF engine-speed result, not proof
that RWKV beats Qwen3.5 on newer hardware or model quality.

Evidence: [`bench/qwen35_v100_hf_matrix_20260712/README.md`](bench/qwen35_v100_hf_matrix_20260712/README.md).

## RTX 4090 promoted rows

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
