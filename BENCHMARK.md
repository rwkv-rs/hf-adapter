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
| RTX 3090 | RWKV-7 7.2B vs Qwen3.5-9B, prompt2048, bsz1/2 | finite logits, greedy equality and cosine `>=0.999995` | self-fused dense prefill `1.0527x–1.1029x`; decode `1.9716x–2.1100x` | **PASS measured cells** |
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

The larger-model HF comparison matrix additionally covers RWKV-7
1.5B/2.9B/7.2B against Qwen3.5 2B/4B/9B over prompt 128/512/2048, decode
128/512, bsz 1/2/4/8 and dense/W8/W4 loads. Its fail-closed result is
`216/216` cells with no red or missing rows. Overall RWKV/Qwen ratios are
`1.246x` minimum prefill and `1.003x` minimum decode. This result uses the
explicitly recorded Qwen Transformers torch-fallback backend rather than an
FLA/causal-conv optimized Qwen lane.

Evidence: [`bench/v100_qwen35_full_matrix_20260713/README.md`](bench/v100_qwen35_full_matrix_20260713/README.md).

## RTX 3090 self-fused long-prefill rows

The vendored sequence-mode DPLR kernel is now the measured production route
for RWKV-7 7.2B long prefill on RTX 3090.  It computes the RWKV-specific DPLR
A/B terms in-register, directly consumes/emits the native recurrent-state
layout and removes standalone gate/residual work.  The effective HF backend is
`native_graph`, not FLA.

| Bsz | RWKV/Qwen prefill tok/s | Prefill ratio | RWKV/Qwen decode tok/s | Decode ratio | Result |
|---:|---:|---:|---:|---:|---|
| 1 | 4,536.404 / 4,113.174 | `1.1029x` | 49.922 / 23.660 | `2.1100x` | PASS |
| 2 | 4,579.237 / 4,349.864 | `1.0527x` | 89.369 / 45.329 | `1.9716x` | PASS |

Both rows use fp16, prompt 2048, decode 128 and three warmups/three measured
runs.  RWKV also has the lower model footprint and peak VRAM in both rows.
This closes these two dense cells only; the full 3090 216-cell dense/W8/W4
matrix remains open.

Evidence: [`bench/3090_self_fused_20260713/README.md`](bench/3090_self_fused_20260713/README.md).

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
