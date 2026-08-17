# RWKV-7 Hugging Face adapter status

Last audited: **2026-08-17**.

## Release status

| Scope | Status |
|---|---|
| HF v0.7 adapter deliverable | **COMPLETE** |
| Native inference route | **CURRENT PERFORMANCE BACKEND** |
| FLA wrapper/reference | **COMPATIBILITY AND ORACLE ONLY** |
| Recurrent cache and serving helpers | **PASS** |
| PEFT, Trainer and TRL workflows | **PASS** |
| Dense HF inference PP/TP boundary | **PASS for declared scope** |
| Native W8/W4 | **PASS for recorded exact-card lines** |

Completion is reported by named scope; there is no official repository-wide
completion percentage.

## Current performance evidence

- Consolidated Qwen3.5 Prefill/Decode:
  [`docs/QWEN35_LATEST_P_D_TOKPS.md`](docs/QWEN35_LATEST_P_D_TOKPS.md).
- Current artifact manifest:
  [`bench/CURRENT_ARTIFACTS.json`](bench/CURRENT_ARTIFACTS.json).
- V100, RTX 3090, RTX 4080 and RTX 4090 paired Prefill/Decode matrices pass all
  retained raw and parameter-adjusted cells.
- RTX 5090 paired Decode passes 48/48; the retained RWKV candidate and frozen
  Qwen reference also recompute to 48/48 Prefill wins.
- RTX 4080 7.2B/B8 FP16-state Decode is **344.39 tok/s** with greedy
  **12,288/12,288**.

Exact claims and scope boundaries are in [`BENCHMARK.md`](BENCHMARK.md) and
[`docs/RESULTS_INDEX.md`](docs/RESULTS_INDEX.md).

## Compatibility

The adapter retains standard Auto classes, `generate(use_cache=True)`,
`RWKV7StateCache`, dynamic batch select/reorder/drop/compact, chunked prefill,
save/reload, PEFT, Trainer, TRL SFT/DPO/GRPO, ZeRO smoke, quantized loading, and
speculative decoding helpers.

Native is the production performance route. FLA is still tested as an explicit
reference/compatibility backend and is used by current cross-backend
correctness probes; removing historical FLA speed artifacts does not remove
that compatibility surface.

## Review entry points

- Acceptance: [`docs/ACCEPTANCE.md`](docs/ACCEPTANCE.md)
- Hardware: [`docs/HARDWARE_MATRIX.md`](docs/HARDWARE_MATRIX.md)
- Training: [`docs/TRAINING.md`](docs/TRAINING.md)
- Quantization: [`docs/QUANTIZATION.md`](docs/QUANTIZATION.md)
- Current benchmark inventory: [`bench/INDEX.md`](bench/INDEX.md)
