# RWKV-7 Hugging Face adapter status

Last audited: **2026-08-20**.

## Release status

| Scope | Status |
|---|---|
| HF v0.8 adapter deliverable | **COMPLETE** |
| Native inference route | **CURRENT PERFORMANCE BACKEND** |
| FLA wrapper/reference | **COMPATIBILITY AND ORACLE ONLY** |
| Recurrent cache and serving helpers | **PASS** |
| PEFT, Trainer and TRL workflows | **PASS** |
| Dense HF inference PP/TP boundary | **PASS for declared scope** |
| Native W8/W4 | **PASS for recorded exact-card lines** |
| Public FP16 model family (0.1B through 13.3B) | **PUBLISHED AND VERIFIED** |
| Prebuilt CUDA kernels | **VERIFIED: CPython 3.11 / CUDA 12.4 / Torch 2.5 SM70 and Torch 2.6 SM89** |

Completion is reported by named scope; there is no official repository-wide
completion percentage.

## Public model distribution

The six ready-to-load model repositories are grouped in the
[`RWKV7-G1 Transformers` Collection](https://huggingface.co/collections/wangyue114514/rwkv7-g1-transformers-6a85b04191034d4c2d1896f1).
They were published with `rwkv7-hf==0.7.0` manifests and remain compatible
with the current `rwkv7-hf==0.8.0` runtime. They use FP16 Safetensors, pinned
source revisions, and repository-local conversion manifests. The complete
matrix, memory guidance, direct loading example, and public verification command are maintained in
[`docs/PUBLISHED_MODELS.md`](docs/PUBLISHED_MODELS.md).

Linux NVIDIA users can avoid first-run compilation on the two published exact
lanes by installing the matching `rwkv7-kernels` wheel with
`rwkv7-hf-kernels install`. Compatibility is checked before any binary module
is imported; unsupported environments retain the existing JIT and portable
fallbacks. See [`docs/KERNEL_WHEELS.md`](docs/KERNEL_WHEELS.md).

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
