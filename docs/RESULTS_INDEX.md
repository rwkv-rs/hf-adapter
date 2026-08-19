# Current results and evidence

Last updated: **2026-08-19**.

This is the compact reviewer index. The latest cross-card throughput table is
[`QWEN35_LATEST_P_D_TOKPS.md`](QWEN35_LATEST_P_D_TOKPS.md); exact raw evidence
is indexed by [`../bench/CURRENT_ARTIFACTS.json`](../bench/CURRENT_ARTIFACTS.json).

## Latest paired Qwen3.5 speed evidence

| GPU | Scope | Result | Evidence |
|---|---|---|---|
| Tesla V100 32GB | 4 model pairs, B1/B8, P128/512/2048, D128/512 | Raw and parameter-adjusted Prefill/Decode 48/48 | [`v100_qwen35_paired_pd_v1_20260814`](../bench/v100_qwen35_paired_pd_v1_20260814/README.md) |
| RTX 3090 | 4 model pairs, B1/B8, P128/512/2048, D128/512 | Raw and parameter-adjusted Prefill/Decode 48/48 | [`3090_qwen35_paired_pd_v2_20260816`](../bench/3090_qwen35_paired_pd_v2_20260816/README.md) |
| RTX 4080 | 3 model pairs, B1/B8, P128/512/2048, D128/512 | Raw and parameter-adjusted Prefill/Decode 36/36 | [`4080_qwen35_paired_pd_v1_20260814`](../bench/4080_qwen35_paired_pd_v1_20260814/README.md) |
| RTX 4090 | 4 model pairs, B1/B8, P128/512/2048, D128/512 | Raw and parameter-adjusted Prefill/Decode 48/48 | [`4090_qwen35_paired_pd_v2_20260815`](../bench/4090_qwen35_paired_pd_v2_20260815/README.md) |
| RTX 5090 | 4 model pairs, B1/B8, P128/512/2048, D128/512 | Raw and parameter-adjusted Decode 48/48 | [`5090_qwen35_paired_decode_v1_20260813`](../bench/5090_qwen35_paired_decode_v1_20260813/README.md) |

The RTX 5090 Qwen-only frozen source is
[`5090_qwen35_best_optimized_hf_v1_20260813`](../bench/5090_qwen35_best_optimized_hf_v1_20260813/README.md).

## Current exact-card and feature evidence

| Platform | Current retained lines |
|---|---|
| Tesla V100 | Dense/Albatross, paired P/D, FP16 state, MM4 Decode, W4 Prefill, Transformers TP |
| RTX 3090 | Paired P/D, Native quantization |
| RTX 4080 | Paired P/D, B8 grouped projection, 7.2B FP16 state |
| RTX 4090 | Paired P/D, Native route tuning, small/large B8 quantization |
| RTX 5070 Laptop | Native exact-card performance, Native quant loading |
| RTX 5090 | Paired Decode, Qwen reference, W4 BN/TN, Native/official alignment, official-math/B16/real-MiniPile training |
| Tesla T4 | Native compatibility, cache, quantization, training |
| AMD gfx1100 | Native exact-card close and quantization |
| Apple M5 | Current B1/B8 paired evidence and production-close standalone rows |
| Moore Threads S70 | Native compatibility and shift-mix kernel evidence |

The full path list and line identifiers are in
[`../bench/CURRENT_ARTIFACTS.json`](../bench/CURRENT_ARTIFACTS.json).

## Public Hugging Face distribution

The public `rwkv7-hf==0.7.0` release has six tagged model repositories from
0.1B through 13.3B. The retained acceptance bundle records a clean PyPI install
and full 0.1B load/forward/generation gate, plus the six-model metadata and LFS
integrity matrix:
[`hf_public_release_20260819`](../bench/hf_public_release_20260819/README.md).

Model links, storage layouts, and copyable verification commands are listed in
[`PUBLISHED_MODELS.md`](PUBLISHED_MODELS.md).

## Compatibility and training

- HF APIs and state cache: [`INFERENCE_WORKFLOWS.md`](INFERENCE_WORKFLOWS.md)
- PEFT, Trainer, TRL and ZeRO: [`TRAINING.md`](TRAINING.md)
- Quantization: [`QUANTIZATION.md`](QUANTIZATION.md)
- Hardware support: [`HARDWARE_MATRIX.md`](HARDWARE_MATRIX.md)
- MATH500 final protocol: [`validation/math500_acceptance.md`](validation/math500_acceptance.md)

RWKV FLA performance history has been removed after Native supersession. FLA
compatibility code and the current cross-backend correctness oracle remain.
