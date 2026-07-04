# RTX 5090 Blackwell native-prefill validation (2026-07-04)

> Superseded note (2026-07-04): the broader RTX 5090 matrix, including native/no-FLA Trainer smoke, now lives in [`../5090_blackwell_hf_matrix_20260704`](../5090_blackwell_hf_matrix_20260704/README.md). This directory remains the first native-prefill fix artifact.

This artifact upgrades the earlier 5090 smoke from forward-prefill-only to the same HF-native route used by the 4090/Ada validation style: standard HF load/generate, HF API contract, native prefill, native-graph decode, dynamic batching, and W8/W4 quantized load smoke.

## Environment

- GPU: NVIDIA GeForce RTX 5090, 32607 MiB, driver 610.43.02
- PyTorch: `2.6.0a0+ecf3bae40a.nv25.01`, CUDA 12.8
- Triton: 3.3.1
- FLA: 0.5.1
- Transformers: 5.13.0
- bitsandbytes: 0.49.2
- Model: `/workspace/models/rwkv7-g1d-0.1b-hf`

## Code fixes validated

1. `native_jit.prefill` now accepts the current 41-field native pack including the stacked R/K/V projection tensor (`RKVw`). This fixes the previous 5090 native prefill failure: `ValueError: too many values to unpack (expected 40)`.
2. `triton_compat.py` is shipped with converted model dirs. It restores the legacy Triton `AttrsDescriptor` import path and applies a Blackwell torch.compile fallback for the early Torch 2.6 + Triton 3.3 stack.
3. `scripts/sync_hf_adapter_code.py` and `scripts/convert_rwkv7_to_hf.py` now copy `triton_compat.py` into remote-code model directories.

## Reusable validation runner

The checked-in runner [`../run_5090_hf_validation.sh`](../run_5090_hf_validation.sh) reproduces this smoke matrix. A shortened script self-test was run on the same 5090 with `PROMPT_TOKENS=16 DECODE_TOKENS=8 MATH_ROLLOUT=1 MATH_LIMIT=1 MATH_MAX_NEW_TOKENS=8 MATH_BSZ=1 BATCH_SIZES=1` and completed successfully.

## Validation commands / logs

| check | log | result |
|---|---|---|
| HF generate smoke | `validation_pass.log` | PASS, `generate_fast_token_backend native_graph` |
| HF API contract | `validation_pass.log` | PASS, beam generate backend `native_graph` |
| Native prefill forward | `validation_pass.log` | PASS, `generate_match=True`, `seen=32` |
| W8 quantized inference | `validation_quant.log` | PASS, footprint 283.4 MB, peak VRAM 554.1 MB |
| W4 quantized inference | `validation_quant.log` | PASS, footprint 242.9 MB, peak VRAM 517.5 MB |
| MATH-style dynamic batching smoke | `run.log`, `summary.json` | PASS |
| bsz sweep | `results_5090.jsonl`, `results_5090.report.json` | PASS telemetry |

## MATH-style native-prefill smoke

Dataset: two synthetic MATH-style rows, rollout 4, bsz 8, max_new_tokens 64. This is a runtime smoke, **not** a MATH500 avg@64 acceptance run.

| metric | value |
|---|---:|
| dynamic bsz | 8 |
| decoded token events | 512 |
| native prefill sec | 0.646 |
| decode sec | 1.085 |
| decode tok/s | 472.0 |
| generation token/s | 295.8 |
| backend | native_graph |

## Batch sweep (0.1B fp16)

| bsz | native_graph decode tok/s | ms/step | peak VRAM MB |
|---:|---:|---:|---:|
| 1 | 945.7 | 1.06 | 631.1 |
| 2 | 1345.6 | 1.49 | 465.0 |
| 4 | 2715.2 | 1.47 | 519.8 |
| 8 | 5338.2 | 1.50 | 623.0 |

## Scope note

This closes the 5090 compatibility blockers found in the first smoke and aligns the 50-series path with the 4090-style HF adapter smoke matrix. It does **not** claim full MATH500 avg@64 parity on 5090; that requires the real MATH500 dataset and the exact acceptance model, then `scripts/run_math500_acceptance.sh` on this hardware.
