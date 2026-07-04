# RTX 5090 Blackwell HF validation matrix (2026-07-04)

This is the full one-command RTX 5090 HF smoke matrix produced by [`../run_5090_hf_validation.sh`](../run_5090_hf_validation.sh). It extends the earlier native-prefill smoke with the Blackwell-required native/no-FLA Trainer path, chunked-prefill rows, and exact-card fused-kernel A/B rows.

## Environment

- GPU: NVIDIA GeForce RTX 5090 (`sm_120`), driver 610.43.02, 32607 MiB
- PyTorch: `2.6.0a0+ecf3bae40a.nv25.01`, CUDA 12.8
- Triton: 3.3.1
- FLA: 0.5.1
- Transformers: 5.13.0
- bitsandbytes: 0.49.2
- Model: `/workspace/models/rwkv7-g1d-0.1b-hf`
- Runtime flags: `TORCH_COMPILE_DISABLE=1`, `TORCH_CUDA_ARCH_LIST=12.0`, `RWKV_V7_ON=1`

## Checks

| check | log | result |
|---|---|---|
| HF generate smoke | `smoke_hf_generate.log` | PASS, `generate_fast_token_backend native_graph` |
| HF API contract | `hf_api_contract.log` | PASS, beam backend `native_graph` |
| Native prefill forward | `fast_prefill_forward.log` | PASS, `generate_match=True`, `seen=32` |
| Native no-FLA Trainer + PEFT LoRA | `native_trainer_smoke.log` | PASS, loss history: [9.3638, 3.9694, 1.0814, 0.5251, 0.5898, 0.4192]; trainable params `72/72` updated |
| W8 quantized inference | `quant_8bit.log` | PASS, footprint 283.4 MB, peak VRAM 554.1 MB |
| W4 quantized inference | `quant_4bit.log` | PASS, footprint 242.9 MB, peak VRAM 517.5 MB |
| Batch sweep | `results_5090.jsonl`, `batch_sweep.log` | PASS, bsz=1/2/4/8 |
| Chunked prefill | `chunked_prefill.log`, `results_5090.jsonl` | PASS, full + chunk sizes 64/128/256 |
| Fused output A/B | `fused_output_ab.log`, `results_5090.jsonl` | PASS, speedup 1.0723x, greedy 32/32 |
| Fused recurrent-output A/B | `fused_recurrent_output_ab.log`, `results_5090.jsonl` | PASS, speedup 1.1963x, greedy 32/32 |
| MATH-style native-prefill dynamic smoke | `math500_native_prefill_smoke/summary.json` | PASS, bsz=8, decoded tokens=512 |

## Batch sweep (0.1B fp16, native_graph token API)

| bsz | native_graph decode tok/s | ms/step | peak VRAM MB |
|---:|---:|---:|---:|
| 1 | 945.9 | 1.06 | 631.1 |
| 2 | 1346.3 | 1.49 | 465.0 |
| 4 | 2714.4 | 1.47 | 519.8 |
| 8 | 5326.4 | 1.50 | 623.0 |

## Chunked prefill (0.1B fp16, bsz=1, prompt=512)

| mode/chunk | prefill tok/s | speed ratio vs full | peak VRAM MB | max diff | decode diff | seq len match |
|---|---:|---:|---:|---:|---:|---|
| full | 24914.3 |  | 658.1 |  |  |  |
| chunk 64 | 3367.7 | 0.1352 | 637.7 | 0.03125 | 0.046875 | True |
| chunk 128 | 6881.5 | 0.2762 | 418.6 | 0.046875 | 0.046875 | True |
| chunk 256 | 13681.5 | 0.5491 | 421.6 | 0.03125 | 0.0625 | True |

## Exact-card fused A/B rows

| axis | baseline ms/step | fused ms/step | speedup | greedy | max abs diff | min cosine | peak VRAM MB |
|---|---:|---:|---:|---:|---:|---:|---:|
| fused output | 1.6093 | 1.5008 | 1.0723x | 32/32 | 0.03125 | 1.0 | 632.2 |
| fused recurrent-output | 1.4937 | 1.2486 | 1.1963x | 32/32 | 0.03125 | 1.0000001192092896 | 632.2 |

## MATH-style native-prefill smoke

Dataset: two synthetic MATH-style rows, rollout 4, bsz 8, max_new_tokens 64. This is a runtime/dynamic-batching smoke, **not** a full MATH500 avg@64 acceptance run.

| metric | value |
|---|---:|
| status | pass |
| dynamic bsz | 8 |
| decoded token events | 512 |
| native prefill sec | 0.623 |
| decode sec | 1.090 |
| generation token/s | 298.8 |
| wall token/s | 243.0 |
| backend | native_graph |

## Scope note

This artifact closes the RTX 5090 HF adapter smoke/support matrix currently required for 50-series work: remote-code import, HF generate/API, native prefill, native_graph decode, dynamic batching, chunked prefill, W8/W4 functional quantized inference, native/no-FLA Trainer smoke, and exact-card fused-kernel A/B rows. It does **not** claim full MATH500 avg@64 acceptance on 5090; the current final MATH500 acceptance evidence remains the 4090 full-run artifacts until the real MATH500 dataset and acceptance model are rerun on 5090.
