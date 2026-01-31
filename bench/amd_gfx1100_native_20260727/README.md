# AMD gfx1100 fully native HF validation — 2026-07-27

This artifact was produced on AMD Navi 31 (`gfx1100`, 47.98 GiB) with ROCm
7.2.1 and PyTorch `2.9.1+rocm7.2.1` from main
`6f7737f68e01a9a2a587d0b7a59a8719ecd68084` plus only the validation runner.
It validates the post-split, fully native HF adapter rather than the legacy FLA
wrapper.

## Decoupling gates

- synced metadata is `native_model.NativeRWKV7ForCausalLM` / `rwkv7_native`;
- legacy `modeling_rwkv7.py` was removed from the converted checkpoint;
- `tests/test_native_model_module_split.py`: 3 passed;
- FLA-blocked direct AutoModel load passed;
- AMD support did not add methods back to `native_model.py`.

## Functional gates

- fp16 AutoModel forward and greedy generation passed;
- generation selected the PyTorch `native_graph` route on this ROCm runtime;
- HF API/beam cache contract passed;
- PEFT LoRA backward produced 72 non-zero gradient tensors;
- dynamic batch/cache select, reorder, compact and continuation passed;
- chunked prefill passed at sizes 1/4/8;
- bf16 HF Trainer + LoRA completed six steps and updated 72/72 trainable tensors.

## 0.1B fp16 baseline

Prompt 128, decode 32, one warmup and two measured prefill runs:

| batch | prefill tok/s total | decode tok/s total | peak VRAM MiB |
|---:|---:|---:|---:|
| 1 | 593.4 | 184.1 | 542.3 |
| 2 | 1057.1 | 344.4 | 582.1 |
| 4 | 1950.2 | 683.7 | 630.2 |
| 8 | 3861.4 | 1304.2 | 712.1 |

For prompt 256, chunk sizes 32/64/128 preserved state length and greedy output;
minimum cosine was `0.99999994` and maximum final/decode absolute difference
was `0.0625`. The no-warmup chunk rows measured 488.9–563.0 tok/s versus a
209.3 tok/s cold full-prefill row; this is cold-start telemetry, not a promoted
speed ratio.

This artifact is a compatibility and baseline-performance result. It does not
claim Albatross parity, production HIP fused kernels, W8/W4 support, or
MI-series coverage.
