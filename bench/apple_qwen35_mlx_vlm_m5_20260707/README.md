# Apple Qwen3.5 MLX-VLM baseline smoke — 2026-07-07

Device: macOS 26.5 arm64, Mac17,3, Apple M5, 16GB unified memory.

This folder records the first real local Hugging Face `mlx-community` Qwen3.5
MLX-VLM baseline row and a same-prompt RWKV-7 0.4B MLX/mm4 comparison row. It is
a short smoke (`prompt-target-chars=128`, `decode-lengths=4`), not the full
0.8B/2B/4B/9B acceptance matrix.

## Source model

- Qwen model files were downloaded from `mlx-community/Qwen3.5-0.8B-MLX-4bit`
  into `/Users/wangyue/Documents/vllmsp/models/qwen35-0.8b-mlx-4bit`.
- Direct `huggingface_hub` / `mlx_vlm.load(repo)` download stalled on this
  machine, while `curl -x http://127.0.0.1:7897 -L .../resolve/main/model.safetensors`
  completed successfully. The runner can use either a repo id or this local dir.

## Files

- `results_qwen35_08b_mlx_vlm.jsonl` — Qwen3.5 0.8B MLX-4bit via `mlx-vlm`.
- `results_rwkv04_mm4.jsonl` — RWKV-7 0.4B HF→MLX mm4 row with the same prompt/decode shape.
- `results_compare.jsonl` — combined rows plus comparison and gap diagnostics.

## Smoke result

| Model | Runtime | Prompt tokens | Generated tokens | TTFT | Prefill tok/s | Decode tok/s | Peak memory | Gate note |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Qwen3.5 0.8B MLX-4bit | mlx-vlm | 32 | 4 | 1.875480s | 17.074117 | 62.404281 | 766525058 B | baseline |
| RWKV-7 0.4B mm4 | rwkv7_hf MLX | 33 | 4 | 0.555066s | 59.581028 | 45.014841 | 514785660 B | faster TTFT/prefill/memory; decode ratio 0.721342 |

Diagnostic row action: `optimize_decode_kernel_or_batching`, requiring about
`1.386305x` decode speedup over the current RWKV 0.4B mm4 short-decode smoke to
match this Qwen3.5 0.8B MLX-VLM row.
