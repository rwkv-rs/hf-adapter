# Apple/Qwen3.5 live smoke — 2026-07-07

Device: macOS 26.5 arm64, Mac17,3, Apple M5, 16GB unified memory.

This folder records the first local Apple live smoke rows for the Qwen3.5
comparison lane. It is not a full acceptance result: Qwen3.5 rows are still
blocked until the Ollama MLX model pull completes.

Files:

- `qwen_pull_preflight_timeout.jsonl` — structured pull-preflight failure row
  showing `qwen3.5:0.8b-mlx` stalled after `pulling model total=1244127078` with
  no byte/status progress before the configured idle timeout.
- `results_rwkv_fp16_smoke.jsonl` — RWKV-7 0.1B HF→MLX fp16 same-schema smoke.
- `results_rwkv_mm4_smoke.jsonl` — RWKV-7 0.1B HF→MLX mm4 same-schema smoke.

Key observed RWKV rows:

| Mode | Prompt tokens | Generated tokens | TTFT | Prefill tok/s | Decode tok/s | MLX active | MLX peak | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| fp16 | 33 | 4 | 0.336768s | 98.351471 | 149.223348 | 387126788 | 388335120 | pass |
| mm4 | 33 | 4 | 0.207046s | 160.016099 | 135.108888 | 311894532 | 313105566 | pass |

Both RWKV rows report `chunked_prefill_max_abs=0.0`. The mm4 row reports 6 Metal
quantized linear calls and lower MLX active/peak memory than the fp16 smoke.
