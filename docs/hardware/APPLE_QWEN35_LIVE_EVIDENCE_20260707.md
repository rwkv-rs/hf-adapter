# Apple/Qwen3.5 live evidence notes — 2026-07-07

This is an incremental evidence note for the Apple/Qwen3.5 acceptance lane. It
does **not** claim the full "beat Qwen3.5" gate is complete; it records the
current same-device status and the next hard blocker.

## Device

- macOS 26.5, arm64
- Hardware model: Mac17,3
- Chip string: Apple M5
- Memory: 16GB system memory, ~11.8GiB MLX-visible unified memory
- Python environment: `/Users/wangyue/Documents/vllmsp/.venv-apple-torch/bin/python`

## Qwen3.5/Ollama baseline status

`ollama serve` is reachable at `http://127.0.0.1:11434`, but the raw
`ollama pull qwen3.5:0.8b-mlx` CLI can sit indefinitely at `pulling manifest` /
`pulling model` without byte progress.  A direct `/api/pull` probe returned:

```text
{"status":"pulling manifest"}
{"status":"pulling model","digest":"sha256:model","total":1244127078}
```

and then repeated the same no-completed-byte event.  The wrapper now uses
`scripts/ollama_pull_with_timeout.py` so future `PULL_QWEN=1` runs fail with a
structured `axis=qwen35_apple_ollama_pull` row after the configured idle timeout
instead of hanging forever.


Committed evidence files:

- [`../../bench/apple_qwen35_live_m5_20260707/qwen_pull_preflight_timeout.jsonl`](../../bench/apple_qwen35_live_m5_20260707/qwen_pull_preflight_timeout.jsonl)
- [`../../bench/apple_qwen35_live_m5_20260707/results_rwkv_fp16_smoke.jsonl`](../../bench/apple_qwen35_live_m5_20260707/results_rwkv_fp16_smoke.jsonl)
- [`../../bench/apple_qwen35_live_m5_20260707/results_rwkv_mm4_smoke.jsonl`](../../bench/apple_qwen35_live_m5_20260707/results_rwkv_mm4_smoke.jsonl)

## RWKV-7 MLX smoke rows collected

The local RWKV MLX path is runnable on the same device with the 0.1B converted
HF checkpoint.  Minimal prompt/decode smoke commands:

```bash
PYTHONPATH=. /Users/wangyue/Documents/vllmsp/.venv-apple-torch/bin/python \
  bench/run_qwen35_apple_baseline.py \
  --results bench/apple_qwen35_live_m5_20260707/results_rwkv_fp16_smoke.jsonl \
  --prompt-target-chars 128 \
  --decode-lengths 4 \
  --repeat 1 \
  --qwen-models '' \
  --rwkv-mlx-models /Users/wangyue/Documents/vllmsp/models/rwkv7-g1d-0.1b-hf \
  --rwkv-dtype fp16 \
  --rwkv-quantization none \
  --rwkv-wkv-backend auto \
  --rwkv-chunk-size 64 \
  --store-responses

PYTHONPATH=. /Users/wangyue/Documents/vllmsp/.venv-apple-torch/bin/python \
  bench/run_qwen35_apple_baseline.py \
  --results bench/apple_qwen35_live_m5_20260707/results_rwkv_mm4_smoke.jsonl \
  --prompt-target-chars 128 \
  --decode-lengths 4 \
  --repeat 1 \
  --qwen-models '' \
  --rwkv-mlx-models /Users/wangyue/Documents/vllmsp/models/rwkv7-g1d-0.1b-hf \
  --rwkv-dtype fp16 \
  --rwkv-quantization mm4 \
  --rwkv-quant-min-params 4000000 \
  --rwkv-quant-backend auto \
  --rwkv-wkv-backend auto \
  --rwkv-chunk-size 64 \
  --store-responses
```

Observed rows:

| Mode | Status | Prompt tokens | Generated tokens | TTFT | Prefill tok/s | Decode tok/s | MLX active | MLX peak | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| fp16 | pass | 33 | 4 | 0.336768s | 98.351471 | 149.223348 | 387126788 | 388335120 | chunked prefill max diff 0.0 |
| mm4 | pass | 33 | 4 | 0.207046s | 160.016099 | 135.108888 | 311894532 | 313105566 | 6 Metal quantized linear calls, chunked prefill max diff 0.0 |

Interpretation:

- RWKV MLX load/generate/state-cache smoke is working.
- W4/MM4 reduces active/peak memory on this tiny smoke and keeps the row
  functionally correct.
- The Qwen3.5 comparison is still missing until the Ollama MLX model is pulled
  and live Qwen rows are collected.

## Next evidence step

After the Ollama pull blocker is resolved, run:

```bash
PYTHON_BIN=/Users/wangyue/Documents/vllmsp/.venv-apple-torch/bin/python \
PULL_QWEN=1 \
RUN_QWEN=1 \
QWEN_MODELS=qwen3.5:0.8b-mlx \
RUN_RWKV=1 \
RWKV_MLX_MODELS=/Users/wangyue/Documents/vllmsp/models/rwkv7-g1d-0.4b-hf \
PROMPT_TARGET_CHARS=512 \
DECODE_LENGTHS=64 \
REPEAT=1 \
STORE_RESPONSES=1 \
RWKV_DTYPE=fp16 \
RWKV_QUANTIZATION=mm4 \
RWKV_QUANT_MIN_PARAMS=4000000 \
RWKV_QUANT_BACKEND=auto \
RWKV_WKV_BACKEND=metal \
RWKV_CHUNK_SIZE=512 \
PAIRS=qwen3.5:0.8b-mlx=rwkv7-g1d-0.4b-hf \
FAIL_ON_GATE=0 \
RESULTS=bench/results_qwen35_apple_live_m5.jsonl \
scripts/run_qwen35_apple_acceptance.sh
```
