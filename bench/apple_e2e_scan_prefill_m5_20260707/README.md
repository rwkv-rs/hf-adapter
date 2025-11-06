# Apple M5 MLX WKV scan prefill end-to-end evidence (2026-07-07)

This directory records the first end-to-end MLX path where the standalone multi-token
WKV scan kernel is wired into `MLXRWKV7Model.forward/prefill` behind the opt-in flag:

```bash
RWKV_WKV_SCAN_PREFILL=1
```

The path converts prefill from token-major execution to layer-major execution.  For
prefill chunks with `T > 1`, each layer now computes the sequence projections and calls
one `wkv_scan(...)` instead of launching one single-token WKV update per token/layer.
Decode remains on the existing single-token path.

## Commands

```bash
RESULTS=bench/apple_e2e_scan_prefill_m5_20260707/results_rwkv04_128_16_scan.jsonl \
PYTHON_BIN=/Users/wangyue/Documents/vllmsp/.venv-apple-torch/bin/python \
RWKV_WKV_SCAN_PREFILL=1 RUN_QWEN=0 RUN_QWEN_MLX_VLM=0 RUN_RWKV=1 \
RWKV_MLX_MODELS=/Users/wangyue/Documents/vllmsp/models/rwkv7-g1d-0.4b-hf \
PROMPT_TARGET_CHARS=128 DECODE_LENGTHS=16 REPEAT=1 WARMUP_REPEATS=0 \
STORE_RESPONSES=0 SKIP_COMPARE=1 SKIP_GOAL_AUDIT=1 \
RWKV_QUANTIZATION=mm4 RWKV_WKV_BACKEND=metal RWKV_CHUNK_SIZE=512 \
bash scripts/run_qwen35_apple_acceptance.sh
```

```bash
RESULTS=bench/apple_e2e_scan_prefill_m5_20260707/results_rwkv15_128_16_scan.jsonl \
PYTHON_BIN=/Users/wangyue/Documents/vllmsp/.venv-apple-torch/bin/python \
RWKV_WKV_SCAN_PREFILL=1 RUN_QWEN=0 RUN_QWEN_MLX_VLM=0 RUN_RWKV=1 \
RWKV_MLX_MODELS=/Users/wangyue/Documents/vllmsp/models/rwkv7-g1g-1.5b-hf \
PROMPT_TARGET_CHARS=128 DECODE_LENGTHS=16 REPEAT=1 WARMUP_REPEATS=0 \
STORE_RESPONSES=0 SKIP_COMPARE=1 SKIP_GOAL_AUDIT=1 \
RWKV_QUANTIZATION=mm4 RWKV_WKV_BACKEND=metal RWKV_CHUNK_SIZE=512 \
bash scripts/run_qwen35_apple_acceptance.sh
```

## Results

| Model | Path | Prefill tok/s | Decode tok/s | TTFT s | Peak memory |
|---|---:|---:|---:|---:|---:|
| RWKV-7 0.4B mm4 | previous token-major Metal WKV | 53.62 | 35.03 | 0.803 | 396 MB |
| RWKV-7 0.4B mm4 | scan prefill | 178.51 | 44.08 | 0.241 | 424 MB |
| RWKV-7 1.5B mm4 | previous token-major Metal WKV | 21.49 | 17.35 | 2.002 | 1.06 GB |
| RWKV-7 1.5B mm4 | scan prefill | 38.11 | 20.47 | 1.130 | 1.12 GB |

## Interpretation

* The scan path is now end-to-end active in `prefill`/`chunked_prefill` when
  `RWKV_WKV_SCAN_PREFILL=1` is set.
* 0.4B short-prompt prefill improved by about 3.33x versus the previous
  token-major end-to-end row.
* 1.5B short-prompt prefill improved by about 1.77x versus the previous
  token-major end-to-end row.
* Decode is mostly unchanged by design because decode still uses the existing
  single-token WKV path.
* The flag remains opt-in while longer prompts, larger batches, quality drift,
  and Qwen3.5 comparison matrices are expanded.
