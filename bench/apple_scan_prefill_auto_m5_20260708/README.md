# Apple M5 MLX scan-prefill auto policy evidence (2026-07-08)

This batch validates the production-shaped scan-prefill policy:

```bash
RWKV_WKV_SCAN_PREFILL=auto
RWKV_WKV_SCAN_PREFILL_MIN_TOKENS=32
```

`auto` keeps single-token decode on the previous per-token path and enables the
multi-token WKV scan for prefill chunks whose token length is at least the
configured threshold.  Telemetry records both the selected mode and reasons:

- `wkv_scan_prefill_mode`
- `wkv_scan_prefill_min_tokens`
- `wkv_scan_prefill_reason_counts`
- `wkv_scan_prefill_counts`

## End-to-end rows

| Model | Shape | Prompt tokens | Prefill tok/s | Decode tok/s | TTFT s | Peak memory | Scan reasons |
|---|---|---:|---:|---:|---:|---:|---|
| RWKV-7 0.4B mm4 | 1024 chars / 128 decode | 326 | 254.27 | 61.29 | 1.285 | 602 MB | `auto=1`, `single_token=128` |
| RWKV-7 1.5B mm4 | 1024 chars / 128 decode | 326 | 61.37 | 28.54 | 5.316 | 1.47 GB | `auto=1`, `single_token=128` |
| RWKV-7 0.4B mm4 | 4096 chars / 128 decode | 1310 | 247.42 | 60.14 | 5.295 | 1.24 GB | `auto=1`, `single_token=128` |
| RWKV-7 1.5B mm4 | 4096 chars / 128 decode | 1310 | 53.60 | 25.40 | 24.443 | 2.08 GB | `auto=1`, `single_token=128` |

## Chunked prefill evidence

4096-char runs use `RWKV_CHUNK_SIZE=512`, so chunked prefill emits multiple scan
calls:

| Model | Shape | Chunked scan calls | State-only intermediate chunks | Chunked max abs diff |
|---|---|---:|---:|---:|
| RWKV-7 0.4B mm4 | 4096 / 128 | 72 | 2 | 0.0625 |
| RWKV-7 1.5B mm4 | 4096 / 128 | 72 | 2 | 0.0625 |

Files:

- `results_rwkv04_1024_128_auto.jsonl`
- `results_rwkv15_1024_128_auto.jsonl`
- `results_rwkv04_4096_128_auto.jsonl`
- `results_rwkv15_4096_128_auto.jsonl`

## Interpretation

The auto policy is now long-context and chunked-prefill validated on Apple M5
for local 0.4B and 1.5B mm4 models.  It does not change decode scheduling:
decode steps are reported as `single_token` and keep the existing single-token
Metal WKV path.  The remaining production work is larger batch validation,
Qwen3.5 same-device comparison refresh, and deeper projection+scan fusion.
