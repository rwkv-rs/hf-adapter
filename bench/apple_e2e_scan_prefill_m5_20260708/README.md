# Apple M5 MLX WKV scan prefill second evidence batch (2026-07-08)

This directory extends the first scan-prefill work with longer same-device rows
and an explicit real-model correctness/speed comparison gate:

```bash
scripts/mlx_scan_prefill_compare.py
```

The comparison script loads the same HF RWKV-7 model twice: once with the
previous token-major prefill path and once with `RWKV7_MLX_WKV_SCAN_PREFILL=1`.
It then compares last-token logits, recurrent state drift, and greedy generated
ids, while recording prefill speedup and kernel-count reduction.

## 512 chars / 64 decode end-to-end rows

| Model | Prompt tokens | Prefill tok/s | Decode tok/s | TTFT s | Peak memory | Scan active |
|---|---:|---:|---:|---:|---:|---|
| RWKV-7 0.4B mm4 | 164 | 205.26 | 58.00 | 0.800 | 500 MB | `wkv_scan_prefill_counts.metal=24` |
| RWKV-7 1.5B mm4 | 164 | 65.20 | 29.25 | 2.516 | 1.27 GB | `wkv_scan_prefill_counts.metal=24` |

Files:

- `results_rwkv04_512_64_scan.jsonl`
- `results_rwkv15_512_64_scan.jsonl`

## Real-model scan vs token-major comparison

| Model | Prompt tokens | Token-major prefill tok/s | Scan prefill tok/s | Speedup | Max logit diff | Generated equal |
|---|---:|---:|---:|---:|---:|---|
| RWKV-7 0.4B mm4 | 131 | 57.00 | 221.10 | 3.88x | 0.0625 | true |
| RWKV-7 1.5B mm4 | 131 | 21.93 | 53.52 | 2.44x | 0.0625 | true |

File:

- `results_scan_compare.jsonl`

## Wrapper integration smoke

`results_wrapper_scan_compare_smoke.jsonl` records that the one-command Apple
acceptance wrapper can run the comparison gate through:

```bash
SCAN_PREFILL_COMPARE_MODELS=/path/to/model \
SCAN_PREFILL_COMPARE_PROMPT_TARGET_CHARS=128 \
SCAN_PREFILL_COMPARE_MAX_NEW_TOKENS=4 \
SCAN_PREFILL_COMPARE_FAIL_ON_GATE=1 \
bash scripts/run_qwen35_apple_acceptance.sh
```

## Interpretation

* The scan path stays opt-in, but now has both end-to-end performance rows and
  real-model equivalence rows.
* WKV kernel usage drops from thousands of per-token calls to 24 scan calls for
  the prefill chunk, while decode intentionally remains on the single-token
  path.
* Generated greedy ids match between token-major and scan-prefill paths in the
  recorded 0.4B and 1.5B rows.
