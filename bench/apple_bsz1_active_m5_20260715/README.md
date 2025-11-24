# Apple M5 B1 W4 active-parameter acceptance

This directory records the batch-1 counterpart to the separately published
B8 target-only gate. It is a measured gap, not a passing production claim.

## Contract

- MacBook Air, Apple M5, 16 GB unified memory, macOS 26.5.
- MLX 0.31.2, MLX-LM 0.31.3, Transformers 5.12.1.
- RWKV-7 1.5B group-128 W4 versus Qwen3.5 2B MLX group-64 W4.
- True batch 1, 512 prompt characters, 64 generated tokens.
- Target-only RWKV: no draft model, no speculative acceptance, and no
  prefix-state coalescing.
- Isolated child processes, one warmup, three measured repeats, ABBA engine
  order, 60-second initial cooldown, and 30 seconds between engines.
- Throughput is normalized as aggregate tok/s multiplied by active text
  parameter count. Raw peak memory is the acceptance memory gate.

The Python file retains its historical `bsz8` name but records
`"batch_size": 1` in every retained row.

## Result

| Metric | RWKV-7 1.5B | Qwen3.5 2B | Ratio | Gate |
|---|---:|---:|---:|---|
| Prefill median | 1,590.32 tok/s | 652.39 tok/s | 1.9786x active-normalized | PASS |
| Decode median | 52.72 tok/s | 57.89 tok/s | 0.7392x active-normalized | **FAIL** |
| Raw peak memory | 1,156,011,966 bytes | 1,297,330,945 bytes | 0.8911x raw | PASS |
| Measured rows | 6/6 | 6/6 | n/a | PASS |

Overall status is **FAIL** because the active-normalized target-only decode
ratio is below 1.0. Raw decode is `0.9107x`; RWKV needs about `1.232x` raw
throughput at these active parameter counts to pass the normalized gate.

The retained telemetry also localizes the immediate optimization gap: the
B8/T1 specialized W4 FFN-key + ReLU-squared kernel does not dispatch at B1, so
all B1 FFN-key calls use the fallback path. A direct attempt to reuse the B8
tile at B1 did not produce a reliable positive A/B and was therefore rejected.
The next valid route is a B1-specific narrow-M kernel/tile search followed by
the same fail-closed ABBA rerun; cache or speculative assistance must remain
disabled for this row.

## Evidence

- `active_1p5b_target_only_vs_qwen_2b.jsonl`: six rows per engine plus the
  fail-closed summary.
- `active_1p5b_target_only_vs_qwen_2b.stdout`: complete collector output.

The summary row records:

```text
status=fail
active_normalized_prefill_ratio=1.9785676844
active_normalized_decode_ratio=0.7391882218
raw_peak_memory_gate_pass=true
rows_gate_pass=true
```

## Reproduce

```bash
PYTHONPATH="$PWD" ../.venv-apple-torch/bin/python \
  bench/run_apple_bsz8_active_compare.py \
  --rwkv-model ../models/rwkv7-g1g-1.5b-hf \
  --qwen-model ../models/qwen35-2b-mlx-4bit \
  --rwkv-draft-model "" \
  --batch-size 1 --prompt-chars 512 --decode-tokens 64 \
  --warmup 1 --repeat 3 --order balanced --cooldown-seconds 30 \
  --rwkv-quant-min-params 1000000 --rwkv-quant-group-size 128 \
  --rwkv-fused-lora-down --rwkv-fused-scan-prep-post \
  --rwkv-fused-sequence-mix --rwkv-fused-add-layer-norm \
  --rwkv-fused-square-qmm --no-rwkv-prefix-cache-dedup \
  --results bench/apple_bsz1_active_m5_20260715/active_1p5b_target_only_vs_qwen_2b.jsonl
```

The command intentionally exits non-zero until both normalized speed gates,
the raw memory gate, and every retained row pass.
