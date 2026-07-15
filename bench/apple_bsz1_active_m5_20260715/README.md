# Apple M5 B1 W4 active-parameter acceptance

This directory records the batch-1 counterpart to the separately published B8
target-only gate. The implementation is materially faster than the previous B1
checkpoint, but the active-parameter-normalized decode gate remains open.

## Contract

- MacBook Air, Apple M5, 16 GB unified memory, macOS 26.5.
- MLX 0.31.2, MLX-LM 0.31.3, Transformers 5.12.1.
- RWKV-7 1.5B group-128 W4 versus Qwen3.5 2B MLX group-64 W4.
- True batch 1, 512 prompt characters, 64 generated tokens.
- Target-only RWKV: no draft model, speculative acceptance, or prefix-state coalescing.
- Isolated child processes, two warmups, three measured repeats, ABBA engine
  order, 120-second initial idle, 30 seconds between engines, and a 60-second
  RWKV idle after untimed compile/parity validation.
- Throughput is normalized as aggregate tok/s multiplied by active text
  parameter count. Raw peak memory is the acceptance memory gate.

The Python collector retains its historical `bsz8` name but records
`"batch_size": 1` in every retained row.

## Result

| Metric | RWKV-7 1.5B | Qwen3.5 2B | Ratio | Gate |
|---|---:|---:|---:|---|
| Prefill median | 1,790.94 tok/s | 1,386.37 tok/s | 1.0485x active-normalized | PASS |
| Decode median | 115.89 tok/s | 103.55 tok/s | 0.9084x active-normalized | **FAIL** |
| Raw decode | 115.89 tok/s | 103.55 tok/s | 1.1191x | PASS |
| Raw peak memory | 1,196,022,527 bytes | 1,297,332,993 bytes | 0.9219x | PASS |
| Measured rows | 6/6 | 6/6 | n/a | PASS |

Overall status is **FAIL** only because active-normalized target-only decode is
below 1.0. At the checked active text parameter counts, RWKV needs another
`1.1007x` raw decode uplift. Compared with the superseded B1 checkpoint,
active-normalized decode improved from `0.7392x` to `0.9084x`; raw RWKV decode
now exceeds Qwen by `11.91%`.

## Implemented optimization

- A B1/N64 SIMD Metal WKV fallback assigns one SIMD group to each state row.
- The B1 production path can fuse WKV, per-head GroupNorm, RWKV bonus, and gate.
- Compiled decode chains lazy recurrent outputs with a bounded evaluation interval.
- `decode_greedy_step` keeps the full vocabulary logits inside the compiled graph
  and exports only the next token and recurrent state.
- The token-only graph has an independent 64-step exact-token/exact-state gate
  against the already validated compiled logits graph.
- Generation no longer computes an unused token N+1 after the final requested token.

Both retained RWKV child processes report exact generated-token agreement and
`state_max_abs=0.0` for the token-only compiled greedy gate. The reference-norm
gate also passes.

## Evidence

- `active_1p5b_target_only_vs_qwen_2b.jsonl`: six rows per engine plus the
  fail-closed summary.
- `active_1p5b_target_only_vs_qwen_2b.stdout`: complete collector output.

The summary row records:

```text
status=fail
active_normalized_prefill_ratio=1.0485160294
raw_decode_ratio=1.1191349852
active_normalized_decode_ratio=0.9083585253
raw_peak_memory_gate_pass=true
rows_gate_pass=true
```

## Reproduce

Start from an idle machine, wait 120 seconds, then run:

```bash
PYTHONPATH="$PWD" ../.venv-apple-torch/bin/python \
  bench/run_apple_bsz8_active_compare.py \
  --rwkv-model ../models/rwkv7-g1g-1.5b-hf \
  --qwen-model ../models/qwen35-2b-mlx-4bit \
  --rwkv-draft-model "" \
  --batch-size 1 --prompt-chars 512 --decode-tokens 64 \
  --warmup 2 --repeat 3 --order balanced --cooldown-seconds 30 \
  --rwkv-quant-min-params 1000000 --rwkv-quant-group-size 128 \
  --rwkv-step-eval-interval 256 \
  --rwkv-post-validation-cooldown-seconds 60 \
  --rwkv-decode-backend compiled --rwkv-decode-validation-steps 64 \
  --rwkv-fused-lora-down --rwkv-fused-scan-prep-post \
  --rwkv-fused-sequence-mix --rwkv-fused-add-layer-norm \
  --rwkv-fused-square-qmm --no-rwkv-prefix-cache-dedup \
  --results bench/apple_bsz1_active_m5_20260715/active_1p5b_target_only_vs_qwen_2b.jsonl
```

Remove or truncate the results file before reproducing because the collector
intentionally appends JSONL rows. The command exits non-zero until both
normalized speed gates, raw memory, and all retained rows pass.
