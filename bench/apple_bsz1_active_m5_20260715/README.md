# Apple M5 B1 W4 active-parameter acceptance

This directory records the batch-1 counterpart to the separately published B8
target-only gate. The checked target-only active-parameter-normalized speed and
raw-memory gates now pass.

## Contract

- MacBook Air, Apple M5, 16 GB unified memory, macOS 26.5.
- MLX 0.31.2, MLX-LM 0.31.3, Transformers 5.12.1.
- RWKV-7 1.5B group-128 W4 versus Qwen3.5 2B MLX group-64 W4.
- True batch 1, 512 prompt characters, 64 generated tokens.
- Target-only RWKV: no draft model, speculative acceptance, or prefix-state coalescing.
- Isolated child processes, two warmups, three measured repeats, ABBA engine
  order, 90-second initial idle, 30 seconds between engines, and a 60-second
  RWKV idle after untimed compile/parity validation.
- Throughput is normalized as aggregate tok/s multiplied by active text
  parameter count. Raw peak memory is the acceptance memory gate.

The Python collector retains its historical `bsz8` name but records
`"batch_size": 1` in every retained row.

## Result

| Metric | RWKV-7 1.5B | Qwen3.5 2B | Ratio | Gate |
|---|---:|---:|---:|---|
| Prefill median | 2,126.06 tok/s | 1,272.86 tok/s | 1.3557x active-normalized | PASS |
| Decode median | 129.15 tok/s | 89.94 tok/s | 1.1655x active-normalized | **PASS** |
| Raw decode | 129.15 tok/s | 89.94 tok/s | 1.4360x | PASS |
| Raw peak memory | 1,189,994,005 bytes | 1,297,316,609 bytes | 0.9173x | PASS |
| Measured rows | 6/6 | 6/6 | n/a | PASS |

Overall status is **PASS**. At the checked active text parameter counts, RWKV
exceeds Qwen by `16.55%` after active-parameter normalization and by `43.60%`
in raw target-only decode throughput. This closes only this fixed M5/B1 profile.

## Implemented optimization

- A B1/N64 SIMD Metal WKV fallback assigns one SIMD group to each state row.
- The B1 production path fuses WKV, per-head GroupNorm, RWKV bonus, and gate.
- Compiled decode chains lazy recurrent outputs with a bounded evaluation interval.
- `decode_greedy_step` keeps full-vocabulary logits inside the compiled graph
  and exports only the next token and recurrent state.
- Asynchronous token-root evaluation every four steps avoids both per-token host
  synchronization and an unbounded lazy graph.
- The recurrent decode cache is stored in FP16 while every WKV state row is
  widened to FP32 registers for recurrence math. This reduces cache-boundary
  traffic and peak memory without changing the 64-step generated-token gate.
- The timed scheduler materializes the complete stacked token stream through one
  graph root. All 73 continuation-cache values are produced on that dependency
  chain; the retained rows verify their post-decode readiness in `12–18 us`.
- Generation no longer computes an unused token N+1 after the final requested token.

Both retained RWKV child processes report exact generated-token agreement and
`state_max_abs=0.0` for the token-only compiled greedy gate. The independent
reference-norm trajectory gate also passes for all 64 steps.

## Evidence

- `active_1p5b_target_only_vs_qwen_2b.jsonl`: six rows per engine plus the
  fail-closed summary.
- `active_1p5b_target_only_vs_qwen_2b.stdout`: complete collector output.

The summary row records:

```text
status=pass
active_normalized_prefill_ratio=1.3557176359
raw_decode_ratio=1.4359605635
active_normalized_decode_ratio=1.1655135771
raw_peak_memory_gate_pass=true
rows_gate_pass=true
```

## Reproduce

Start from an idle machine, wait 90 seconds, truncate the append-only evidence
file, then run:

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
  --rwkv-async-decode --rwkv-async-decode-interval 4 \
  --rwkv-decode-fp16-state \
  --rwkv-fused-lora-down --rwkv-fused-scan-prep-post \
  --rwkv-fused-sequence-mix --rwkv-fused-add-layer-norm \
  --rwkv-fused-square-qmm --no-rwkv-prefix-cache-dedup \
  --results bench/apple_bsz1_active_m5_20260715/active_1p5b_target_only_vs_qwen_2b.jsonl
```

Remove or truncate the results file before reproducing because the collector
intentionally appends JSONL rows. The command exits non-zero unless both
normalized speed gates, raw memory, and all retained rows pass.
