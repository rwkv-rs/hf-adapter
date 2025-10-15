# Apple/Qwen3.5 goal audit — Apple M5, 2026-07-07

Derived audit rows from existing Apple M5 evidence directories:

- `../apple_qwen35_08b_tokenonly_m5_20260707/`
- `../apple_qwen35_2b_tokenonly_m5_20260707/`

Command:

```bash
PYTHONPATH=. python bench/audit_qwen35_apple_goal.py \
  --results bench/apple_qwen35_08b_tokenonly_m5_20260707 \
  --results bench/apple_qwen35_2b_tokenonly_m5_20260707 \
  --required-shape chars512:64 \
  --require-quality \
  --require-coreml \
  > bench/apple_qwen35_goal_audit_m5_20260707/results_goal_audit_512_64.jsonl
```

Summary: the current 0.8B and 2B Apple MLX/token-only rows have same-prompt
Qwen and RWKV coverage plus W4/state-cache evidence, but the goal audit remains
non-passing because speed/latency comparison rows fail, quality comparison rows
are missing, long-context rows are missing, and stateful CoreML decode/prefill
runtime evidence is missing.  The 4B and 9B public tiers are still uncovered in
this local audit input.
