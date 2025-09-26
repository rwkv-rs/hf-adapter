# MATH500 deferred verification smoke on RTX 4090

This artifact validates the opt-in `--defer-verification` path added to `bench/eval_math500_hf.py`.

## Command shape

- Model: `/tmp/rwkv7_repo_code_model_dynmath_full_avg64`
- Dataset: `/workspace/projects/Albatross/faster3a_2605/dataset/MATH500.jsonl`
- Tasks: `--limit 4`
- Rollout: `4`
- Batch size: `4`
- Max new tokens: `256`
- Seed/RNG: `--seed 43 --rng-mode global`
- Backend: `--prefill-backend native --decode-backend fast_token`

The deferred run adds:

```bash
--defer-verification --verify-workers 2 --summary-speed-timing generation
```

## Result

| Metric | Inline verification | Deferred verification |
|---|---:|---:|
| Rows | `16` | `16` |
| Correct generations | `3` | `3` |
| Pass@rollout | `0.25` | `0.25` |
| Completion mismatches | `0` | `0` |
| Correctness mismatches | `0` | `0` |
| Decode seconds | `8.062` | `6.856` |
| Speed timing | `wall` | `generation` |
| Token/s | `358.850` | `411.810` |
| Wall token/s | `358.850` | `259.081` |
| Generation token/s | `358.850` | `411.810` |
| Verification seconds | inline | `4.983` |

## Conclusion

Deferred verification preserves generated completions and verifier decisions on this dynamic-batching smoke, while moving CPU `math_verify` work out of the GPU decode/refill loop.  It is therefore the correct next full-run path for the remaining G1 speed gate.

A full seed43 avg@64 run was launched on the 4090 server with output path `/tmp/math500_hf_dynamic_full_avg64_seed43_defer_20260704` and log `/tmp/math500_hf_dynamic_full_avg64_seed43_defer_20260704.log`.
