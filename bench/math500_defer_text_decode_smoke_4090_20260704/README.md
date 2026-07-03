# MATH500 deferred text-decode smoke on RTX 4090

This artifact validates the opt-in `--defer-text-decode` path added to `bench/eval_math500_hf.py` on top of deferred verification.

## Command shape

- Model: `/tmp/rwkv7_repo_code_model_dynmath_full_avg64`
- Dataset: `/workspace/projects/Albatross/faster3a_2605/dataset/MATH500.jsonl`
- Tasks / rollout: `--limit 4 --rollout 4`
- Batch size: `4`
- Max new tokens: `256`
- Shared options: `--defer-verification --verify-workers 2 --summary-speed-timing generation`
- Text-decode variant adds: `--defer-text-decode`

## Result

| Metric | Deferred verification | Deferred verification + deferred text decode |
|---|---:|---:|
| Rows | `16` | `16` |
| Correct generations | `3` | `3` |
| Pass@rollout | `0.25` | `0.25` |
| Completion mismatches | `0` | `0` |
| Correctness mismatches | `0` | `0` |
| Stop mismatches | `0` | `0` |
| Decode seconds | `6.970` | `6.845` |
| Generation token/s | `405.818` | `410.009` |

## Conclusion

Deferred text decode preserves completions, correctness, and stop reasons on this dynamic smoke.  It removes per-token `tokenizer.decode(...)` from the decode/refill loop and is enabled only for benchmark runs that opt in.
