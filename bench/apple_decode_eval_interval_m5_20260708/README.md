# Apple M5 decode eval-interval experiment (2026-07-08)

This batch tested whether increasing `RWKV7_MLX_STEP_EVAL_INTERVAL` from the
current default policy to `16` improves decode throughput after scan-prefill.

File:

- `results_rwkv15_512_64_eval16.jsonl`

Observed row:

| Model | Shape | Eval interval | Prefill tok/s | Decode tok/s | Result |
|---|---|---:|---:|---:|---|
| RWKV-7 1.5B mm4 | 512 / 64 | 16 | 37.25 | 18.75 | regression/no win |

Conclusion: keep the existing wrapper default.  Larger eval intervals did not
produce a reliable decode gain on this Apple M5 run.
