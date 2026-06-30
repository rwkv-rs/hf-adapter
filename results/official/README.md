# Official RWKV-7 checkpoint oracle results

These bundles compare the pure-PyTorch HF reference model directly with the
token-wise equations vendored from BlinkDL/RWKV-LM commit
`524481fe1d5d0f321443f1fe025ce03a13f08feb`.  They do not use FLA as a
correctness oracle.

Reference-model source: `d2f8de695826af9ddd0dbf1054f73637b6797953`.

| Hardware | Models | Dtypes | Cases | Status |
|---|---|---|---|---|
| Tesla V100 32GB | 0.1B, 0.4B, 1.5B | FP32, FP16 | B=1/4; T=1/17/128; cached teacher forcing; 64-token greedy | 6/6 passed |
| RTX 4080 16GB | 0.1B, 0.4B, 1.5B | FP32, FP16, BF16 | B=1/4; T=1/17/128; cached teacher forcing; 64-token decode | 9/9 passed |

Each directory contains the raw JSONL record and a rendered Markdown summary.
The raw record includes environment, checkpoint/model hashes, per-case logits
and state metrics, loss, cache/decode results, and the original stricter target
as a diagnostic.  See `docs/EVALUATION.md` for the calibrated release gate and
the reason equivalent CUDA layouts can produce small reduction-order rounding
differences.
