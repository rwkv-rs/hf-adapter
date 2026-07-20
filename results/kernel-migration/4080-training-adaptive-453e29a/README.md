# RTX 4080 adaptive training acceptance

This compact bundle records the canonical LoRA SFT, DPO and GRPO acceptance
for the optional RWKV-7 training leaves. Model weights, raw datasets, Trainer
checkpoints, W&B payloads and large stdout/stderr logs are intentionally not
included.

## Immutable runtime

- `rwkv7-hf` wheel SHA256:
  `7189c8b5615628c19befc6076bea0e80af83fd5b7c522d64ba82e297353a48f9`
- `rwkv7-kernels` wheel SHA256:
  `c814b13f2e82020c6418f7c5fbad8cab5a1310b586b2a256264679f64c84b278`
- Main SFT/DPO/GRPO harness: `60a5be18edb3`
- Affected-only resume correction harness: `cf65bfdf7d21`
- Device: NVIDIA GeForce RTX 4080

## Result

`validation.json` reports `passed` and has SHA256
`c856730cc2e908b6fc23353dcf5387a9b1298844580a29b009cffff4905d61e9`.
Each method completed 100 optimizer steps, recorded finite loss and nonzero
gradients, changed 144 LoRA parameters, and reloaded its saved adapter with
zero logits difference.

| Method | Factorized recurrent calls | Exact matrix calls | Flattened linear calls |
|---|---:|---:|---:|
| SFT | 2,184 | 216 | 28,665 |
| DPO | 480 | 1,920 | 6,660 |
| GRPO | 216 | 2,184 | 2,835 |

The readable Hugging Face model loop remained active. Adaptive routing used
the factorized recurrent and flattened linear leaves only for fully active,
16-token-aligned BF16 requests; masked or unaligned requests used the exact
matrix recurrent and reference linears.

Checkpoint 100 resumed to step 101, and the W&B offline run completed. The
first resume artifact is retained under `sft-resume-failed-60a5be18edb3`:
PEFT had promoted its restored LoRA matrices to FP32 while the old checker
forced a BF16 reload. The corrected checker matched the actual FP32 adapter
runtime and obtained zero logits difference. No completed SFT/DPO/GRPO unit
was rerun for this ancillary checker correction.
