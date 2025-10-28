# V100 sm70 native-prefill policy evidence (2026-07-10)

This directory promotes the V100-specific native-prefill work from experimental
environment switches into the default `KernelPolicy` for Volta. All speed rows
were measured on one `Tesla V100-PCIE-32GB`, fp16, prompt length 512. Albatross
used the matching official `.pth` checkpoint with `faster3a --wkv fp32io16` on
the same card.

## What changed

The old fused state-prep + recurrent-scan kernel kept the complete 64x64 state
matrix live in one Triton program. On sm70 this is register constrained. The new
kernel splits the output rows (`block_m < head_dim`) while preserving one
program's sequential token recurrence for each row tile.

The promoted Volta policy is batch-aware:

- bsz 1: `block_m=16`, fused state-prep + split-row recurrent scan;
- bsz 2: `block_m=16`, split recurrent scan + separate fused state-prep;
- bsz >=4: `block_m=32`, split recurrent scan + separate fused state-prep;
- all: fused prefill output prep enabled;
- `RWKV7_NATIVE_PREFILL_FUSED_STATE_SCAN_MAX_BATCH=1` limits the larger fused
  state-scan route to the batch where it wins.

Explicit environment variables still override every default. Set
`RWKV7_FAST_PREFILL=0` to disable native prefill globally, or set individual
`RWKV7_NATIVE_PREFILL_*` switches to `0` for A/B and fallback.

## 0.1B result: confirmed route vs Albatross

The HF value is the median of three fresh-process runs. `speedup` compares the
new batch-routed path with the old full-head fused state-scan (`block_m=64`).
The project prefill ladder is P1 >=0.60x and P2 >=0.80x Albatross.

| bsz | HF tok/s | Albatross tok/s | ratio | stage | speedup vs full-head |
|---:|---:|---:|---:|---|---:|
| 1 | 32,058.9 | 39,323.63 | 0.8153x | P2 | 1.1622x |
| 2 | 56,598.4 | 71,382.68 | 0.7929x | P1, near P2 | 1.1884x |
| 4 | 94,135.9 | 109,051.25 | 0.8632x | P2 | 1.2336x |
| 8 | 122,043.7 | 153,368.36 | 0.7958x | P1, near P2 | 1.1925x |

A separate no-environment-override run proves runtime dispatch and records
30,053.4 / 53,182.5 / 93,756.4 / 121,693.8 tok/s for bsz 1/2/4/8. Its rows
show `fused_scan_effective=true`, tile 16/16/32/32, and fused state-scan only at
bsz 1.

## Larger matching-checkpoint rows

| model | bsz | HF tok/s | Albatross tok/s | ratio | stage | peak VRAM |
|---|---:|---:|---:|---:|---|---:|
| 0.4B | 1 | 16,439.1 | 18,462.45 | 0.8904x | P2 | 1,011.2 MiB |
| 0.4B | 2 | 27,492.5 | 31,264.66 | 0.8793x | P2 | 1,014.6 MiB |
| 0.4B | 4 | 38,753.8 | 45,953.77 | 0.8433x | P2 | 1,151.2 MiB |
| 0.4B | 8 | 46,475.0 | 59,046.69 | 0.7871x | P1, near P2 | 1,416.2 MiB |
| 1.5B | 1 | 10,305.4 | 11,911.85 | 0.8651x | P2 | 3,207.9 MiB |
| 1.5B | 2 | 14,419.5 | 16,332.13 | 0.8829x | P2 | 3,189.1 MiB |
| 1.5B | 4 | 17,108.3 | 20,141.39 | 0.8494x | P2 | 3,452.6 MiB |
| 1.5B | 8 | 17,752.3 | 21,807.28 | 0.8141x | P2 | 3,971.5 MiB |

## Correctness and HF integration

Independent correctness uses an unfused native one-token loop as the reference,
not the routed prefill method itself. For 0.1B, 0.4B, and 1.5B at bsz 1 and 4:

- all six prefill argmax checks pass;
- all six cached next-token argmax checks pass;
- cosine is >=0.99999994;
- prefill max-absolute logit difference is <=0.125;
- cached decode max-absolute logit difference is <=0.125.

Ordinary `model.generate(..., use_cache=True)` also passes for all three models
at bsz 1 and 4, reporting `fast_prefill_backend=native_prefill` and
`fast_token_backend=native_graph` without performance environment variables.
The direct split kernel test covers both V-gate branches: split and full-head
outputs are bit-identical, and the torch fallback is within 2e-4.

The V100-safe integration check is reproducible without entering FLA prefill:

```bash
python tests/test_fast_prefill_forward.py \
  --model /path/to/current-hf-model-dir \
  --prompt-tokens 512 --gen-tokens 8 \
  --reference-backend native-token-loop
```

The current repo-code run reports prefill/decode greedy match, generation match,
`min_cos=1.0`, and `seen=512`.

The optional FLA reference path is not a V100 acceptance oracle in this software
stack: disabling native prefill reaches an existing sm70 Triton/LLVM
`Unsupported rounding mode for conversion` failure inside FLA. The independent
native token-loop alignment above avoids both that incompatibility and the
self-reference problem of a policy-routed HF forward.

## Raw files

- `comparison.jsonl`: promoted ratios and stage labels.
- `default_policy.jsonl`: no-env policy dispatch and end-to-end prefill rows.
- `adaptive_confirm.jsonl`: three fresh-process 0.1B confirmations.
- `split_state_scan_sweep.jsonl`: split-row tile sweep, including full-head 64.
- `larger_hf.jsonl` / `larger_albatross.jsonl`: 0.4B and 1.5B rows.
- `token_loop_alignment.jsonl`: independent correctness evidence.
- `generate.jsonl`: ordinary HF generation integration.
- `profile.jsonl`: pre-change full-head bottleneck profile.
- `albatross_0.1b.jsonl`: same-card 0.1B reference.

This closes V100 prefill P1 and reaches P2 on most measured model/batch rows. It
does not claim Albatross parity, quantized-speed completion, or cross-card
promotion; those remain separate acceptance items.
