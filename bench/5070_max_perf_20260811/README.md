# RTX 5070 Laptop exact-card Native performance close

This directory freezes the 2026-08-11 exact-card Native/no-FLA optimization
work on `NVIDIA GeForce RTX 5070 Laptop GPU` (`sm_120`, 8GB, Windows/WDDM).
The accepted routes are deliberately model- and batch-shaped; adjacent RTX
5070 desktop, RTX 5070 Ti Laptop, and RTX 5070 SUPER Laptop names do not inherit
them.

## Environment and gate

- Repository code at the commit produced by this work, FP16 weights,
  `native_graph` cached decode, and repo-code model loading.
- Models: RWKV-7 0.4B (`hidden=1024`, 24 layers) and 1.5B (`hidden=2048`,
  24 layers).
- Matrix: batch `1/2/4/8`, prompt `128/512`, decode `128` unless the artifact
  names a longer correctness trace.
- The WDDM laptop used its normal dynamic power policy. It reports a 50 W
  default and 115 W hardware maximum; no fixed power lock was applied.
- Promotion requires repeated end-to-end improvement of at least `1.01x`,
  first-step cosine at least `0.999`, exact greedy tokens, and no adjacent-card
  policy widening.

Full software and host metadata are in [`environment.json`](environment.json).

## Promoted routes

| Route | Exact policy boundary | Evidence-backed result |
|---|---|---|
| Native prefill graph + fused recurrent scan | 0.4B/1.5B, B1/B2/B4/B8, P128/P512 | output and decode-handoff cosine pass; greedy matches; P128 representative speedups over `native-direct` range from `4.242x` to `89.249x` |
| Fused decode norm/mix | 0.4B B1/B2/B4/B8; 1.5B B1/B2/B8 | 0.4B gains `1.0968x-1.1630x`; 1.5B gains `1.0373x-1.1087x` |
| Raw fused recurrent preparation | 0.4B/1.5B B1/B2/B4/B8 | gains `1.0272x-1.1265x`; all recorded greedy traces match |
| Triton FP16 recurrent state | 0.4B/1.5B B8 only | repeated gains `1.0082x-1.0478x`; allocated VRAM falls `16.875-58.125 MiB` |

The 1.5B/B4 norm/mix row is excluded because its measured speedup is
`0.97337x`. Raw recurrent keeps the default eight-warp launch: all alternate
warp rows stayed below the `1.01x` promotion threshold.

### Representative P128 prefill A/B

| Model | B1 | B2 | B4 | B8 |
|---|---:|---:|---:|---:|
| 0.4B | `89.249x` | `86.686x` | `30.297x` | `13.756x` |
| 1.5B | `27.764x` | `17.240x` | `8.318x` | `4.242x` |

P512 B1/B4/B8 also pass output, state-handoff, and greedy gates. The recorded
0.4B speedups are `160.883x/24.284x/12.966x`; the 1.5B rows are
`27.464x/8.815x/5.000x`.

### Final no-override P128 rows

| Model | Batch | Prefill tok/s | Decode tok/s | Decode ms/step | Peak allocated MiB |
|---|---:|---:|---:|---:|---:|
| 0.4B | 1 | 16,573.2 | 301.9 | 3.31 | 928.3 |
| 0.4B | 2 | 25,584.0 | 532.7 | 3.75 | 977.8 |
| 0.4B | 4 | 30,390.1 | 1,055.0 | 3.79 | 1,042.8 |
| 0.4B | 8 | 28,575.3 | 2,059.2 | 3.88 | 1,169.7 |
| 1.5B | 1 | 8,147.5 | 113.4 | 8.82 | 3,014.5 |
| 1.5B | 2 | 9,328.7 | 208.4 | 9.60 | 3,095.7 |
| 1.5B | 4 | 8,387.5 | 383.5 | 10.43 | 3,207.2 |
| 1.5B | 8 | 8,184.6 | 799.1 | 10.01 | 3,450.6 |

## Rejected candidates

The following remain disabled because end-to-end timing did not meet the
promotion gate, even when an isolated kernel looked attractive:

| Candidate | Measured boundary |
|---|---|
| Fused projection | `0.94467x` B1, `0.97703x` B8 |
| Fused WAVG LoRA | `0.75356x` B1, `0.88400x` B8 |
| Fused WAG LoRA | `0.78727x` B1 |
| Fused output-project | `1.00640x` B1, `1.00328x` B8 |
| Precomputed embedding/LN0 | `1.00548x` B1, `1.00906x` B8 |
| Ada linear override | `0.99837x` B2, `1.00751x` B4 |

## Artifact map

- `promoted_final_default.jsonl`: final B1/B2/B4/B8 endpoint rows.
- `decode_fp16_state_balanced.jsonl`: FP32/FP16 state paired-order A/B.
- `decode_fused_norm_mix_*.jsonl`: accepted and rejected shape gates.
- `decode_fused_recurrent_raw_*.jsonl`: accepted raw recurrent routes.
- `decode_*_balanced.jsonl`: rejected candidates and launch-count probes.
- `prefill_fused_scan_paired_*.jsonl`: P128/P512 graph+scan correctness and
  timing evidence.

## Reproduction

Use a converted local checkpoint and current repo code:

```powershell
python bench/bench_batch_sweep.py `
  --hf-dir D:\path\to\rwkv7-hf `
  --code-source repo --dtype fp16 --device cuda `
  --fast-token-backend native_graph `
  --batch-sizes 1 2 4 8 --prompt-tokens 128 --decode-tokens 128

python bench/bench_native_prefill_scan.py `
  --model D:\path\to\rwkv7-hf `
  --code-source repo --device cuda --dtype fp16 `
  --batch-sizes 1 2 4 8 --prompt-tokens 128,512 `
  --fused-scan auto --reference-backend native-direct --min-cosine 0.9999
```
