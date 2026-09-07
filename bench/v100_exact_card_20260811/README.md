# Tesla V100 exact-card FP16-state promotion

This artifact records the 2026-08-11 follow-up on one
`Tesla V100-PCIE-32GB` (`sm_70`) from a two-card host. All tuning used GPU 0
only; GPU 1 was left idle. No system packages, clocks, power limits, drivers,
or model files were changed.

## Environment

| Item | Value |
|---|---|
| GPU | Tesla V100-PCIE-32GB, `sm_70`, 32GB |
| Driver | 580.173.02 |
| PyTorch | 2.5.1+cu124 |
| CUDA used by PyTorch | 12.4 |
| Transformers | 5.12.1 |
| Triton | 3.3.0 |
| Models | RWKV-7 0.4B (`hidden=1024`, 24 layers), 1.5B (`hidden=2048`, 24 layers) |
| Runtime | FP16 weights, Native/no-FLA repo code, `native_graph` decode |

## Promoted exact shape

Triton FP16 recurrent state is enabled only for the exact V100 profiles
`(hidden=1024, layers=24, batch=8)` and
`(hidden=2048, layers=24, batch=8)`. B1/B2/B4, other model shapes, and other
Volta products keep FP32 recurrent state unless explicitly overridden.

Two balanced processes use opposite A/B order. Both compare the same loaded
model, prompt, cache state, and graph mode.

| Model | Candidate-first speedup | Baseline-first speedup | Peak allocation delta | First-step cosine | Greedy |
|---|---:|---:|---:|---:|---:|
| 0.4B/B8 | `1.02879x` | `1.02176x` | `-16.875` to `-33.125 MiB` | `0.99999344` | `2,048/2,048` in each process |
| 1.5B/B8 | `1.02163x` | `1.02403x` | `-41.875` to `-58.125 MiB` | `0.99999547` | `2,048/2,048` in each process |

The no-override endpoint confirms policy selection:

| Model | Batch | State | Prefill tok/s | Decode tok/s | Decode ms/step | Peak allocated MiB |
|---|---:|---|---:|---:|---:|---:|
| 0.4B | 1 | FP32 | 10,904.0 | 437.2 | 2.29 | 1,104.2 |
| 0.4B | 8 | FP16 | 45,514.4 | 2,028.6 | 3.94 | 1,273.1 |
| 1.5B | 1 | FP32 | 6,975.2 | 229.5 | 4.36 | 3,752.2 |
| 1.5B | 8 | FP16 | 18,536.6 | 955.1 | 8.38 | 4,072.7 |

## Audited defaults and rejected launch changes

- Existing B8 WAVG fusion remains valuable: the paired 0.4B row is
  `1.36374x` with greedy `512/512`.
- A new WAG fusion is neutral/negative (`1.00146x` B1, `0.99662x` B8), so it
  remains disabled.
- Raw recurrent four-warps and norm/mix eight-warps remained below the `1.01x`
  promotion threshold in confirmation runs; the prior defaults remain.
- Prefill scan block/warp candidates split by prompt length and disappeared in
  longer confirmation timing. The default B1 block-16/four-warp and B8
  block-32/four-warp routes remain unchanged.

All recorded prefill rows pass output cosine, greedy, and decode-after-prefill
handoff gates.

## Artifact map

- `fp16_state_candidate_first.jsonl` and
  `fp16_state_baseline_first.jsonl`: decisive paired-order evidence.
- `final_default.jsonl`: no-override B1/B8 endpoint and effective-route fields.
- `decode_feature_audit.jsonl`: WAVG/WAG audit.
- `decode_*_warps.jsonl`: rejected launch overrides.
- `prefill_default.jsonl`, `prefill_tile_sweep.jsonl`, and
  `prefill_tile_confirm.jsonl`: accepted baseline and rejected tile probes.

## Reproduction

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 PYTHONPATH=. \
python bench/bench_native_graph_state_dtype.py \
  --hf-dir /path/to/rwkv7-hf --batch-size 8 --prompt-tokens 128 \
  --correctness-steps 256 --warmup 16 --steps 256 --paired-rounds 2 \
  --candidate-feature fp16-state --force-candidate --candidate-first

CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 PYTHONPATH=. \
python bench/bench_batch_sweep.py \
  --hf-dir /path/to/rwkv7-hf --code-source repo --dtype fp16 --device cuda \
  --fast-token-backend native_graph --batch-sizes 1 8 \
  --prompt-tokens 128 --decode-tokens 128 --warmup 8 --runs 3
```
