# MATH500 accuracy parity exploration

Branch: `wangyue/math500-accuracy-parity`

Base: `wangyue/math500-acceptance-clean` / PR #104.

## Goal

Close the current full MATH500 avg@64 accuracy gap against BlinkDL/Albatross while preserving the HF adapter dynamic-speed advantage.

Current acceptance baseline from `bench/math500_acceptance_4090_20260703`:

| Metric | HF adapter dynamic | Albatross | Gap |
|---|---:|---:|---:|
| Pass@64 | `0.358` | `0.370` | `-0.012` |
| Correct generations | `4421/32000` | `4670/32000` | `-249` |
| Summary token/s | `9161.229` | `3903.633` | `2.347x` |

## Working hypothesis

The speed path is already strong.  The next work should focus on numerical / sampling parity:

1. Compare HF vs Albatross prefill logits for identical prompts.
2. Compare teacher-forced decode logits over fixed tokens.
3. Compare native prefill + fast-token vs forward prefill + forward decode.
4. Isolate whether the gap comes from native prefill, recurrent state update, logits dtype/cast, sampler RNG/refill order, or verifier/stop handling.

## Initial high-signal tasks

From the full-run diff, prioritize small rollout64 subsets before another full 500-task run:

- Albatross advantage: `73`, `160`, `116`, `67`, `277`.
- HF advantage: `374`, `383`, `319`, `72`.

## Albatross reference tuning notes

Do not treat the current Albatross `v3a` reference as the only possible speed ceiling:

- Some Albatross `v4` configurations may be faster than `v3a` on the same model / MATH500 shape.
- `linear_orig_layout_launch` is GPU-sensitive and should be tuned per GPU instead of hard-coded globally.
- The benchmark harness should therefore record the exact Albatross backend/config used for the reference (`v3a` vs `v4`, layout launch policy, GPU name/SM, CUDA/Torch versions).
- For acceptance comparisons, keep two numbers when available:
  1. **fixed-current reference**: the committed `v3a` run used by PR #104;
  2. **best-tuned reference**: the fastest valid Albatross config for that GPU after `linear_orig_layout_launch` tuning.
- A HF speed win should eventually be claimed against the **best-tuned per-GPU Albatross reference**, not only against the first `v3a` baseline.

## Acceptance gate

A parity fix should satisfy:

- `pass@64 >= 0.370` on the full MATH500 benchmark, or statistically clear evidence on targeted subsets before full rerun.
- HF dynamic throughput remains `>= 2x` Albatross on the same 4090 acceptance benchmark.
