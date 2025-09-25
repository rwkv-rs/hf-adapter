# 4090 MATH500 avg@64 acceptance comparison — 2026-07-03

This benchmark compares the RWKV-7 HF adapter dynamic path against BlinkDL/Albatross under the same full MATH500 shape.

> **Final evaluation standard for this branch.** This comparison follows the
> requester's / bounty-owner's MATH500 avg@64 instruction: full MATH500,
> rollout=64, Albatross sampler/prompt policy, dynamic batching, fastest
> practical GPU speed, and direct HF-adapter vs Albatross speed+accuracy
> comparison. Treat this as the current final acceptance benchmark rather than
> a preliminary smoke benchmark.


## Shape

- Tasks: `500`
- Rollout: `64`
- Generations: `32000`
- Sampler/prompt: temperature=1.0, top_k=32, top_p=0.28, `fake_think`
- GPU: RTX 4090 validation host

## Results

| Metric | HF adapter dynamic | Albatross | Delta / ratio |
|---|---:|---:|---:|
| Correct generations | 4421/32000 | 4670/32000 | -249 |
| Rollout accuracy | 0.13815625 | 0.14593750 | -0.00778125 |
| Pass@64 | 0.358000 | 0.370000 | -0.012000 |
| Summary token/s | 9161.229 | 3903.633 | 2.347x |
| Steady decode token/s | 9215.893 | 3970.135 | 2.321x |
| Sample/s | 14.9449 | 6.3616 | 2.349x |

## Current acceptance interpretation

- Speed target: **passes** for service-style dynamic MATH500; HF adapter is `~2.35x` Albatross by summary token/s.
- Accuracy target: **not yet fully matched**; HF adapter is `-0.012` absolute pass@64 and `-249/32000` correct generations behind Albatross.
- Next gating work: remove the accuracy delta while preserving the dynamic path speed.

## Source artifacts

- HF: `bench/math500_hf_dynamic_full_avg64_20260703/summary.json`, `run.log`
- Albatross: `bench/math500_albatross_full_avg64_20260703/summary.json`, `run.log`
- Machine-readable comparison: `bench/math500_acceptance_4090_20260703/comparison.json`
