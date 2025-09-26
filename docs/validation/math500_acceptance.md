# MATH500 avg@64 acceptance workflow

This project uses the BlinkDL/Albatross MATH500 evaluation shape as the current
speed+accuracy acceptance overlay for the RWKV-7 HF adapter.

> **Final evaluation standard.** This benchmark follows the requester's /
> bounty-owner's stated acceptance command: use the BlinkDL/Albatross
> `faster3a_2605/eval_math500.py` MATH500 avg@64 workflow, find the fastest GPU
> speed through the best batch policy, and compare both speed and MATH500
> avg@64 accuracy under the same sampling/prompt policy.  The committed
> `bsz=128` deferred-text run below is therefore the current final acceptance
> benchmark, not just a smoke test.

## Benchmark shape

- dataset: full MATH500 (`500` tasks)
- rollout: `64`
- max new tokens: `1500`
- sampler: `temperature -> top_k -> top_p`
- temperature / top-p / top-k: `1.0 / 0.28 / 32`
- prompt style: `fake_think`
- seed: `43`
- dynamic batching: `bsz=128`
- prompt prefill cache enabled
- HF path: native prefill + `native_graph` fast-token decode
- speed timing: generation time (`prefill_sec + decode_sec`)
- CPU verifier: deferred out of the GPU decode/refill loop
- text decode: deferred out of the per-token loop

The deferred verifier/text-decode flags are benchmark-only opt-ins.  Default HF
runtime behavior remains unchanged unless the acceptance benchmark enables them.

## Run the HF adapter evaluation

On the 4090 validation host:

```bash
cd /workspace/projects/rwkv7-hf-adapter-060
source /workspace/activate_rwkv7.sh >/dev/null 2>&1 || true
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0
MODEL=/tmp/rwkv7_repo_code_model_dynmath_full_avg64 \
DATASET=/workspace/projects/Albatross/faster3a_2605/dataset/MATH500.jsonl \
OUT_DIR=/tmp/math500_hf_dynamic_full_avg64_seed43_bsz128_defer_text \
bash scripts/run_math500_acceptance.sh
```

Important defaults in `scripts/run_math500_acceptance.sh`:

```bash
ROLLOUT=64
BSZ=128
MAX_NEW_TOKENS=1500
SEED=43
PREFILL_BACKEND=native
DECODE_BACKEND=fast_token
DEFER_VERIFICATION=1
VERIFY_WORKERS=4
SUMMARY_SPEED_TIMING=generation
DEFER_TEXT_DECODE=1
```

The script writes `summary.json` and `generations.jsonl` under `OUT_DIR`.

## Compare against Albatross and enforce gates

When an Albatross full reference summary/log is available, the same script can
write comparison artifacts and fail non-zero if the acceptance gates are missed:

```bash
MODEL=/tmp/rwkv7_repo_code_model_dynmath_full_avg64 \
DATASET=/workspace/projects/Albatross/faster3a_2605/dataset/MATH500.jsonl \
OUT_DIR=/tmp/math500_hf_dynamic_full_avg64_seed43_bsz128_defer_text \
ALBATROSS_SUMMARY=/tmp/albatross_math500_full_avg64_20260703/summary.json \
ALBATROSS_LOG=/tmp/albatross_math500_full_avg64_20260703.log \
COMPARISON_OUT_DIR=/tmp/math500_hf_dynamic_full_avg64_seed43_bsz128_defer_text/comparison \
bash scripts/run_math500_acceptance.sh
```

Default gates:

- compatible shape: `500` tasks, rollout `64`, `32000` generations
- HF `pass@64 >= 0.370`
- HF / Albatross summary token/s ratio `>= 2.0`
- HF / Albatross steady decode token/s ratio `>= 2.0` when `ALBATROSS_LOG` is provided

The comparator can also be run directly:

```bash
python bench/compare_math500_summaries.py \
  --hf-summary bench/math500_hf_seed43_bsz128_defer_text_full_compare_4090_20260704/hf_seed43_bsz128_defer_text_summary.json \
  --albatross-summary bench/math500_hf_seed43_bsz128_defer_text_full_compare_4090_20260704/albatross_summary.json \
  --albatross-log bench/math500_hf_seed43_bsz128_defer_text_full_compare_4090_20260704/albatross_run.log \
  --require-compatible-shape \
  --min-pass-at-rollout 0.370 \
  --min-summary-speed-ratio 2.0 \
  --min-decode-speed-ratio 2.0 \
  --fail-on-gate
```

## Current 4090 full benchmark artifacts

Passing committed artifact:

- `bench/math500_hf_seed43_bsz128_defer_text_full_compare_4090_20260704/README.md`
- `bench/math500_hf_seed43_bsz128_defer_text_full_compare_4090_20260704/comparison.json`
- `bench/math500_hf_seed43_bsz128_defer_text_full_compare_4090_20260704/comparison.txt`
- `bench/math500_hf_seed43_bsz128_defer_text_full_compare_4090_20260704/hf_seed43_bsz128_defer_text_summary.json`
- `bench/math500_hf_seed43_bsz128_defer_text_full_compare_4090_20260704/albatross_summary.json`

Current result:

| Metric | HF seed43 bsz128 deferred-text | Albatross full reference | Delta / ratio |
|---|---:|---:|---:|
| Correct generations | `4489/32000` | `4670/32000` | `-181` |
| Rollout accuracy | `0.14028125` | `0.14593750` | `-0.00565625` |
| Pass@64 | `0.380000` | `0.370000` | `+0.010000` |
| Summary token/s | `10426.943` | `3903.633` | `2.671x` |
| Wall token/s | `10053.618` | `3903.633` | `2.575x` |
| Decode token/s | `11588.182` | `3970.135` | `2.919x` |

Acceptance interpretation:

- MATH500 avg@64 accuracy: **passed** (`0.380 >= 0.370`).
- Service-style dynamic speed: **passed** (`>=2x` by generation, wall, and decode token/s).
- Correct-generation count remains lower than Albatross, but the current stated
  acceptance gate is avg@64/pass@64 plus speed.

## Tuned-Albatross caveat

`docs/validation/math500_accuracy_parity.md` records the RTX 4090 v3a/v4 and
`linear_orig_layout_launch` tuning smoke.  Albatross v4 is a higher prefill-speed
ceiling on this GPU, but no full avg@64 v4 accuracy/speed artifact is committed.
For final reporting, keep both:

1. the committed full Albatross reference comparison above; and
2. the separate tuned-Albatross smoke/tuning evidence.
