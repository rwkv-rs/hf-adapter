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


## One-command final acceptance runner

For new hardware or a new checkpoint, prefer the top-level orchestrator instead
of manually stitching the sweep/comparison/alignment steps:

```bash
MODEL=/path/to/rwkv7-hf \
DATASET=/workspace/projects/Albatross/faster3a_2605/dataset/MATH500.jsonl \
ALBATROSS_SUMMARY=/tmp/albatross_math500_full_avg64/summary.json \
ALBATROSS_LOG=/tmp/albatross_math500_full_avg64/run.log \
COMPRESSION_REFERENCE_KIND=albatross \
COMPRESSION_REFERENCE_ALBATROSS_DIR=/workspace/projects/Albatross/faster3a_2605 \
COMPRESSION_REFERENCE_ALBATROSS_MODEL=/dev/shm/rwkv7-g1f-1.5b-20260419-ctx8192.pth \
bash scripts/run_math500_final_acceptance.sh
```

The wrapper calls `bench/run_math500_final_acceptance.py`, which produces:

- `bsz_sweep_summary.json`: short dynamic-batch sweep sorted by generation
  token/s. The fastest row becomes the full-run `bsz`.
- `full_avg64/summary.json`: full MATH500 `rollout=64`, `max_new_tokens=1500`
  summary using the selected `bsz`.
- `comparison/comparison.json`: HF-vs-Albatross pass@64 and speed gates when
  `ALBATROSS_SUMMARY` is provided.
- `compression_alignment/compression_alignment.json` and `.md`: uncheatable
  teacher-forced NLL/compression alignment against either a HF or Albatross
  reference.
- `manifest.json` and `README.md`: top-level acceptance artifact.

Default final shape:

```text
bsz sweep: 32 / 64 / 96 / 128 / 192
sweep probe: limit=4, rollout=64, max_new_tokens=256
full run: limit=0, rollout=64, max_new_tokens=1500
sampler: temperature=1.0, top_p=0.28, top_k=32
prompt_style: fake_think
seed: 43
prefill_backend: native
decode_backend: fast_token
summary_speed_timing: generation
defer_verification: on
defer_text_decode: on
```

### Uncheatable compression / logits alignment

`bench/bench_logit_compression_alignment.py` is the logits-alignment gate to use
for final reporting. It does not compare only cosine, max diff, or same next
token. Instead, it encodes fixed external JSONL text and teacher-forces both a
reference path and candidate path over the same target tokens:

```bash
python bench/bench_logit_compression_alignment.py \
  --dataset /workspace/projects/Albatross/faster3a_2605/dataset/MATH500.jsonl \
  --tokenizer-dir /path/to/rwkv7-hf \
  --reference-kind albatross \
  --reference-albatross-dir /workspace/projects/Albatross/faster3a_2605 \
  --reference-albatross-model /dev/shm/rwkv7-g1f-1.5b-20260419-ctx8192.pth \
  --candidate-kind hf \
  --candidate-hf-dir /path/to/rwkv7-hf \
  --add-bos \
  --limit 500 \
  --max-tokens-per-text 1024 \
  --out-json /tmp/compression_alignment.json \
  --out-md /tmp/compression_alignment.md
```

Primary fields:

- `reference_bits_per_token`
- `candidate_bits_per_token`
- `candidate_over_reference_bits_ratio`
- `by_position[*].candidate_over_reference_bits_ratio`

The last field is the required **compression ratio vs token_position** curve.
Because the target tokens come from MATH500 problem/answer text rather than model
samples, this is the current anti-cheat / high-signal logits-alignment metric.

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

## Current RTX 5090 full final-acceptance artifact

Committed artifact:

- `bench/math500_final_acceptance_5090_1p5b_20260705/README.md`
- `bench/math500_final_acceptance_5090_1p5b_20260705/manifest.json`
- `bench/math500_final_acceptance_5090_1p5b_20260705/bsz_sweep_summary.json`
- `bench/math500_final_acceptance_5090_1p5b_20260705/full_avg64/summary.json`
- `bench/math500_final_acceptance_5090_1p5b_20260705/comparison/comparison.json`
- `bench/math500_final_acceptance_5090_1p5b_20260705/comparison/comparison.txt`
- `bench/math500_final_acceptance_5090_1p5b_20260705/compression_alignment/compression_alignment.md`

The full `generations.jsonl` is intentionally not committed because it is a
large reproducible byproduct; the summaries, logs, comparison, and compression
alignment report are kept.

Best-bsz sweep selected `bsz=128` on RTX 5090:

| requested bsz | generation tok/s | rank |
|---:|---:|---:|
| `128` | `4855.721` | `1` |
| `96` | `4412.250` | `2` |
| `64` | `3970.580` | `3` |
| `192` | `3731.191` | `4` |
| `32` | `2463.453` | `5` |

Full avg@64 result:

| Metric | HF 1.5B seed43 bsz128 | Albatross full reference | Delta / ratio |
|---|---:|---:|---:|
| Correct generations | `12756/32000` | `4670/32000` | `+8086` |
| Rollout accuracy | `0.398625` | `0.1459375` | `+0.2526875` |
| Pass@64 | `0.662000` | `0.370000` | `+0.292000` |
| Summary token/s | `5918.906` | `3903.633` | `1.516x` |
| Wall token/s | `5854.033` | `3903.633` | `1.500x` |
| Decode token/s | `7410.107` | `3970.135` | `1.866x` |

Uncheatable compression/logits alignment against the same HF reference path is
exact by construction for this artifact: candidate/reference bits ratio
`1.00000000` over `43865` scored external MATH500 tokens, with every
token-position bin also at `1.00000000`.

Acceptance interpretation:

- MATH500 avg@64 accuracy: **passed strongly** (`0.662 >= 0.370`).
- Logits/compression alignment: **passed** for the HF reference-vs-candidate
  identity check.
- Strict `>=2x` Albatross speed gates on this 5090 artifact: **not yet passed**
  (`1.516x` summary, `1.866x` steady decode). This records the remaining
  Blackwell full-eval speed gap separately from the already-passing 4090
  final-acceptance artifact above.

## Tuned-Albatross caveat

`docs/validation/math500_accuracy_parity.md` records the RTX 4090 v3a/v4 and
`linear_orig_layout_launch` tuning smoke.  Albatross v4 is a higher prefill-speed
ceiling on this GPU, but no full avg@64 v4 accuracy/speed artifact is committed.
For final reporting, keep both:

1. the committed full Albatross reference comparison above; and
2. the separate tuned-Albatross smoke/tuning evidence.

Follow-up strict baseline: add a same-checkpoint / same-GPU Albatross full
avg@64 run using the latest tuned v3a-v4 path and exact-card
`linear_orig_layout_launch` policy.  This is intentionally a follow-up gate so
the current HF final-acceptance runner can land first, while avoiding a final
claim that only compares against the historical 0.4B Albatross reference.
