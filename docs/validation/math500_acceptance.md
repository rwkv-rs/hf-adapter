# MATH500 avg@64 acceptance workflow

This project uses the BlinkDL/Albatross MATH500 evaluation shape as the current
speed+accuracy acceptance overlay for the RWKV-7 HF adapter:

> **Final evaluation standard.** This benchmark is intentionally defined from the
> requester's / bounty-owner's stated acceptance command: use the
> BlinkDL/Albatross `faster3a_2605/eval_math500.py` MATH500 avg@64 workflow,
> find the fastest GPU speed through the best batch policy, and compare both
> speed and MATH500 avg@64 accuracy under the same sampling/prompt/stop policy.
> In this branch, the artifacts below are therefore the current final
> acceptance benchmark, not just a smoke test.


- dataset: full MATH500 (`500` tasks)
- rollout: `64`
- max new tokens: `1500`
- sampler: `temperature -> top_k -> top_p`
- prompt style: `fake_think`
- dynamic batching: `bsz=64`
- prompt prefill cache enabled
- HF path: native prefill + `native_graph` fast-token decode

## Run the HF adapter evaluation

On the 4090 validation host:

```bash
cd /workspace/projects/rwkv7-hf-adapter-060
source /workspace/activate_rwkv7.sh >/dev/null 2>&1 || true
export PYTHONPATH=.
export CUDA_VISIBLE_DEVICES=0
MODEL=/workspace/models/rwkv7/rwkv7-g1d-0.4b-hf \
DATASET=/workspace/projects/Albatross/faster3a_2605/dataset/MATH500.jsonl \
OUT_DIR=/tmp/math500_hf_dynamic_full_avg64 \
bash scripts/run_math500_acceptance.sh
```

The script writes `summary.json` and `generations.jsonl` under `OUT_DIR`.

## Compare against Albatross

After an Albatross run produces `summary.json` and a run log:

```bash
python bench/compare_math500_summaries.py \
  --hf-summary bench/math500_hf_dynamic_full_avg64_20260703/summary.json \
  --albatross-summary /tmp/albatross_math500_full_avg64_20260703/summary.json \
  --albatross-log /tmp/albatross_math500_full_avg64_20260703.log
```

The comparator reports:

- shape compatibility (`num_tasks`, `rollout`, `total_generations`)
- correct generation delta
- rollout accuracy delta
- pass@rollout delta
- summary token/s ratio
- steady decode token/s ratio from the Albatross `dynamic done ... decode_s=...`
  log line when available

## Current 4090 full benchmark artifacts

Committed source artifacts:

- HF adapter dynamic: `bench/math500_hf_dynamic_full_avg64_20260703/summary.json`
  and `bench/math500_hf_dynamic_full_avg64_20260703/run.log`
- Albatross reference: `bench/math500_albatross_full_avg64_20260703/summary.json`
  and `bench/math500_albatross_full_avg64_20260703/run.log`
- Acceptance comparison: `bench/math500_acceptance_4090_20260703/README.md`,
  `comparison.json`, and `comparison.txt`

Current result:

| Metric | HF adapter dynamic | Albatross | Delta / ratio |
|---|---:|---:|---:|
| Correct generations | `4421/32000` | `4670/32000` | `-249` |
| Rollout accuracy | `0.13815625` | `0.1459375` | `-0.00778125` |
| Pass@64 | `0.358` | `0.37` | `-0.012` |
| Summary token/s | `9161.229` | `3903.633` | `2.347x` |
| Steady decode token/s | `9215.893` | `3970.135` | `2.321x` |
| Sample/s | `14.9449` | `6.3616` | `2.349x` |

Acceptance interpretation:

- Service-style dynamic MATH500 speed: **ahead of Albatross** by about `2.3x`.
- MATH500 avg@64 accuracy: **not fully matched yet**; HF adapter trails by
  `1.2` absolute pass@64 points and `249/32000` correct generations.
- The next target is accuracy parity first (`pass@64 >= 0.37` under this exact
  benchmark), while preserving `>= 2x` Albatross dynamic throughput.
