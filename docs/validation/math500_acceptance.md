# MATH500 avg@64 acceptance workflow

This project uses the BlinkDL/Albatross MATH500 evaluation shape as the current
speed+accuracy acceptance overlay for the RWKV-7 HF adapter:

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

## Current 4090 HF full artifact

Committed artifact:

- `bench/math500_hf_dynamic_full_avg64_20260703/summary.json`
- `bench/math500_hf_dynamic_full_avg64_20260703/run.log`

Result:

- `500` tasks, rollout `64`, total generations `32000`
- correct generations: `4421/32000`
- rollout accuracy: `0.13815625`
- pass@64: `0.358`
- truncated rate: `0.21609375`
- mean generated tokens: `612.2159`
- decoded token events: `19,615,994`
- decode time: `2128.4963s`
- elapsed time: `2141.1968s`
- token throughput: `9161.229 tok/s`
- sample throughput: `14.9449 sample/s`

A full Albatross run under the same shape is required before making a final
"exceeds Albatross" claim. A 2-task smoke comparison already matched accuracy
and showed the HF dynamic path ahead on steady decode speed.
