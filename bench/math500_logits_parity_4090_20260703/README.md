# HF vs Albatross logits parity probe

## Config

- Tasks: `[73, 160, 116, 67, 277, 374, 383, 319, 72]`
- Continuation source: `/tmp/albatross_math500_full_avg64_20260703/generations.jsonl`, sample `0`
- Max teacher-forced steps: `64`
- HF dir: `/tmp/rwkv7_repo_code_model_dynmath_full_avg64`
- Albatross module/model: `rwkv7_fast_v3a` / `/workspace/models/rwkv7/raw/rwkv7-g1d-0.4b-20260210-ctx8192.pth`

## Aggregate

- Statuses: `{'pass': 9}`
- Prompt ID mismatches: `0`
- Continuation ID mismatches: `0`

### prefill_forward_vs_albatross
- argmax_match_rate: `1.0`
- cosine_mean/min: `0.9999997748268975` / `0.9999995231628418`
- mean_abs_mean: `0.024562279383341473`
- max_abs_max: `0.15625`
- target NLL delta mean (HF - Albatross): `-0.011285993787977431`

### prefill_native_vs_albatross
- argmax_match_rate: `1.0`
- cosine_mean/min: `0.9999997350904677` / `0.9999995231628418`
- mean_abs_mean: `0.023098049892319575`
- max_abs_max: `0.21875`
- target NLL delta mean (HF - Albatross): `-0.028528001573350694`

### prefill_native_vs_forward
- argmax_match_rate: `1.0`
- cosine_mean/min: `0.9999997085995145` / `0.9999992847442627`
- mean_abs_mean: `0.0264320390092002`
- max_abs_max: `0.1875`
- target NLL delta mean (HF - Albatross): `-0.017242007785373263`

### teacher_forced_all_logits
- steps mean/min/max: `64.0` / `64.0` / `64.0`
- argmax_match_rate mean: `0.9982638888888888`
- cosine mean/min aggregate: `0.9999999765099751` / `0.9999895691871643`
- mean_abs_mean: `0.0155310902337078`
- max_abs_max: `0.484375`
- target NLL delta sum mean (HF - Albatross): `0.01776821295435285`

### teacher_forced_dynamic_path
- steps mean/min/max: `64.0` / `64.0` / `64.0`
- argmax_match_rate mean: `1.0`
- cosine mean/min aggregate: `0.9999999654375844` / `0.9999945759773254`
- mean_abs_mean: `0.016146939239762206`
- max_abs_max: `0.4375`
- target NLL delta sum mean (HF - Albatross): `0.01616100163977939`

## Per-task summary

| Task | Status | Prompt IDs | Cont IDs | Prefill argmax fwd/alb | Prefill argmax native/alb | TF all argmax | TF dynamic argmax |
|---:|---|---:|---:|---:|---:|---:|---:|
| 73 | pass | True | True | True | True | 1.0 | 1.0 |
| 160 | pass | True | True | True | True | 1.0 | 1.0 |
| 116 | pass | True | True | True | True | 1.0 | 1.0 |
| 67 | pass | True | True | True | True | 1.0 | 1.0 |
| 277 | pass | True | True | True | True | 0.984375 | 1.0 |
| 374 | pass | True | True | True | True | 1.0 | 1.0 |
| 383 | pass | True | True | True | True | 1.0 | 1.0 |
| 319 | pass | True | True | True | True | 1.0 | 1.0 |
| 72 | pass | True | True | True | True | 1.0 | 1.0 |

## Interpretation guide

- If `prefill_forward_vs_albatross` is already far from parity, inspect HF weight/layout/math vs Albatross before sampler work.
- If prefill is close but `teacher_forced_dynamic_path` diverges, inspect recurrent state update / fast-token cache path.
- If logits are close but sampled generations still diverge, inspect sampler RNG and dynamic refill order.
