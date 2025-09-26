# Albatross linear_orig_layout 4090 tuning

GPU: `NVIDIA GeForce RTX 4090` sm `8.9`
Model: `rwkv7-g1d-0.4b-20260210-ctx8192.pth`, dims `{'C': 1024, 'H': 16, 'N': 64, 'V': 65536, 'F': 4096}`

| Case | v4 policy | v4 p50 ms | best label | best p50 ms | best/v4 | max abs diff(best) |
|---|---|---:|---|---:|---:|---:|
| att_c2c_b1t1 | `exact_t128_o2_u1` | 0.020560 | `rows_r1_o4` | 0.020448 | 1.005x | 0.0078125 |
| att_c2c_b64t1 | `lt_ws32_a6` | 0.053696 | `cfg_t32_r3_o4` | 0.025088 | 2.140x | 0.0625 |
| att_c2c_b1t512 | `lt_ws32_a1` | 0.052224 | `orig` | 0.025632 | 2.037x | 0 |
| ffn_key_b1t1 | `exact_t128_o2_u1` | 0.020480 | `rows_r3_o2` | 0.020400 | 1.004x | 0.03125 |
| ffn_key_b64t1 | `lt_ws0_a0` | 0.048128 | `orig` | 0.026624 | 1.808x | 0 |
| ffn_key_b1t512 | `lt_ws128_a3` | 0.075776 | `orig` | 0.041984 | 1.805x | 0 |
| head_b1 | `exact_t128_o2_u1` | 0.148480 | `exact_t128_o2_u0` | 0.148480 | 1.000x | 0.03125 |
| head_b64 | `orig` | 0.167968 | `orig` | 0.167968 | 1.000x | 0 |


## Commands

```bash
python /tmp/bench_albatross_linear_orig_layout.py \
  --albatross-dir /workspace/projects/Albatross/faster3a_2605 \
  --model /workspace/models/rwkv7/raw/rwkv7-g1d-0.4b-20260210-ctx8192.pth \
  --out-json /tmp/albatross_linear_orig_layout_tune_4090_20260704/tuning_report.json \
  --out-md /tmp/albatross_linear_orig_layout_tune_4090_20260704/README.md \
  --warmup 5 --iters 20
```

## Follow-up full-model patch check

The fastest isolated candidates were also tested by patching a temporary v4 C++ copy; see `bench/albatross_v4_linear_policy_patch_4090_20260704/`.  The patch made model-forward slower on the main cases, so these isolated winners are recorded as tuning evidence but are **not** promoted to the tuned Albatross reference.

Interpretation: ratios above `1.0x` mean the current v4 policy is slower than the fastest passing candidate in this synthetic microbench bucket. Use this as per-GPU reference-tuning evidence, not as a drop-in correctness proof for full MATH500.
