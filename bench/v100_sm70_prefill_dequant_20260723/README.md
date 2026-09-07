# Tesla V100 exact-sm70 W4 prefill closure — 2026-07-23

Status: **the 1.5B head-only speed profile passes all B1/B2/B4/B8 all-phase
gates; full-memory prefill is substantially faster but remains a capacity
lane**.

This artifact validates an exact-`sm_70` dispatch split:

- cached decode and direct-output calls retain the measured packed-W4 DP4A
  kernels and CUDA-graph-safe output buffers;
- prefill matrices with at least 16 logical rows dequantize W4 directly to a
  temporary FP16 weight and use cuBLAS through `torch.nn.functional.linear`;
- non-`sm_70` devices and smaller row counts retain the prior DP4A path.

The dequantized tensor is temporary. Packed model storage, module selection,
decode schedules and model configuration do not change. The route can be
forced back to DP4A with `RWKV7_SM70_W4_PREFILL_BACKEND=dp4a`; the default
`auto` route selects `dequant_blas` only on exact `sm_70` with `rows >= 16`.

## Environment and gate

- GPU: Tesla V100-PCIE-32GB, CUDA capability `sm_70`
- PyTorch: 2.5.1+cu124; Transformers: 5.12.1
- Checkpoint: official converted `rwkv7-g1g-1.5b-hf`
- Precision/backend: FP16, `fused_recurrent`, `native_graph`
- Shape: prompt 128, decode 128, paired FP16 baseline in each fresh process
- Timing: two warmups and at least five repeats

Every promoted speed row must have lower static model footprint, prefill and
decode ratios `>=1.0x` paired FP16, prompt/final cosine `>=0.9995`, complete
greedy equality and repeat-stable SHA256.

## Head-only group256 speed profile

The `speed + group256 + lm_head` profile replaces one module. The group-size
256 constructor bug was fixed so this is real groupwise storage rather than a
rowwise fallback.

| B | Footprint / FP16 | Prefill / FP16 | Decode / FP16 | Prompt cosine | Final cosine | Complete gate |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | `0.9344x` | `1.0098x` | `1.0353x` | `0.99981844` | `0.99982339` | PASS |
| 2 | `0.9344x` | `1.0119x` | `1.0252x` | `0.99980950` | `0.99981630` | PASS |
| 4 | `0.9344x` | `1.0057x` | `1.0125x` | `0.99980724` | `0.99981266` | PASS |
| 8 | `0.9344x` | `1.0032x` | `1.0011x` | `0.99977553` | `0.99978340` | PASS |

All four rows match the complete FP16 greedy sequence and are deterministic
across five repeats; B8 uses seven repeats. This is the strict V100 quantized
speed acceptance lane for the measured 1.5B P128/D128 matrix.

## Full-memory improvement and boundary

The `memory + group128 + lm_head` profile replaces 49 modules and retains its
`0.5395x` static model footprint. Forced-DP4A and automatic dequant+BLAS runs
use the same code, checkpoint, shape and five-repeat paired methodology.

| B | Forced DP4A prefill / FP16 | Auto prefill / FP16 | Quant tok/s gain | Auto decode / FP16 | Final cosine |
|---:|---:|---:|---:|---:|---:|
| 1 | `0.2820x` | `0.7816x` | `2.790x` | `1.1786x` | `0.99848509` |
| 8 | `0.1171x` | `0.9076x` | `7.731x` | `1.0411x` | `0.99838990` |

This closes the immediate packed-W4 prefill bottleneck by a large margin but
does **not** promote full-memory W4 as universally FP16-or-faster. It remains
the capacity lane until every declared prefill cell reaches the speed gate.

## Reproduce

Run once with the default route and once with the forced old route. Replace
`BATCH` and `OUT` as needed.

```bash
export PYTHONPATH=$PWD
export RWKV_V7_ON=1
export RWKV7_NATIVE_MODEL=1
export RWKV7_FAST_TOKEN_BACKEND=native_graph
export RWKV7_SM70_W4_FUSED_EPILOGUE=1

python bench/bench_native_quant_e2e_decode.py \
  --hf-dir /path/to/rwkv7-g1g-1.5b-hf --code-source repo \
  --model-size-label 1.5b --dtype fp16 --device cuda \
  --attn-mode fused_recurrent --fuse-norm false --fast-cache true \
  --fast-token-backend native_graph --single-quantization mm4 \
  --min-params 8000000 --mm4-group-size 128 \
  --mm4-group-policy lm_head --policy memory \
  --batch-size BATCH --prompt-tokens 128 --decode-tokens 128 \
  --warmup 2 --timing-repeats 5 --paired-baseline --results OUT

RWKV7_SM70_W4_PREFILL_BACKEND=dp4a python \
  bench/bench_native_quant_e2e_decode.py ...
```

For the speed profile, use `--policy speed --mm4-group-size 256` and leave
`RWKV7_SM70_W4_FUSED_EPILOGUE=0`.

Verify the committed rows and raw-file hashes:

```bash
python bench/v100_sm70_prefill_dequant_20260723/check_results.py
cd bench/v100_sm70_prefill_dequant_20260723
shasum -a 256 -c SHA256SUMS
```

`environment.txt` records the runtime and model-config digest. JSONL files are
the machine-readable rows; `.log` files preserve complete command output.
