# RTX 5090 BN/TN W4 model-matrix close — 2026-07-16

Status: **production gate passed for the measured g1h 1.5B, 2.9B, 7.2B and
13.3B B1/B8 prompt128/decode128 lanes**. The smaller g1d 0.4B full-FFN
candidate was measured and rejected, so it remains on the previous head-only /
generic fallback.

Environment: NVIDIA GeForce RTX 5090 (SM120, 32 GB), driver 595.58.03,
PyTorch 2.11.0+cu128, Triton 3.6.0, TorchAO 0.17.0, BF16 activations and
symmetric Marlin group-128 W4 FFN weights. Every selected row is a paired
same-process BF16/W4 run after warmup.

## Promoted model profiles

The runtime selects these profiles automatically from exact GPU, dtype,
hidden/intermediate size, layer count, role and weight shape. `dense head`
means that `lm_head` is intentionally not quantized. `skip=1` leaves the final
FFN key/value pair dense to preserve the `0.9995` prompt/final cosine floor.

| Model | Profile | Batch | Footprint/BF16 | Prefill/BF16 | Decode/BF16 | Prompt cosine | Final cosine | Same next |
|---|---|---:|---:|---:|---:|---:|---:|---|
| g1h 1.5B | 46 Marlin FFN, dense head, skip=1 | 1 | `0.6250x` | `1.2788x` | `1.1854x` | `0.99980593` | `0.99984407` | yes |
| g1h 1.5B | same | 8 | `0.6250x` | `1.0097x` | `1.2133x` | `0.99972481` | `0.99975127` | yes |
| g1h 2.9B | 64 Marlin FFN, dense head | 1 | `0.5776x` | `1.0092x` | `1.2222x` | `0.99964404` | `0.99965632` | yes |
| g1h 2.9B | same | 8 | `0.5776x` | `1.0116x` | `1.2894x` | `0.99955779` | `0.99958199` | yes |
| g1h 7.2B | 64 Marlin FFN + TorchAO W4 head | 1 | `0.5298x` | `1.0010x` | `1.5068x` | — | `0.99963713` | yes |
| g1h 7.2B | same | 8 | `0.5298x` | `1.1561x` | `1.4978x` | — | `0.99954909` | yes |
| g1h 13.3B | 120 Marlin FFN + TorchAO W4 head, skip=1 | 1 | `0.5347x` | `1.0153x` | `1.4957x` | `0.99965739` | `0.99966073` | yes |
| g1h 13.3B | same | 8 | `0.5347x` | `1.1549x` | `1.4670x` | `0.99955201` | `0.99955237` | yes |

The 13.3B B1 row uses nine timing repeats because prefill is its tightest
gate. The 7.2B values are the already promoted post-audit rows in
[`../5090_bn_tn_tensorcore_20260716/`](../5090_bn_tn_tensorcore_20260716/README.md).

The separate three-repeat B1 automatic-profile smokes for 1.5B, 2.9B and
13.3B pass without supplying `--quantize-head` or a layer exception. They
resolve exactly to `46/64/121` replaced modules, head policies
`dense/dense/W4`, and skipped-final-layer counts `1/0/1`; see
`e2e_1p5b_auto_smoke.jsonl`, `e2e_2p9b_auto_smoke.jsonl` and
`e2e_13p3b_auto_smoke.jsonl`. Their short timing samples are route smokes, not
replacements for the promoted 5/9-repeat rows above. The 13.3B smoke was
repeated after rebasing the V100 BN/TN work from main and additionally records
complete greedy-sequence equality and repeat determinism.

## Why profiles differ

Blindly quantizing every candidate is not production-safe:

- 1.5B full FFN plus W4 head is fast but final cosine falls to
  `0.99927425/0.99916410` at B1/B8. Keeping the head and final FFN pair dense
  raises the minima above `0.99972` while retaining a `0.6250x` footprint.
- 2.9B full FFN plus W4 head similarly falls below the quality floor. Keeping
  only the head dense closes both speed and quality.
- 13.3B originally OOMed while packing the head after paired dense graph
  construction. Quantization now releases stale dense graph/operand caches and
  packs the FFN matrices before the head. Leaving the final FFN pair dense
  closes the B8 prompt-cosine boundary.
- g1d 0.4B full FFN Marlin measures decode `0.9256x/0.9698x` and cosine below
  `0.9995`; it is deliberately absent from the production shape table.

## Grid and schedule evidence

- Group-128 physical contract: **280/280 pass**, **280/280 bit-exact** against
  unguarded Marlin, and **280/280** intentionally wrong BN checks rejected.
- Group-32 experimental contract: **48/48** for the same three checks.
- The screen covers all eight FFN GEMM directions for 0.4B/1.5B/2.9B/7.2B
  (13.3B shares the 7.2B shape), rows 1/8/128/1024 and five schedules.
- `build_marlin_autotune_profile.py` converts correct, repeat-stable wins into
  an exact-runtime JSON profile. Runtime consumption is explicit through
  `RWKV7_MARLIN_AUTOTUNE_PROFILE`; no unvalidated schedule changes defaults.

The checked-in `autotune_g128.json` and `autotune_g32.json` are reproducible
offline tuner outputs, not globally enabled schedules. The production model
profiles above still use the proven Marlin automatic scheduler and fail closed
outside their exact scope.

## Reproduce

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export TORCH_CUDA_ARCH_LIST=12.0
export PYTHONPATH=$PWD

python bench/bench_marlin_bn_tn_contract.py \
  --shapes 1024x4096 4096x1024 2048x8192 8192x2048 \
           2560x10240 10240x2560 4096x16384 16384x4096 \
  --rows 1 2 3 4 5 7 8 9 15 16 17 24 31 32 33 63 64 65 \
         96 127 128 129 255 256 257 511 512 513 1023 1024 \
         1025 1536 2048 4096 8192 \
  --output /tmp/bntn-contract.jsonl

python bench/bench_native_quant_e2e_decode.py \
  --hf-dir /path/to/rwkv7-g1h-2.9b-hf --model-size-label 2.9b \
  --dtype bf16 --device cuda --attn-mode fused_recurrent \
  --fast-cache true --fast-token-backend native_graph \
  --single-quantization torchao_w4 --min-params 1 --policy speed \
  --batch-size 8 --prompt-tokens 128 --decode-tokens 128 \
  --warmup 1 --timing-repeats 5 --paired-baseline \
  --results /tmp/bntn-2p9b.jsonl
```

With the default `--quantize-head auto` and
`--marlin-skip-last-layers -1`, the second command resolves the promoted 2.9B
profile without user-supplied BN/TN or layer exceptions.
