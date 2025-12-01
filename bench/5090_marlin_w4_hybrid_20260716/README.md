# RTX 5090 BF16/W4 Marlin hybrid acceptance — 2026-07-16

> Historical baseline. The production BN/TN grid and fused FFN-key successor
> is [`../5090_bn_tn_tensorcore_20260716/`](../5090_bn_tn_tensorcore_20260716/README.md).

## Result

Exact-card paired-baseline acceptance passes for official g1h 1.5B and 7.2B
at B1/B8, prompt 128 and decode 128. Each row measures steady prefill and
decode five times for 1.5B and three times for 7.2B, reporting medians.

| Model | Batch | W4 route | Footprint / BF16 | Prefill / BF16 | Decode / BF16 | Final cosine | Same token |
|---|---:|---|---:|---:|---:|---:|---|
| 1.5B | 1 | TorchAO asymmetric W4 head | `0.9355x` | `1.0083x` | `1.0335x` | `0.99969822` | yes |
| 1.5B | 8 | TorchAO asymmetric W4 head | `0.9355x` | `1.0090x` | `1.0187x` | `0.99960977` | yes |
| 7.2B | 1 | Marlin symmetric W4 FFN pair + TorchAO W4 head | `0.5298x` | `1.2240x` | `1.4944x` | `0.99963725` | yes |
| 7.2B | 8 | Marlin symmetric W4 FFN pair + TorchAO W4 head | `0.5298x` | `1.0835x` | `1.4872x` | `0.99955124` | yes |

The 7.2B route replaces 64 FFN matrices (`ffn.key` and `ffn.value` in 32
layers) with group-128 Marlin BF16/W4 and keeps the asymmetric TorchAO W4
head. The 4096-square attention projections remain dense because they were not
part of the passing shape set. This is a large-payload hybrid speed result, not
a claim that every projection is quantized.

## Why the route changed

The preceding TorchAO-only experiment passed decode but failed 7.2B prefill:

| Model/batch | TorchAO prefill / BF16 | TorchAO decode / BF16 |
|---|---:|---:|
| 7.2B B1 | `0.9176x` | `1.5040x` |
| 7.2B B8 | `0.2711x` | `1.4142x` |

The isolated Marlin role sweep then showed that both 7.2B FFN directions beat
dense BF16 for effective rows 1, 8, 128 and 1024. Representative speedups are
`5.49x–6.40x` at rows 1/8, `1.14x–2.27x` at row 128, and
`1.23x–1.24x` at row 1024. The packed group-128 payload is 25.78% of a BF16
weight before module-level fixed buffers.

## Environment

- NVIDIA GeForce RTX 5090 32 GB, `sm_120`
- driver `595.58.03`
- PyTorch `2.11.0+cu128`, CUDA toolkit 12.8
- Triton 3.6.0, TorchAO 0.17.0, Transformers 5.12.1
- BF16 model and activation dtype

The vendored Apache-2.0 Marlin torch.ops extension is compiled lazily on first
use and cached by PyTorch. A compatible local CUDA toolkit is currently
required; missing compiler/toolkit is a fail-closed error for this exact path.

## Reproduction

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export TORCH_CUDA_ARCH_LIST=12.0
export PYTHONPATH=$PWD

python bench/bench_native_quant_e2e_decode.py \
  --hf-dir /home/ubuntu/models/rwkv7/rwkv7-g1h-7.2b-hf \
  --model-size-label 7.2b --dtype bf16 --device cuda \
  --attn-mode fused_recurrent --fast-cache true \
  --fast-token-backend native_graph \
  --single-quantization torchao_w4 --min-params 1 --policy speed \
  --batch-size 8 --prompt-tokens 128 --decode-tokens 128 \
  --warmup 1 --timing-repeats 3 --paired-baseline \
  --results /tmp/5090_marlin_speed_7p2_b8.jsonl
```

## Raw files

- `5090_marlin_speed_{1p5,7p2}_b{1,8}.jsonl`: final paired e2e rows.
- `5090_marlin_role_probe.jsonl`: isolated dense-versus-Marlin shape sweep.
- `5090_marlin_environment.txt`: software/hardware versions.
- The failed TorchAO-only prefill evidence is retained in
  `../5090_torchao_w4_hybrid_20260716/`.
