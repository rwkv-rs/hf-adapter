# V100 RWKV-7 vs Qwen3.5 HF matrix

Date: 2026-07-12

Hardware: one `Tesla V100-PCIE-32GB` (`sm_70`) on GPU1.

> Historical diagnostic only: every Qwen row in this artifact used the
> Transformers Torch fallback. The current optimized-Qwen contract requires a
> new exact-card result set with `--qwen-backend fla` and verified FLA core
> operator binding.
> These rows must not be cited as an FLA comparison.

This artifact compares the RWKV-7 HF adapter against official text-only
Qwen3.5 HF checkpoints at matched tensor shapes. It measures inference-engine
speed and memory only. It does not evaluate instruction following, reasoning,
math, code, multilingual quality, or model accuracy.

Results are generated from the completed `current_main_matrix` run only. The
interrupted legacy `full_matrix` output is not part of this artifact.

## Coverage and gate

- Model pairs: RWKV-7 1.5B / Qwen3.5 2B, RWKV-7 2.9B / Qwen3.5 4B, and
  RWKV-7 7.2B / Qwen3.5 9B.
- Shapes: prompt 128/512/2048, decode 128/512, bsz 1/2/4/8.
- Modes: fp16, bitsandbytes 8-bit, and bitsandbytes 4-bit.
- Raw rows: `432/432` pass.
- Joined comparison cells: `216/216` complete.
- Gate: prefill and decode must each be at least `1.05x` Qwen.

The matrix process exits `0`. The comparison process exits `1` because nine
bnb4 decode cells are below the strict `1.05x` speed threshold.

| Metric | Minimum | Median | Maximum | Cells faster than Qwen | Cells >= 1.05x |
|---|---:|---:|---:|---:|---:|
| Prefill | `1.246x` | `1.936x` | `8.141x` | 216/216 | 216/216 |
| Decode | `0.947x` | `1.317x` | `10.832x` | 213/216 | 207/216 |

## Model-pair breakdown

Ranges are minimum / median / maximum RWKV-to-Qwen ratios.

| Pair | Cells | Prefill range | Decode range | Decode >= 1.05x | Model footprint range | Peak VRAM wins |
|---|---:|---:|---:|---:|---:|---:|
| 1.5B / 2B | 72 | `1.412/2.023/8.141x` | `1.054/1.312/10.832x` | 72/72 | `0.729-0.812x` | 60/72 |
| 2.9B / 4B | 72 | `1.751/2.154/8.114x` | `0.955/1.345/9.251x` | 66/72 | `0.694-0.701x` | 72/72 |
| 7.2B / 9B | 72 | `1.246/1.533/3.863x` | `0.947/1.310/3.893x` | 69/72 | `0.629-0.804x` | 60/72 |

## Quantization breakdown

| Mode | Cells | Prefill range | Decode range | Decode >= 1.05x | Model footprint range | Peak VRAM wins |
|---|---:|---:|---:|---:|---:|---:|
| fp16 | 72 | `1.518/2.499/8.141x` | `1.697/5.051/10.832x` | 72/72 | `0.701-0.812x` | 48/72 |
| bnb8 | 72 | `1.246/1.775/2.450x` | `1.054/1.309/1.661x` | 72/72 | `0.698-0.772x` | 72/72 |
| bnb4 | 72 | `1.303/2.008/3.422x` | `0.947/1.207/1.547x` | 63/72 | `0.629-0.729x` | 72/72 |

All nine strict-gate misses are bnb4 decode cells. Six remain faster than Qwen
but are below `1.05x`; three are true decode losses:

| Pair | Prompt | Decode | Bsz | RWKV/Qwen decode |
|---|---:|---:|---:|---:|
| 7.2B / 9B | 2048 | 128 | 1 | `0.947x` |
| 2.9B / 4B | 2048 | 128 | 1 | `0.955x` |
| 2.9B / 4B | 512 | 128 | 2 | `0.998x` |

The full nine-cell list is retained in [`summary.md`](summary.md).

## Memory

RWKV has the lower static model footprint in all `216/216` cells. The ratio is
`0.629x-0.812x` the paired Qwen footprint. Peak allocated VRAM is lower in
`192/216` cells, with an overall ratio range of `0.390x-1.068x`.

The 24 peak-VRAM losses are all fp16: the 1.5B/2B and 7.2B/9B pairs at bsz
1/2/4 each contribute 12 cells. Every bsz8 cell and every bnb8/bnb4 cell uses
less peak VRAM than its Qwen pair.

## Qwen backend limitation

These are official ordinary text-only `Qwen3.5-2B`, `Qwen3.5-4B`, and
`Qwen3.5-9B` checkpoints (`model_type=qwen3_5_text`). On this V100 environment,
all 216 Qwen rows explicitly record:

- `qwen_backend_requested=torch`;
- `qwen_force_torch=true`;
- `qwen_fla_importable=false`;
- `effective_backend=transformers_torch_fallback`.

Therefore the Qwen side is the Transformers/PyTorch fallback, not an optimized
FLA/Triton Qwen backend. The speed ratios are valid for this exact V100 HF
fallback setup and must not be generalized to optimized Qwen deployments on
newer GPUs. The RWKV fp16 rows use `native_graph`; its bnb rows use the FLA
tensor path. Backend differences are part of the measured result and are not a
model-quality claim.

## Files

- [`results.jsonl`](results.jsonl): all 432 raw rows.
- [`summary.json`](summary.json): all 216 joined cells and gate decisions.
- [`summary.md`](summary.md): generated strict-gate report and red cells.
- [`environment.log`](environment.log): GPU and runtime identity.
- `matrix_exit_code.txt`: `0`.
- `compare_exit_code.txt`: `1`, reflecting the nine speed-gate misses.

## Reproduction

```bash
CUDA_VISIBLE_DEVICES=1 \
ROOT=/path/to/hf-adapter \
OUT_DIR=/path/to/current_main_matrix \
PYTHON_BIN=/path/to/python \
bash bench/run_v100_qwen35_speed_matrix.sh
```

The wrapper is resumable and uses `--skip-existing`; reruns must keep the same
results path rather than merging a different matrix.
