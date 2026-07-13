# RTX 5070 Laptop native quant large-model evidence

Date: 2026-07-13

Hardware: one `NVIDIA GeForce RTX 5070 Laptop GPU`, exact `sm_120`, with
`8151 MiB` visible VRAM.

The official 2.9B and 7.2B checkpoints came from `BlinkDL/rwkv7-g1`. Both
downloads were size- and SHA256-verified before low-memory conversion to fp16
HF safetensors with `fused_recurrent`, no fused norm, and current repository
code.

## 2.9B strict matrix

The expanded matrix completed `42/42` fresh-process rows with no execution
failures. It covers seven batch/context/decode cells and fp16, MM8 off/up/deep,
and MM4 off/up. Every quant row uses an exact-shape same-process fp16 baseline.

| Path | Speed >= fp16 | Decode / fp16 range | Footprint / fp16 | Greedy | Accepted |
|---|---:|---:|---:|---:|---:|
| MM8 off | 7/7 | `1.0870x-1.1887x` | `0.6876x` | 7/7 | yes |
| MM8 up | 7/7 | `1.0567x-1.1906x` | `0.6876x` | 7/7 | yes |
| MM8 deep | 7/7 | `1.1019x-1.1918x` | `0.6876x` | 7/7 | yes |
| MM4 off | 7/7 | `1.1012x-1.3737x` | `0.5310x` | 0/7 | no |
| MM4 up | 7/7 | `1.1518x-1.3834x` | `0.5310x` | 0/7 | no |

MM8 closes the strict speed, footprint, and greedy gates for all three measured
lanes on this exact card and model. The fused-up lane is not uniformly better
than off: its paired median is `0.9891x`, with only two wins. Deep versus up has
median `1.0157x`, five wins, and a `0.9762x` minimum. Both fusion flags therefore
remain default-off despite every independent MM8 lane passing fp16.

MM4 is faster and smaller in every cell, but all seven greedy checks diverge;
minimum final-logits cosine is about `0.9676`. This is a quality failure, not a
completed MM4 path.

## 7.2B 8GB feasibility

Dense fp16 has a `13731.3 MiB` model footprint and cannot fit on this 8GB card.
The benchmark's explicit `--quantize-before-device` mode loads dense weights on
CPU, quantizes there, and moves only the packed model to CUDA. It intentionally
leaves same-card fp16 speed, cosine, and greedy-comparison fields null.

| Path | Model footprint | Footprint / fp16 | Peak VRAM | Decode | Next token |
|---|---:|---:|---:|---:|---:|
| MM4 up | `4140.5 MiB` | `0.3015x` | `4769.9 MiB` | `40.1 tok/s` | `31261` |
| MM8 deep | `7340.5 MiB` | `0.5346x` | `7700.4 MiB` | `32.7 tok/s` | `31261` |

Both quant-only rows execute successfully with `RWKV7StateCache`. Their token
also matches the exact-shape V100 fp16 reference, but that is cross-card
corroboration, not a same-card timing or logits gate. MM8 leaves less than
`451 MiB` between measured peak allocation and visible VRAM, so this row is a
bsz1 boundary smoke and must not be generalized to larger batches.

## Reproduction

The 2.9B raw rows and strict summary are `results-2.9b.jsonl` and
`summary-2.9b.json`. The 7.2B quant-only rows are `results-7.2b.jsonl`.
`summary.json` makes the differing acceptance scopes machine-readable.

For a 7.2B quant-only row, use:

```powershell
python bench\bench_native_quant_e2e_decode.py `
  --hf-dir D:\models\rwkv7\rwkv7-g1g-7.2b-hf --code-source repo `
  --single-quantization mm4 --quantize-before-device `
  --allow-missing-baseline --policy memory --min-params 8000000 `
  --batch-size 1 --prompt-tokens 128 --decode-tokens 128 `
  --fused-quant-ffn --timing-repeats 3
```

All fused quant flags remain default-off. This artifact closes 2.9B MM8 on the
exact RTX 5070 Laptop matrix; it does not close 2.9B MM4, 7.2B same-card speed,
or universal Blackwell defaults.
