# RTX 5070 Laptop CPU-first native quant loading smoke

Date: 2026-07-16

This artifact validates the explicit `--quantize-before-device` path on one
`NVIDIA GeForce RTX 5070 Laptop GPU` with 8151 MiB visible VRAM. Dense fp16
weights were loaded on CPU, native `memory` policy packing ran on CPU, and only
the packed model was moved to CUDA.

The 0.4B model used `min_params=4000000`, batch 1, prompt 16, and decode 4.
Both rows completed recurrent-cache decode and replaced 49 modules:

| Format | Packed footprint | Dense footprint | Ratio | Peak VRAM | Status |
|---|---:|---:|---:|---:|---|
| MM4 | 477.0 MiB | 859.8 MiB | 0.5548x | 493.1 MiB | pass |
| MM8 | 605.0 MiB | 859.8 MiB | 0.7037x | 621.1 MiB | pass |

These are quant-only smoke rows. They deliberately leave paired speed,
logits-cosine, and greedy-parity fields null, so this artifact proves execution
and lower packed footprint only. The short decode timing is telemetry, not a
speed claim.

Reproduce MM4 from the repository root:

```powershell
$env:PYTHONPATH = "."
python bench\bench_native_quant_e2e_decode.py `
  --hf-dir D:\models\rwkv7\rwkv7-g1d-0.4b-hf `
  --device cuda --dtype fp16 --fast-token-backend auto `
  --single-quantization mm4 --policy memory --min-params 4000000 `
  --quantize-before-device --allow-missing-baseline `
  --batch-size 1 --prompt-tokens 16 --decode-tokens 4 `
  --warmup 0 --timing-repeats 1 --results results.jsonl
```

Use `mm8` for the W8 row. Large-model deployments still require enough host
RAM for dense loading and packing, and must run model-specific quality gates.
