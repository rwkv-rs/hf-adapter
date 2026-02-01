# AMD gfx1100 fused decode validation (2026-07-28)

This directory is the exact-card evidence used to promote the existing Triton
decode fusions on AMD `gfx1100`. It is deliberately **not** a family-wide AMD
promotion: other GCN architectures retain the conservative fallback.

## System and scope

- AMD Navi 31, reported by the image as `AMD Radeon Graphics`.
- `gcnArchName=gfx1100`, 47.98 GiB VRAM.
- ROCm 7.2.1, PyTorch `2.9.1+rocm7.2.1.gitff65f5bc`.
- Triton `3.5.1+rocm7.2.1.gita272dfa8`.
- fp16, native HF model, `native_graph` decode.
- Converted G1D 0.1B/0.4B and G1H 1.5B/2.9B checkpoints, batch 1 and 8.

The promoted policy combines:

1. recurrent update + output preparation;
2. raw recurrent state path;
3. output preparation;
4. four-warp norm/shift-mix.

## Full-policy A/B

The baseline disables all four features. The candidate enables the complete
card policy in one graph. Correctness generation is outside the timed region,
so device-to-host token reads do not inflate decode latency.

| model | batch | baseline tok/s | policy tok/s | speedup | max abs | min cosine | greedy |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.1B | 1 | 184.7 | 347.1 | 1.8795x | 0.0625 | 1.0000000 | 32/32 |
| 0.1B | 8 | 1307.0 | 2666.5 | 2.0402x | 0.0625 | 1.0000001 | 256/256 |
| 0.4B | 1 | 81.2 | 141.8 | 1.7458x | 0.03125 | 1.0000001 | 32/32 |
| 0.4B | 8 | 615.8 | 1073.2 | 1.7428x | 0.03125 | 1.0000000 | 256/256 |
| 1.5B | 1 | 51.0 | 71.3 | 1.3987x | 0.0625 | 1.0000001 | 32/32 |
| 1.5B | 8 | 350.1 | 514.2 | 1.4685x | 0.0625 | 1.0000001 | 256/256 |
| 2.9B | 1 | 35.0 | 47.7 | 1.3658x | 0.0625 | 1.0000001 | 32/32 |
| 2.9B | 8 | 250.3 | 353.0 | 1.4102x | 0.0625 | 1.0000000 | 256/256 |

Every row used a 128-token prompt, 32 correctness steps, 8 warmup steps and
256 timed decode steps for 0.1B/0.4B and 128 timed steps for 1.5B/2.9B. Both
sides reported `native_graph`; cache hit rate was above 99% after capture.

`fusion_components.jsonl` contains the preceding component A/B sweep. Every
component compiled on ROCm, preserved the tested greedy stream, and improved
B1 and B8. The four-warp norm/mix launch was selected because it was the best
B8 row (`1.2265x`) while remaining positive at B1 (`1.2039x`).

## Reproduce

```bash
export PYTHONPATH=$PWD
python scripts/sync_hf_adapter_code.py /path/to/rwkv7-g1d-0.4b-hf

for batch in 1 8; do
  python bench/bench_native_graph_policy_ab.py \
    --hf-dir /path/to/rwkv7-g1d-0.4b-hf \
    --dtype fp16 --device cuda --batch-size "$batch" \
    --prompt-tokens 128 --correctness-steps 32 \
    --warmup 8 --steps 256 \
    --min-speedup 1.0 --require-accelerated-policy \
    --results policy_ab.jsonl
done
```

## Promotion boundary

- Exact gate: `torch.cuda.get_device_properties(i).gcnArchName == "gfx1100"`.
- Generic names such as `AMD Radeon Graphics` are not used as identity.
- `gfx1101`, `gfx1102`, MI-series and unknown HIP devices remain off.
- Fused prefill remains an open gate. Output-head MM8/MM4 speed promotion is
  validated separately in `../amd_gfx1100_quant_20260728/`; full-model memory
  quantization remains open.
