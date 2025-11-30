# RTX 5090 TorchAO-only W4 diagnostic — 2026-07-16

This retained negative experiment established the dispatch boundary that led
to the Marlin hybrid. TorchAO tiled W4 is profitable for the 1.5B head and for
7.2B FFN decode rows, but its fixed-cost tinygemm path does not scale to the
7.2B prompt effective-row counts.

| Model | Batch | Footprint / BF16 | Prefill / BF16 | Decode / BF16 | Final cosine | Same token |
|---|---:|---:|---:|---:|---:|---:|---|
| 1.5B | 1 | `0.9355x` | `1.2450x` | `1.0339x` | `0.99968880` | yes |
| 1.5B | 8 | `0.9355x` | `0.9993x` | `1.0193x` | `0.99960572` | yes |
| 7.2B | 1 | `0.5345x` | `0.9176x` | `1.5040x` | `0.99804878` | yes |
| 7.2B | 8 | `0.5345x` | `0.2711x` | `1.4142x` | `0.99795473` | yes |

The isolated role sweep shows why block tuning cannot repair the original
Triton nibble-unpack kernel: its best tested configuration improves the old W4
kernel by about `1.92x` but remains roughly 22 times slower than BF16 for the
representative square projection. The accepted successor is documented in
[`../5090_marlin_w4_hybrid_20260716/README.md`](../5090_marlin_w4_hybrid_20260716/README.md).

Raw files:

- `5090_torchao_w4_role_sweep.jsonl`: exact-role B1/B8 microbench.
- `5090_torchao_speed_*.jsonl`: paired end-to-end prefill/decode rows.
