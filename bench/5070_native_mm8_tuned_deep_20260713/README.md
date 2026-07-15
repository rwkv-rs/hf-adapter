# RTX 5070 Laptop tuned deep-MM8 close

Date: 2026-07-13

Hardware: one `NVIDIA GeForce RTX 5070 Laptop GPU`, observed `sm_120`, 8 GB.

This artifact closes the 1.5B full-memory MM8 speed/footprint gate for the
seven-cell expanded matrix. The path combines the existing default-off FFN-up
ReLU-square and FFN-down residual epilogues with the exact-card MM8 decode
tile selected by a local sweep:

```text
RWKV7_NATIVE_GRAPH_FUSED_QUANT_FFN=1
RWKV7_NATIVE_GRAPH_FUSED_QUANT_FFN_DOWN_ADD=1
RWKV7_NATIVE_MM8_BLOCK_M=64
RWKV7_NATIVE_MM8_BLOCK_N=256
```

`64x256` is also the automatic MM8 decode tile for device names containing
`5070`; environment variables remain authoritative overrides. The fused FFN
flags remain default-off.

| Bsz | Prompt | Decode | MM8 tok/s | fp16 tok/s | MM8 / fp16 |
|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 128 | 121.6 | 105.3 | `1.1548x` |
| 2 | 128 | 128 | 205.1 | 186.8 | `1.0980x` |
| 4 | 128 | 128 | 407.2 | 372.1 | `1.0943x` |
| 8 | 128 | 128 | 772.8 | 717.9 | `1.0765x` |
| 1 | 512 | 128 | 118.0 | 103.8 | `1.1368x` |
| 1 | 2048 | 128 | 117.4 | 102.5 | `1.1454x` |
| 1 | 128 | 512 | 109.7 | 99.4 | `1.1036x` |

All `7/7` rows beat their same-process fp16 baseline. The minimum/median/
maximum ratios are `1.0765x/1.1036x/1.1548x`. Model footprint is `0.6932x`
fp16, minimum prompt/final cosine is `0.99995458/0.99995530`, and all `7/7`
greedy tokens match.

`default_smoke.jsonl` repeats bsz1 without block environment variables. The
runtime resolves `(64, 256)` from the exact card and reaches `1.1404x` fp16
with the same greedy token.

An additional shift-mix-to-key fusion was implemented and measured during the
probe. Although its isolated FFN kernel was faster, it regressed tuned
end-to-end bsz1/2 and was removed before commit. The rejected boundary is
documented in `docs/plans/2026-07-13-native-quant-shift-mix-fusion-design.md`.

This closes MM8 for this exact model/card/matrix. MM4 was subsequently closed
for the same exact matrix by
[`../5070_native_mm4_tuned_deep_20260713/README.md`](../5070_native_mm4_tuned_deep_20260713/README.md).
Other model sizes and Blackwell cards remain open.
