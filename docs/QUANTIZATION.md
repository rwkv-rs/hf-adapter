# W8/W4 quantization status

## Supported paths

| Path | Purpose | Current status |
|---|---|---|
| bitsandbytes 8-bit / 4-bit | Standard HF compatibility and memory reduction | Functional across tested CUDA cards; not generally faster than native fp16 |
| Native MM8/MM4 `speed` policy | Preserve dense block speed and quantize selected expensive projections | Promoted on measured V100/4090/5090 lanes |
| Native MM8/MM4 `memory` policy | Quantize many eligible Linear modules for larger footprint reduction | Functional and memory-saving; universal fp16-or-faster speed is open |
| Apple MLX packed W8/W4 | Apple GPU inference and mobile memory lane | W4 production evidence exists on M5; broader device/shape gates remain |
| CoreML INT8/INT4 | Apple deployment package/runtime path | Stateful correctness and INT8 evidence exist; INT4 quality/ANE placement remains open |

## RTX 4090 g1h 7.2B promoted result

The bsz8 matrix covers prompt 128/512/2048 and decode 128/512. Route
composition selects the BNB8+A8W8-head hybrid for all six W8 cells and native
MM4 or TorchAO per cell for W4.

- W8 minimum prefill/decode/total speed versus RWKV fp16 is
  `1.472988x/1.356914x/1.360072x`; maximum footprint/peak ratio is
  `0.533926x/0.455834x`.
- W4 minimum prefill/decode/total speed is
  `0.976859x/1.022724x/1.013273x`; maximum footprint/peak ratio is
  `0.972617x/0.983054x`. W4 therefore uses the disclosed exact-cell total
  latency fallback rather than claiming every prefill phase is faster.
- BNB8 and MM4 same-quant native/HF probes pass cosine and greedy-token gates.
- Full BNB4 offers deeper compression but is not selected because it misses
  the no-slower speed contract.

Evidence: [`../bench/4090_g1h_7p2_bsz8_20260715/README.md`](../bench/4090_g1h_7p2_bsz8_20260715/README.md).

## RTX 4090 small-model promoted result

The 0.4B, 1.5B and 2.9B pair matrices add 36 selected quant cells, all with
lower model footprint and peak VRAM than matching RWKV fp16. Worst exact-cell
total-latency speedups are:

| RWKV size | W8 total min | W8 footprint/peak max | W4 total min | W4 footprint/peak max |
|---|---:|---:|---:|---:|
| 0.4B | `1.011441x` | `0.925797x / 0.963266x` | `1.029994x` | `0.890672x / 0.945793x` |
| 1.5B | `1.131672x` | `0.560704x / 0.625465x` | `1.027211x` | `0.935468x / 0.968566x` |
| 2.9B | `1.176050x` | `0.544714x / 0.509156x` | `1.014959x` | `0.961227x / 0.977123x` |

W4 prefill is not universally faster (`0.930925x` worst at 1.5B), so the
published claim remains complete-cell non-inferiority, not per-phase
superiority. Native A8W8 or the BNB8+A8W8-head hybrid supplies W8; native MM4
and TorchAO W4 are selected per exact cell.

Evidence: [`../bench/4090_small_bsz8_20260715/README.md`](../bench/4090_small_bsz8_20260715/README.md).

## RTX 5090 promoted result

The 36-row pressure artifact covers 1.5B/2.9B/7.2B × fp16/MM8/MM4 × prompt
128/2048 × decode 128/512 × bsz8.

- All 24 quant rows reduce footprint and preserve the fp16 greedy next token.
- Every 2.9B/7.2B W8/W4 row is within 1% of paired fp16 decode speed.
- The combined matrix passes a conservative 2% equivalence gate.
- One 1.5B W8 row is `0.9841x`; universal strict no-slower is not claimed.
- The earlier g1g 13.3B speed-policy boundaries are W8 `0.9912x` and W4
  `0.9889x`.
- The latest g1h 13.3B B8 prompt128/decode128 rerun measures MM8 `1.0013x`
  and MM4 `0.9845x` paired-fp16 decode, with footprint `0.9899x/0.9848x`,
  cosine above `0.99985`, and matching next tokens. Each row replaces only
  `lm_head`; neither generation is a full-memory quantization claim.

Evidence: [`../bench/5090_blackwell_production_close_20260712/README.md`](../bench/5090_blackwell_production_close_20260712/README.md)
and [`../bench/5090_g1h_13p3_20260715/README.md`](../bench/5090_g1h_13p3_20260715/README.md).

## V100 fused FFN probe

The opt-in native fused FFN epilogue passes isolated MM8/MM4 bsz1/2/4/8
correctness and speed A/B. On the paired 1.5B/bsz1/prompt128/decode128
end-to-end row, MM4 reaches `1.1867x` fp16 with `0.5389x` model footprint and
the same next token. MM8 remains a memory lane at `0.4145x` fp16 speed and
`0.6932x` footprint. This is a selected MM4 close, not a universal
full-memory quant speed promotion, so `RWKV7_NATIVE_GRAPH_FUSED_QUANT_FFN`
remains disabled by default.

Evidence: [`../bench/v100_native_fused_quant_ffn_20260712/README.md`](../bench/v100_native_fused_quant_ffn_20260712/README.md).

## V100 expanded full-memory matrix

The complete 126-row matrix covers 1.5B/2.9B/7.2B and seven batch/context/
decode cells per model. MM4 off and fused-up both beat fp16 in all 21 cells;
fused-up ranges from `1.0415x` to `1.9110x` across the three models while
reducing footprint to `0.5389x/0.5306x/0.3010x`. It is not accepted because
greedy equality is only 6/7, 6/7, and 4/7. MM8 remains a footprint-only path:
all off/up/deep lanes are 0/7 on speed for every model, with ratios
`0.1123x-0.4394x` fp16.

A batched W4A16 probe proved activation quantization alone is not the MM4
quality fix: it matched the dequantized linear oracle but still diverged
end-to-end and reduced bsz8 speed below fp16. Better groupwise/K-style W4 is
the next quality direction. Both fused flags remain default-off.

Evidence: [`../bench/v100_native_quant_full_matrix_20260713/README.md`](../bench/v100_native_quant_full_matrix_20260713/README.md).

## RTX 5070 Laptop deep MM8 matrix

The 1.5B expanded matrix passes 42/42 rows. Full-memory MM8 reduces model
footprint to `0.6932x`; off/up/deep median decode ratios are
`0.9551x/0.9620x/0.9671x` fp16. Deep down+residual fusion wins 5/7 paired
cells over up-only with median `1.0059x`, but its minimum is `0.9888x`.
Full-memory MM4 reduces footprint to `0.5394x` while decoding at a median
`0.8171x` fp16. All 35 quant rows preserve the fp16 greedy token.

`RWKV7_NATIVE_GRAPH_FUSED_QUANT_FFN_DOWN_ADD=1` is independent and default-off;
it also requires `RWKV7_NATIVE_GRAPH_FUSED_QUANT_FFN=1`. Do not project this
small Blackwell gain onto V100, where the measured deep route regresses.

Evidence: [`../bench/5070_native_fused_quant_ffn_20260713/README.md`](../bench/5070_native_fused_quant_ffn_20260713/README.md).

The follow-up MM8 tile sweep selects exact-card `64x256` instead of the prior
Blackwell `128x128` small-row tile. With deep FFN epilogues, the seven 1.5B
cells reach `1.0765x-1.1548x` fp16 at `0.6932x` footprint; minimum final cosine
is `0.9999553` and greedy is 7/7. RTX 5070 selects this tile automatically,
while `RWKV7_NATIVE_MM8_BLOCK_M/N` remain explicit overrides. Fused FFN flags
remain default-off. The MM4 follow-up below closes its matching exact-card
matrix separately.

Evidence: [`../bench/5070_native_mm8_tuned_deep_20260713/README.md`](../bench/5070_native_mm8_tuned_deep_20260713/README.md).

The subsequent MM4 pass adds the matching fused residual epilogue, tunes the
bsz1 paired-nibble GEMV to `64x256`, and routes exact-card bsz2+ through an
output-aware tensor-core dot kernel. The seven 1.5B cells reach
`1.0580x-1.2525x` fp16 at `0.5394x` footprint; minimum final cosine is
`0.99809039` and greedy is 7/7. RTX 5070 selects the measured tiles
automatically, with explicit `RWKV7_NATIVE_MM4_BLOCK_*` and
`RWKV7_NATIVE_MM4_DOT_BLOCK_*` overrides. Both FFN fusion flags remain
default-off, and this result does not promote another card or model size.

Evidence: [`../bench/5070_native_mm4_tuned_deep_20260713/README.md`](../bench/5070_native_mm4_tuned_deep_20260713/README.md).

## RTX 5070 Laptop large-model follow-up

The 2.9B expanded matrix completes 42/42 rows. MM8 off, up, and deep each pass
all 7/7 speed, footprint, and greedy cells; the combined range is
`1.0567x-1.1918x` fp16 at `0.6876x` model footprint. This closes the three
measured MM8 lanes on the exact card/model, but does not justify defaulting a
fusion: up versus off has median `0.9891x`, while deep versus up has a `0.9762x`
minimum. MM4 reaches `1.1012x-1.3834x` and `0.5310x` footprint but greedy is
0/7, so it fails quality acceptance.

For 7.2B on the 8GB card, `--quantize-before-device` loads dense weights on CPU,
packs native MM8/MM4 there, then moves only the quantized model to CUDA. MM4 up
fits at `4140.5 MiB` model / `4769.9 MiB` peak; MM8 deep fits narrowly at
`7340.5 MiB` / `7700.4 MiB`. Dense fp16 is `13731.3 MiB` and cannot provide a
same-card baseline, so the reported `40.1` and `32.7 tok/s` are standalone
telemetry, not fp16 speed claims. The local quant tokens match the exact-shape
V100 fp16 token, but same-card cosine and greedy gates remain unevaluated.

Evidence: [`../bench/5070_native_quant_large_models_20260713/README.md`](../bench/5070_native_quant_large_models_20260713/README.md).

## RTX 5070 Laptop 2.9B groupwise MM4 close

The independent `mm4_groupwise` prototype replaces the old whole-matrix
factorized affine approximation with scale and bias per K group. Exact-weight
oracles showed group32 had the lowest error, but native_graph probes reached
only `0.6045x` fp16; group64 reached `0.9049x`. Group128 preserves the corrected
quality while reducing scale traffic enough for the fused path.

The final group128 implementation uses paired-nibble GEMV for bsz1 and a
tensor-core groupwise batched dot path from bsz2. On 2.9B, all seven strict
cells pass at `1.0895x-1.1656x` fp16 with `0.5402x` footprint, minimum final
cosine `0.99966836`, and greedy 7/7. The format and fused FFN epilogues remain
explicit/default-off; this evidence does not promote V100, another 50-series
card, or 7.2B.

Evidence: [`../bench/5070_native_mm4_groupwise_20260713/README.md`](../bench/5070_native_mm4_groupwise_20260713/README.md).

## Acceptance gate

A promoted quant row must provide:

- lower model footprint than the matching fp16/bf16 baseline;
- finite logits and the configured prompt/final cosine floor;
- same next token for the deterministic check;
- paired or otherwise controlled timing;
- explicit policy and replaced-module count;
- no silent replacement of older-architecture dispatch without card-local A/B.

Example:

```bash
python bench/summarize_blackwell_quant_matrix.py \
  bench/5090_blackwell_production_close_20260712/quant_gap_close.jsonl \
  --gate --expected-rows 36 --min-speed-ratio 0.98
```

## Main open item

The remaining quantization problem is not loading or correctness. It is a fused
full-memory projection/prefill implementation that provides the large W8/W4
payload reduction while remaining fp16-or-faster across old and new cards.
