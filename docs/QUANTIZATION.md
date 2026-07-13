# W8/W4 quantization status

## Supported paths

| Path | Purpose | Current status |
|---|---|---|
| bitsandbytes 8-bit / 4-bit | Standard HF compatibility and memory reduction | Functional across tested CUDA cards; not generally faster than native fp16 |
| Native MM8/MM4 `speed` policy | Preserve dense block speed and quantize selected expensive projections | Promoted on measured V100/4090/5090 lanes |
| Native MM8/MM4 `memory` policy | Quantize many eligible Linear modules for larger footprint reduction | Functional and memory-saving; universal fp16-or-faster speed is open |
| Apple MLX packed W8/W4 | Apple GPU inference and mobile memory lane | W4 production evidence exists on M5; broader device/shape gates remain |
| CoreML INT8/INT4 | Apple deployment package/runtime path | Stateful correctness and INT8 evidence exist; INT4 quality/ANE placement remains open |

## RTX 5090 promoted result

The 36-row pressure artifact covers 1.5B/2.9B/7.2B × fp16/MM8/MM4 × prompt
128/2048 × decode 128/512 × bsz8.

- All 24 quant rows reduce footprint and preserve the fp16 greedy next token.
- Every 2.9B/7.2B W8/W4 row is within 1% of paired fp16 decode speed.
- The combined matrix passes a conservative 2% equivalence gate.
- One 1.5B W8 row is `0.9841x`; universal strict no-slower is not claimed.
- 13.3B speed-policy boundaries are W8 `0.9912x` and W4 `0.9889x`; these are
  selected-module speed-policy rows, not full-memory quantization claims.

Evidence: [`../bench/5090_blackwell_production_close_20260712/README.md`](../bench/5090_blackwell_production_close_20260712/README.md).

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
