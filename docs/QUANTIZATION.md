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
