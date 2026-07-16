# W8/W4 quantization status

## Supported paths

| Path | Purpose | Current status |
|---|---|---|
| bitsandbytes 8-bit / 4-bit | Standard HF compatibility and memory reduction | Functional across tested CUDA cards; not generally faster than native fp16 |
| Native MM8/MM4 `speed` policy | Preserve dense block speed and quantize selected expensive projections | Promoted on measured V100/4090/5090 lanes |
| Native MM8/MM4 `memory` policy | Quantize many eligible Linear modules for larger footprint reduction | Functional and memory-saving; universal fp16-or-faster speed is open |
| Apple MLX packed W8/W4 | Apple GPU inference and mobile memory lane | W4 production evidence exists on M5; broader device/shape gates remain |
| CoreML INT8/INT4 | Apple deployment package/runtime path | Stateful correctness and INT8 evidence exist; INT4 quality/ANE placement remains open |

## RTX 5090 BF16/W4 Marlin hybrid close

The exact RTX 5090 g1h route now uses two complementary packed kernels under
`torchao_w4 --policy speed`:

- 1.5B keeps the high-accuracy TorchAO asymmetric W4 `lm_head` route.
- 7.2B uses group-128 symmetric Marlin W4 for all 64 FFN key/value matrices
  and TorchAO W4 for `lm_head`; 4096-square projections remain dense.

Paired BF16 acceptance at prompt128/decode128 passes every measured phase:

| Model | Batch | Footprint ratio | Prefill speed | Decode speed | Final cosine |
|---|---:|---:|---:|---:|---:|
| 1.5B | 1 | `0.9355x` | `1.0083x` | `1.0335x` | `0.99969822` |
| 1.5B | 8 | `0.9355x` | `1.0090x` | `1.0187x` | `0.99960977` |
| 7.2B | 1 | `0.5298x` | `1.2240x` | `1.4944x` | `0.99963725` |
| 7.2B | 8 | `0.5298x` | `1.0835x` | `1.4872x` | `0.99955124` |

All rows preserve the deterministic next token. The route is gated by exact
device name, SM120 capability, BF16 dtype, module role and measured matrix
shape. It does not alter fallback dispatch on any other card. The Marlin
extension is compiled lazily from vendored Apache-2.0 sources and currently
requires a compatible local CUDA toolkit.

Evidence: [`../bench/5090_marlin_w4_hybrid_20260716/README.md`](../bench/5090_marlin_w4_hybrid_20260716/README.md).

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

The RTX 5090 7.2B FFN-heavy W4 lane now provides a large payload reduction and
all-phase speed win. The remaining project-wide problem is extending that
result to the still-dense square projections, W8, old cards, Hopper, AMD and
the rest of the declared common-card matrix without regressing any measured
fallback.
