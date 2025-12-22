# W8/W4 quantization status

This page records promoted status and evidence. For copyable bitsandbytes,
native MM8/MM4, Apple MLX W8/W4 commands, acceptance gates, and failure
boundaries, read [`QUANTIZATION_USAGE.md`](QUANTIZATION_USAGE.md) or
[`QUANTIZATION_USAGE.md`](QUANTIZATION_USAGE.md).

## Supported paths

| Path | Purpose | Current status |
|---|---|---|
| bitsandbytes 8-bit / 4-bit | Standard HF compatibility and memory reduction | Functional across tested CUDA cards; not generally faster than native fp16 |
| Native MM8/MM4 `speed` policy | Preserve dense block speed and quantize selected expensive projections | Promoted on measured V100/Tesla-T4/4080/4090/5090 lanes |
| Native MM8/MM4 `memory` policy | Quantize many eligible Linear modules for larger footprint reduction | Functional and memory-saving; universal fp16-or-faster speed is open |
| Apple MLX packed W8/W4 | Apple GPU inference and mobile memory lane | W4 production evidence exists on M5; broader device/shape gates remain |
| CoreML INT8/INT4 | Apple deployment package/runtime path | Stateful correctness and INT8 evidence exist; INT4 quality/ANE placement remains open |

## V100 packed MM4 decode profiles

Exact-sm70 MM4 now has production-gated cached-decode profiles for the
official 1.5B, 2.9B and 7.2B checkpoints. The kernel uses card-local BN/TN
tables, A16 at B1, A8/DP4A at B2/B4/B8, and optional groupwise head scales.

| Model | Policy / head groups | Decode range | Footprint | Complete gate |
|---|---|---:|---:|---:|
| 1.5B | memory / 128 / fused epilogue | `1.0255x-1.1837x` | `0.5395x` | `7/7` |
| 2.9B | speed / 256 / unfused | `1.0111x-1.0346x` | `0.9573x` | `7/7` |
| 7.2B | memory / 128 / unfused | `1.0810x-1.8422x` | `0.3013x` | `7/7` |

Each profile is one load-time configuration across B1/B2/B4/B8, prompt
128/512/2048 and decode 128/512. Every row has lower model footprint, final
cosine `>=0.998`, complete greedy equality and repeat determinism. The weakest
cell for each profile was independently confirmed with five repeats after the
latest-main rebase. Fused epilogues remain default-off outside the exact 1.5B
profile.

This is not a universal full-memory prefill promotion. The 1.5B/7.2B memory
profiles trade prefill speed for footprint; the 2.9B speed profile replaces
only `lm_head`, saves less memory, and separately passes its seven paired
prefill cells at `1.0006x-1.0603x`. Group128 and group256 remain
explicit opt-ins and do not change non-V100 defaults. Copyable configuration
is in [`QUANTIZATION_USAGE.md`](QUANTIZATION_USAGE.md); raw evidence is in
[`../bench/v100_sm70_mm4_bntn_20260716/`](../bench/v100_sm70_mm4_bntn_20260716/README.md).

## Tesla T4 exact-card DP4A lanes

The measured Tesla T4 route reuses the Volta/Turing DP4A extension but is
enabled only for a token-exact T4 device name. RTX 2080, NVIDIA T400 and other
`sm_75` devices remain on their prior fallback until separately validated.

The head-only speed lane passes all 26 W8/W4 decode cells across 0.1B–2.9B at
minimum `1.0207x` fp16. W8/W4 footprint is `0.8686x–0.9716x` /
`0.8043x–0.9578x`, final-logit cosine is at least `0.9999345` / `0.9996467`,
and greedy equality is 26/26.

The full-model lane reduces footprint much further to `0.5291x–0.6331x` W8
and `0.3004x–0.4542x` W4. It wins all B1 decode rows and preserves greedy
tokens 26/26, but prefill is `0.1272x–0.6984x` and small-model B4/B8 decode
can remain below fp16. It is therefore a memory/B1-decode lane, not universal
quantized-speed promotion. Evidence:
[`../bench/t4_production_close_20260720/`](../bench/t4_production_close_20260720/README.md).

## RTX 5090 production BN/TN BF16/W4 model matrix

The exact RTX 5090 g1h route uses group-128 symmetric Marlin W4 for selected
FFN key/value matrices and TorchAO W4 for the head only where the model-level
quality profile permits it. The profile is selected automatically from exact
GPU, dtype, hidden/intermediate size and layer count.

Paired BF16 acceptance at prompt128/decode128 passes every measured phase:

| Model | Batch | Footprint ratio | Prefill speed | Decode speed | Final cosine |
|---|---:|---:|---:|---:|---:|
| 1.5B | 1 | `0.6250x` | `1.2788x` | `1.1854x` | `0.99984407` |
| 1.5B | 8 | `0.6250x` | `1.0097x` | `1.2133x` | `0.99975127` |
| 2.9B | 1 | `0.5776x` | `1.0092x` | `1.2222x` | `0.99965632` |
| 2.9B | 8 | `0.5776x` | `1.0116x` | `1.2894x` | `0.99958199` |
| 7.2B | 1 | `0.5298x` | `1.0010x` | `1.5068x` | `0.99963713` |
| 7.2B | 8 | `0.5298x` | `1.1561x` | `1.4978x` | `0.99954909` |
| 13.3B | 1 | `0.5347x` | `1.0153x` | `1.4957x` | `0.99966073` |
| 13.3B | 8 | `0.5347x` | `1.1549x` | `1.4670x` | `0.99955237` |

All rows preserve the deterministic next token. The route is gated by exact
device name, SM120 capability, BF16 dtype, module role and measured matrix
shape. It does not alter fallback dispatch on any other card. The Marlin
extension is compiled lazily from vendored Apache-2.0 sources and currently
requires a compatible local CUDA toolkit.

The route asserts physical BN/TN per internal scheduler segment, uses a
bit-exact fused FFN-key ReLU-square epilogue, and preserves plain-Linear
semantics for generic HF callers. The expanded group-128 contract passes
280/280 checks across eight FFN directions through 8192 rows; all are
bit-exact against unguarded Marlin and reject a wrong BN. Group-32 experimental
coverage passes another 48/48. Evidence:
[`../bench/5090_bntn_all_models_20260716/README.md`](../bench/5090_bntn_all_models_20260716/README.md).

## CPU-first native memory loading

Large dense checkpoints can exceed GPU memory before native MM8/MM4 packing
has a chance to reduce the model. The end-to-end decode runner now has an
explicit CPU-first route: load dense weights on CPU, apply the native `memory`
policy there, release dense module payloads during replacement, and move only
the packed model to CUDA.

```bash
PYTHONPATH=. python bench/bench_native_quant_e2e_decode.py \
  --hf-dir /path/to/rwkv7-model-hf \
  --device cuda --dtype fp16 \
  --single-quantization mm4 \
  --policy memory --min-params 8000000 \
  --quantize-before-device --allow-missing-baseline \
  --batch-size 1 --prompt-tokens 128 --decode-tokens 128 \
  --results bench/results-memory-mm4.jsonl
```

Use `mm8` instead of `mm4` for W8. The flag intentionally rejects `speed`
policy, non-native formats, multi-quant runs, CPU targets, and in-process paired
baselines. It reduces peak **GPU** loading pressure; the machine still needs
enough host RAM for the dense checkpoint and temporary packing work.

Without a cached fp16 baseline, the row may report packed model footprint and
successful decode, but speed ratio, logits cosine, and greedy parity remain
unset. Such a row is memory-feasibility evidence only. Run a separate fp16 row
on hardware that can fit it before making quality or speed claims.

In Windows PowerShell, set `$env:PYTHONPATH = "."` before the same command and
use backticks for line continuation.

Exact RTX 5070 Laptop 0.4B smoke evidence is in
[`../bench/5070_native_memory_loading_20260716/README.md`](../bench/5070_native_memory_loading_20260716/README.md).

## RTX 4080 output-head speed and full-model memory lanes

The exact RTX 4080 matrix covers official 0.4B/1.5B/2.9B checkpoints at B1/B8,
prompt 128/512/2048 and decode 128/512. Full-model BNB8/BNB4 are memory routes:
all 72 rows execute with finite logits and reduce footprint to
`0.573136x-0.665038x` and `0.359704x-0.497558x` dense. No full-model speed claim
is attached to them.

The paired speed routes replace one output-head module and measure fp16 and
quantized execution in the same process:

| Route | Prefill min | Decode min | `prefill + decode` min | Footprint | Min cosine | Greedy |
|---|---:|---:|---:|---:|---:|---:|
| A8W8 head | telemetry | `>=1.0045x` | `>=1.003101x` | `0.9258x-0.9716x` | `>=0.999931` | 36/36 |
| TorchAO-W4 head | telemetry | `>=1.0246x` | `>=1.015996x` | `0.8907x-0.9612x` | `>=0.999475` | 36/36 |

As on the promoted RTX 3090/4090 lanes, the quant speed contract requires both
cached decode and complete-cell `prefill + decode` latency to be no slower than
fp16. Phase prefill is retained as telemetry and is not independently described
as faster. Direct 8-row A8W8 GEMV and group64 W4 probes were slower than the
selected routes and remain unpromoted.

The 7.2B full-memory MM8/MM4 B1 rows reduce footprint to `0.5346x/0.3015x`
and preserve the measured fp16 greedy sequence, but are slower than fp16.
The 13.3B CPU-first MM8/MM4 routes fit in 16GB and execute deterministically;
because the fp16 model does not fit, they are capacity routes without an fp16
speed or logits-parity claim.

Evidence: [`../bench/4080_full_model_ladder_20260719/README.md`](../bench/4080_full_model_ladder_20260719/README.md).

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
