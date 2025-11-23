# RTX 4090 RWKV-7 g1h 7.2B vs Qwen3.5-9B bsz8 acceptance

Date: 2026-07-15

This artifact records an exact-card, resident-model speed and memory matrix on
one NVIDIA GeForce RTX 4090 24GB (`sm_89`). It applies the same fail-closed
bsz8 contract as the promoted RTX 3090 g1h 7.2B artifact to the official
RWKV-7 g1h 7.2B and Qwen3.5-9B checkpoints. The conclusion is limited to the
declared inference cells. It is not a response-quality comparison and does not
generalize to other batch sizes or Ada cards.

## Scope

- RWKV checkpoint: `rwkv7-g1h-7.2b-20260710-ctx10240.pth`, converted through
  the repository low-memory HF converter. Its official ModelScope SHA-256 and
  the converted safetensors SHA-256 are retained.
- Qwen checkpoint: official `Qwen3.5-9B`; SHA-256 hashes for all four model
  shards, index, config and tokenizer files are retained.
- Shapes: prompt 128/512/2048, decode 128/512, batch 8.
- Precision families: dense fp16, W8 and W4; 18 joined cells total.
- Prefill chunking: 512 tokens for both models. Default timing is one warmup
  and three measured runs per resident-model cell. The two dense prompt-128
  rows were rerun with two warmups and five measured runs after the exact-shape
  scan policy was promoted.
- RWKV route: native prefill graph plus `native_graph` cached decode.
- Qwen route: FLA Gated DeltaNet. All six dense reference cells verify all 24
  layers' FLA chunk-prefill, fused-recurrent decode, fused gated norm and
  causal-conv1d prefill/update bindings.
- Quant route composition: `bnb8_a8w8_head` is selected for W8 in all six
  shapes. W4 selects native MM4 in three shapes and TorchAO W4 in three. Full
  BNB4 remains in `memory.jsonl` as a deeper-compression baseline but is not a
  selected speed route.

## Fail-closed gates

Dense RWKV must reach at least `1.00x` Qwen prefill and `1.50x` Qwen cached
decode throughput in every shape. Because RWKV and Qwen have different active
parameter counts, dense decode additionally requires
`(RWKV tok/s * active parameters) / (Qwen tok/s * active parameters) >= 1.00`.
The dense active-parameter ratio is `0.804032x`; this prevents a raw speed win
from hiding lower executed-parameter work on decode. Prefill keeps the direct
token-throughput gate and reports active work separately.

W8/W4 must reduce both physical model footprint and peak allocated VRAM and
must not be slower than matching RWKV fp16. The quant speed check accepts both
phases individually or the explicitly enabled exact-cell `prefill + decode`
total-latency non-inferiority path. Phase metrics are always retained. W4 uses
that fallback: its worst prefill ratio is `0.976859x`, while its worst decode
and total-latency ratios are `1.022724x` and `1.013273x`.

## Result

Overall: **PASS — 18/18 joined cells, zero red cells, zero missing rows, and
all pipeline/checker exit codes are zero.** Every selected dense/W8/W4 cell is
at least as fast as Qwen prefill and exceeds the pair's Qwen decode gate.

| Metric | Minimum | Median | Maximum | Passing cells |
|---|---:|---:|---:|---:|
| Prefill RWKV/Qwen, all families | `1.000256x` | `1.183563x` | `2.098652x` | 18/18 |
| Decode RWKV/Qwen, all families | `2.210065x` | `2.274017x` | `3.026755x` | 18/18 |
| Model footprint RWKV/Qwen | `0.429295x` | `0.781789x` | `0.804034x` | 18/18 no larger |
| Prefill tok/s per active-B | `1.244051x` | `1.472036x` | `2.711255x` | 18/18 |
| Decode tok/s per active-B | `2.748725x` | `2.879065x` | `3.910275x` | 18/18 |

Dense fp16 and quant-local detail:

| Family | Cells | RWKV/Qwen prefill min/median | RWKV/Qwen decode min/median | Prefill vs fp16 min | Decode vs fp16 min | Total vs fp16 min | Footprint vs fp16 max | Peak vs fp16 max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fp16 | 6 | `1.023951x / 1.118749x` | `2.210065x / 2.222768x` | — | — | — | — | — |
| W8 / BNB8+A8W8 head | 6 | `1.508672x / 1.943141x` | `3.002438x / 3.016961x` | `1.472988x` | `1.356914x` | `1.360072x` | `0.533926x` | `0.455834x` |
| W4 / MM4 or TorchAO | 6 | `1.000256x / 1.117632x` | `2.260570x / 2.274017x` | `0.976859x` | `1.022724x` | `1.013273x` | `0.972617x` | `0.983054x` |

For dense fp16, decode active-parameter work rate is
`1.776961x–1.793098x` Qwen (median `1.787177x`) and passes in all 6/6 cells.
Dense prefill active work (`0.823289x–0.951896x`) remains disclosed telemetry;
direct prefill token throughput is the acceptance gate.

The dense RWKV model footprint is `0.804034x` Qwen, but dense peak allocated
VRAM is `1.156353x–1.209017x` Qwen under the shared 512-token chunk policy.
That cross-model peak is telemetry, not a passing memory claim. The required
quant-local memory gate does pass: selected W8 and W4 both lower footprint and
peak VRAM versus matching RWKV fp16 in all cells.

Native-versus-HF same-quant probes pass for BNB8 and MM4 at batch 8/prompt 128.
BNB8 prefill/decode minimum cosine is `0.99996334`/`0.99996018`; MM4 reports
`1.00000024` for both, and both preserve greedy tokens. All performance rows
report finite logits. The focused policy, route-composer and comparison suite
passes 71 tests. These checks do not establish downstream task-quality
superiority.

## Optimizations promoted

- Exact RTX 4090 batch-8/prompt-128 scan tile selection (`block_m=32`); other
  measured prompt/chunk shapes retain the row-8 path.
- Exact RTX 4090 batched tensor-core MM4 output-head dispatch, avoiding eight
  separately captured GEMV kernels and their graph-pool pressure.
- Threshold-zero BNB8 native prefill/decode bridge with graph-safe external
  quantization and card-local mix-kernel block sizes.
- Policy stays exact-card guarded. Non-4090 `sm_89` cards keep their prior
  compatible routes until they have their own measurements.

## Reproduce

After downloading and converting the official models:

```bash
PYTHON_BIN=/path/to/python \
BATCH_SIZES=8 \
PREFILL_CHUNK_SIZE=512 \
  bench/run_4090_qwen35_pair_acceptance.sh \
  rwkv-7.2b__qwen3.5-9b \
  /path/to/rwkv7-g1h-7.2b-hf \
  /path/to/Qwen3.5-9B \
  /tmp/rwkv-g1h-7p2-qwen35-9b
```

The runner rejects non-4090 devices and fails closed on missing rows,
non-native RWKV routes, Qwen optimized-operator binding failure, dense speed,
dense decode active work, quant speed and quant-local memory. It records
independent exit codes for route composition and all three comparators.

## Artifacts

- `combined_auto.jsonl`: selected dense/W8/W4 candidate and reference rows.
- `dense.jsonl`, `memory.jsonl`, `native_speed.jsonl`,
  `hybrid_speed.jsonl`: retained route inputs, including initial failed
  alternatives and successful reruns.
- `route_manifest.json`: every alternative, gate and selected W8/W4 route.
- `summary_speed.{json,md}`: final 18-cell speed and quant-memory verdict.
- `summary_active_work.{json,md}`: dense-only full-FLA and active-work gate.
- `summary_memory.{json,md}`: independent memory verdict.
- `quant_correctness.jsonl`: BNB8/MM4 same-quant native/HF probes.
- `environment.json`: exact hardware, software, model and source provenance.
- `*_exit_code.txt`, `matrix_failures.txt`, `progress.log`: fail-closed process
  status and retained rerun chronology.
- `selected_tests.log`, `rwkv_sha256.log`, `rwkv_hf_sha256.log`,
  `qwen_sha256.log`: focused tests and model integrity hashes.
- `SHA256SUMS`: integrity hashes for all retained evidence files.
