# RTX 3090 RWKV-7 g1h 7.2B vs Qwen3.5-9B bsz8 acceptance

Date: 2026-07-14

This artifact records an exact-card, resident-model speed and memory matrix on
one NVIDIA GeForce RTX 3090 24GB (`sm_86`). It compares the official RWKV-7
g1h 7.2B checkpoint with official Qwen3.5-9B. The conclusion is limited to the
declared `bsz=8` inference cells; it is not a model-quality comparison and does
not generalize to other batch sizes or cards.

## Scope

- RWKV checkpoint: `rwkv7-g1h-7.2b-20260710-ctx10240.pth`, converted through
  the repository low-memory HF converter. The downloaded checkpoint passed its
  ModelScope SHA-256 check.
- Qwen checkpoint: official `Qwen3.5-9B`; all four safetensor shards passed the
  ModelScope download checksum check.
- Shapes: prompt 128/512/2048, decode 128/512, batch 8.
- Precision families: dense fp16, W8 and W4; 18 joined cells total.
- Timing: one warmup and three measured runs per resident-model cell.
- RWKV route: native prefill graph plus `native_graph` cached decode.
- Qwen route: FLA Gated DeltaNet. All six dense reference cells verify all 24
  layers' FLA chunk-prefill, fused-recurrent decode and causal-conv1d bindings.
- Quant route composition: BNB8 is selected for W8 and native MM4 for W4 in all
  six shapes. Quantized RWKV is gated against matching RWKV fp16; quantized
  Qwen is deliberately not an acceptance dependency.

## Fail-closed gates

Dense RWKV must reach at least `1.00x` Qwen prefill and `1.50x` Qwen cached
decode throughput in every shape. Because RWKV and Qwen have different active
parameter counts, dense decode additionally requires
`(RWKV tok/s * active parameters) / (Qwen tok/s * active parameters) >= 1.00`.
The dense active-parameter ratio is `0.804032x`; this prevents a raw speed win
from hiding lower executed-parameter work on decode. Prefill keeps the direct
token-throughput gate and reports active work as telemetry.

W8/W4 must reduce both physical model footprint and peak allocated VRAM, and
must not be slower than matching RWKV fp16. The quant speed check accepts
either both phases individually or the explicitly enabled exact-cell
`prefill + decode` total-latency non-inferiority path. Phase metrics are always
retained. This matters for W4: its worst prefill cell is `0.988822x` fp16, but
its worst total-latency ratio is `1.014527x` and worst decode ratio is
`1.025666x`.

## Result

Overall: **PASS — 18/18 joined cells, zero red cells, zero missing rows, and
all pipeline/checker exit codes are zero.**

| Metric | Minimum | Median | Maximum | Passing cells |
|---|---:|---:|---:|---:|
| Prefill RWKV/Qwen, all families | `1.052379x` | `1.110550x` | `2.163321x` | 18/18 |
| Decode RWKV/Qwen, all families | `1.788418x` | `1.856048x` | `1.988157x` | 18/18 |
| Model footprint RWKV/Qwen | `0.444273x` | `0.781561x` | `0.804034x` | 18/18 no larger |
| Peak VRAM RWKV/Qwen | `0.504485x` | `0.857592x` | `0.995460x` | 18/18 no larger |
| Prefill tok/s per active-B | `1.316997x` | `1.388832x` | `2.690592x` | 18/18 |
| Decode tok/s per active-B | `2.224314x` | `2.397833x` | `2.472729x` | 18/18 |

Dense fp16 and quant-local detail:

| Family | Cells | RWKV/Qwen prefill min/median | RWKV/Qwen decode min/median | Prefill vs fp16 min | Decode vs fp16 min | Total vs fp16 min | Footprint vs fp16 max | Peak vs fp16 max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fp16 | 6 | `1.058907x / 1.063553x` | `1.788418x / 1.808213x` | — | — | — | — | — |
| W8 / BNB8 | 6 | `1.901037x / 2.148238x` | `1.941481x / 1.962023x` | `1.697094x` | `1.084305x` | `1.098658x` | `0.552555x` | `0.702992x` |
| W4 / native MM4 | 6 | `1.052379x / 1.062301x` | `1.838668x / 1.856048x` | `0.988822x` | `1.025666x` | `1.014527x` | `0.972049x` | `0.981434x` |

For dense fp16, decode active-parameter work rate is
`1.437946x–1.464674x` Qwen (median `1.453859x`) and passes in all 6/6 cells.
Dense prefill active work is `0.851395x–0.905145x` and is disclosed rather than
used as a gate. Runtime working set excluding model weights is also larger than
Qwen in all 18 cells (`1.842016x–2.239202x`), while total peak VRAM—the actual
fit constraint—is lower in all 18.

All performance rows report finite logits. The Qwen optimized operator
contract, route composition, memory/speed checks and 51 focused harness tests
pass. This artifact does not claim response-quality superiority or replace a
task-quality evaluation.

## Reproduce

After downloading and converting the official models:

```bash
PYTHON_BIN=/path/to/python \
BATCH_SIZES=8 \
  bench/run_3090_qwen35_pair_acceptance.sh \
  rwkv-7.2b__qwen3.5-9b \
  /path/to/rwkv7-g1h-7.2b-hf \
  /path/to/Qwen3.5-9B \
  /tmp/rwkv-g1h-7p2-qwen35-9b
```

The runner fails closed on missing rows, non-native RWKV routes, Qwen FLA
binding failure, dense speed, dense decode active work, quant speed and quant
memory. It records independent exit codes for route composition and all three
comparators.

## Artifacts

- `combined_auto.jsonl`: selected dense/W8/W4 candidate and reference rows.
- `dense.jsonl`, `memory.jsonl`, `native_speed.jsonl`,
  `hybrid_speed.jsonl`: retained route inputs.
- `route_manifest.json`: every alternative, check and selected W8/W4 route.
- `summary_speed.{json,md}`: final 18-cell speed and memory verdict.
- `summary_active_work.{json,md}`: dense-only full-FLA and active-work gate.
- `summary_memory.{json,md}`: independent memory verdict.
- `environment.json`: exact hardware, software, model and source provenance.
- `*_exit_code.txt`, `matrix_failures.txt`, `progress.log`: fail-closed process
  status.
- `selected_tests.log`, `rwkv_sha256.log`, `qwen_checksum.log`: focused tests
  and model download integrity checks.
- `SHA256SUMS`: integrity hashes for the retained evidence files.
