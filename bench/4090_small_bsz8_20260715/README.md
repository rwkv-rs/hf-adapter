# RTX 4090 small-model RWKV-7 vs Qwen3.5 bsz8 acceptance

Date: 2026-07-15

This artifact records the fail-closed dense/W8/W4 matrix for all currently
published RWKV-7 checkpoints below 7.2B on one NVIDIA GeForce RTX 4090 24GB
(`sm_89`).  It extends the separate 7.2B/9B artifact to the 0.4B, 1.5B and
2.9B model pairs.  The conclusion is limited to the declared inference cells;
it is not a response-quality comparison and does not generalize to other batch
sizes or Ada cards.

## Scope and gates

- Pairs: RWKV-7 0.4B vs Qwen3.5-0.8B, RWKV-7 1.5B vs Qwen3.5-2B, and
  RWKV-7 2.9B vs Qwen3.5-4B.
- Shapes: batch 8, prompt 128/512/2048, decode 128/512, shared 512-token
  prefill chunking.
- Precision families: dense fp16, W8 and W4; 54 joined cells total.
- Candidate route: repository native prefill graph plus `native_graph` cached
  decode. Every selected row reports finite logits.
- Reference route: Qwen FLA Gated DeltaNet. All 18 dense reference cells pass
  the full chunk-prefill, fused-recurrent decode, gated-norm and causal-conv1d
  operator contract; all 54 joined rows retain a verified FLA reference.
- Dense gate: RWKV/Qwen prefill `>=1.00x` in every cell. Cached decode must be
  `>=1.05x`, `>=1.65x` and `>=1.75x` for the 0.4B/0.8B, 1.5B/2B and 2.9B/4B
  pairs respectively. Dense decode also passes the active-parameter work gate.
- Quant memory gate: selected W8/W4 footprint and peak allocated VRAM must both
  be strictly lower than the matching RWKV fp16 row.
- Quant speed gate: both phases must be no slower than fp16, or the explicitly
  enabled exact-cell `(prefill + decode)` total latency must be no worse. Phase
  ratios remain visible and are never replaced by the aggregate number.

## Result

Overall: **PASS — 54/54 joined cells, 54/54 verified Qwen FLA references,
zero red cells, zero missing rows, and all comparator/composer exit codes are
zero.**

Dense cross-model and quant-local minima:

| Pair | Coverage | Dense prefill min vs Qwen | Dense decode min vs Qwen | Dense decode active work min | W8 total min vs fp16 | W4 total min vs fp16 |
|---|---:|---:|---:|---:|---:|---:|
| RWKV 0.4B / Qwen 0.8B | 18/18 | `1.370369x` | `12.101818x` | `7.250339x` | `1.011441x` | `1.029994x` |
| RWKV 1.5B / Qwen 2B | 18/18 | `1.041959x` | `5.636846x` | `4.575207x` | `1.131672x` | `1.027211x` |
| RWKV 2.9B / Qwen 4B | 18/18 | `1.305103x` | `4.214362x` | `2.953767x` | `1.176050x` | `1.014959x` |

Selected quant routes reduce physical memory in every cell:

| Pair | W8 prefill/decode min vs fp16 | W8 footprint/peak max | W4 prefill/decode min vs fp16 | W4 footprint/peak max |
|---|---:|---:|---:|---:|
| RWKV 0.4B | `0.999856x / 1.014214x` | `0.925797x / 0.963266x` | `0.999344x / 1.041423x` | `0.890672x / 0.945793x` |
| RWKV 1.5B | `1.293543x / 1.129083x` | `0.560704x / 0.625465x` | `0.930925x / 1.038061x` | `0.935468x / 0.968566x` |
| RWKV 2.9B | `1.270651x / 1.172780x` | `0.544714x / 0.509156x` | `0.986393x / 1.024407x` | `0.961227x / 0.977123x` |

The sub-1.00 W4 prefill ratios are disclosed rather than hidden. Decode gains
make every complete W4 cell faster than fp16 under the declared total-latency
rule. The 0.4B W8 prefill minimum is within `0.0144%` of fp16 and likewise
passes through a `1.011441x` worst-case total-latency result.

The selected W8 routes are native A8W8 for all six 0.4B shapes and the
BNB8+A8W8-head hybrid for all 1.5B/2.9B shapes. W4 selects native MM4 in 9
cells and TorchAO W4 in 9 cells. The manifest retains every rejected route and
its speed/memory checks.

## Promoted 1.5B prompt-512 policy

The initial 1.5B dense prompt-512 rows were just below the direct Qwen prefill
gate. An exact-card tile sweep selected recurrent-scan `block_m=32` instead of
the historical row-8 default. The policy is keyed by hidden size, batch and
prompt (`2048x8x512`) so the 7.2B prompt-512 route remains row-8.

The retained no-override probe reports:

```text
requested=null, effective=32, hidden=2048, batch=8, prompt=512
```

The final automatic-policy rows reach at least `1.041959x` Qwen dense prefill.
Focused policy, prefill, comparator, route-composer and benchmark-contract
tests pass `74/74`.

## Reproduce

After converting the official checkpoints, run each pair with the exact-card
entrypoint:

```bash
PYTHON_BIN=/path/to/python BATCH_SIZES=8 PREFILL_CHUNK_SIZE=512 \
  bench/run_4090_qwen35_pair_acceptance.sh \
  rwkv-1.5b__qwen3.5-2b \
  /path/to/rwkv7-g1g-1.5b-hf \
  /path/to/Qwen3.5-2B \
  /tmp/rwkv15-qwen2
```

Repeat with `rwkv-0.4b__qwen3.5-0.8b` and
`rwkv-2.9b__qwen3.5-4b`. The runner rejects non-4090 devices and fails closed
on coverage, native candidate routes, Qwen FLA bindings, dense speed, dense
decode active work, quant speed and quant-local physical memory.

## Artifacts

- `final_summary.json`: machine-readable combined verdict and per-family
  extrema.
- `pair_*/combined_auto.jsonl`: selected dense/W8/W4 candidate/reference rows.
- `pair_*/route_manifest.json`: alternatives, checks and exact-cell route
  decisions.
- `pair_*/summary_{speed,memory,active_work}.{json,md}`: independent
  fail-closed reports.
- `scan_policy_probe.jsonl`: no-override proof for the promoted 1.5B tile.
- `pair_1.5b_2b/dense_p512_tune.jsonl`: retained row-16/32/64 tile sweep.
- `system_metadata.txt`, `selected_tests.log`, `*_sha256.log`: exact stack,
  focused tests and model integrity records.
- `*_exit_code.txt`, `matrix_failures.txt`: retained pipeline status.
- `SHA256SUMS`: integrity hashes for every published evidence file.
