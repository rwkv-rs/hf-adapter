# RTX 3090 bsz8 RWKV-7 vs Qwen3.5 acceptance

Exact-card speed and memory evidence for two small RWKV-7/Qwen3.5 pairs on an
NVIDIA GeForce RTX 3090 24GB. This artifact closes only the declared bsz8
matrix; it does not generalize the result to other batch sizes, model sizes,
hardware, or model quality.

## Scope and gate

- Pairs: RWKV-7 1.5B vs Qwen3.5 2B and RWKV-7 2.9B vs Qwen3.5 4B.
- Shapes: bsz8, prompt 128/512/2048, decode 128/512, fp16.
- Candidate: repository native prefill plus `native_graph` decode.
- Reference: Qwen FLA Gated DeltaNet path. Every row must satisfy the operator
  contract; `auto`/Transformers fallback is not accepted.
- Dense gate: RWKV/Qwen prefill `>=1.00x`; cached decode `>=1.65x` for 1.5B/2B
  and `>=1.75x` for 2.9B/4B.
- Quant memory gate: RWKV W8/W4 model footprint and peak VRAM must both be
  strictly below the matching RWKV fp16 row.
- Quant speed gate: either both prefill and decode are `>=1.00x` fp16, or the
  explicitly enabled exact-cell `(prefill + decode)` total latency is no worse
  than fp16. Per-phase ratios remain visible and are never replaced by the
  aggregate number.

The exact-cell total-latency option matters for W4: it is marginally slower in
prefill but faster in decode, and all measured full inference cells finish
faster than fp16.

## Results

Both pair gates pass with zero red or missing cells:

| Pair | Coverage | Qwen FLA | Dense prefill min | Dense decode min | W8 total min vs fp16 | W4 prefill min vs fp16 | W4 decode min vs fp16 | W4 total min vs fp16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| RWKV 1.5B / Qwen 2B | 18/18 | 18/18 | `1.0306x` | `3.3828x` | `1.1929x` | `0.9835x` | `1.0279x` | `1.0107x` |
| RWKV 2.9B / Qwen 4B | 18/18 | 18/18 | `1.3559x` | `2.9213x` | `1.1809x` | `0.9863x` | `1.0198x` | `1.0068x` |

Quantized physical-memory ratios are also below fp16 in every selected cell:

| Pair | W8 footprint max | W8 peak max | W4 footprint max | W4 peak max |
|---|---:|---:|---:|---:|
| RWKV 1.5B | `0.6046x` | `0.8168x` | `0.9342x` | `0.9695x` |
| RWKV 2.9B | `0.5731x` | `0.7603x` | `0.9612x` | `0.9760x` |

Overall: **PASS, 36/36 joined cells, 36/36 verified Qwen FLA references, zero red cells.**

## Reproduce

Convert the official checkpoints first, then run each pair:

```bash
PYTHON_BIN=/path/to/python \
BATCH_SIZES=8 \
  bench/run_3090_qwen35_pair_acceptance.sh \
  rwkv-1.5b__qwen3.5-2b \
  /path/to/rwkv7-g1g-1.5b-hf \
  /path/to/Qwen3.5-2B \
  /tmp/rwkv15-qwen2

PYTHON_BIN=/path/to/python \
BATCH_SIZES=8 \
  bench/run_3090_qwen35_pair_acceptance.sh \
  rwkv-2.9b__qwen3.5-4b \
  /path/to/rwkv7-g1g-2.9b-hf \
  /path/to/Qwen3.5-4B \
  /tmp/rwkv29-qwen4
```

The entrypoint forces Qwen `--qwen-backend fla`, fails closed on backend,
coverage, speed, memory, and prefill-mode mismatches, and records both phase and
total-latency ratios.

## Artifacts

- `final_summary.json`: compact machine-readable combined verdict.
- `pair_*/summary_speed.{json,md}`: per-pair fail-closed comparison.
- `pair_*/route_manifest.json`: all quant route decisions, metrics and checks.
- `pair_*/combined_auto.jsonl`: selected candidate/reference rows.
- `system_metadata.txt`: exact GPU and software stack.
- `selected_tests.log`: benchmark-harness regression result (`44 passed`).
- `SHA256SUMS`: integrity hashes for every retained artifact.
