# RTX 3090 native fused + quant evidence

Exact-card evidence for RWKV-7 7.2B on an NVIDIA GeForce RTX 3090. The RWKV
candidate uses the repository `native_graph` / native prefill path; FLA is not
the RWKV production backend. Qwen3.5 uses its verified FLA Gated DeltaNet path
with a separately recorded causal-convolution contract where a dense
cross-model reference is required.

## Acceptance policy

The current production policy is deliberately asymmetric:

1. **Dense:** compare RWKV dense fp16 with Qwen3.5 dense fp16 at `bsz=8`.
   Account for active parameter count rather than accepting a universal
   `1.05x` margin. The 7.2B/9B pair target is `1.50x`; the 1.5B/2B and
   2.9B/4B pair targets are `1.65x` and `1.75x` respectively.
2. **Quant:** compare RWKV W8/W4 only with the same RWKV dense fp16 row.
   Matching quantized Qwen performance is optional context, not an acceptance
   dependency. The default requires prefill/decode `>=1.00x`; the explicit
   total-latency option accepts a row when `dense total latency / quant total
   latency >=1.00x`, while still reporting both phase ratios. Lower model
   footprint and peak VRAM, finite logits, greedy alignment and the quality
   gate remain mandatory.
3. All promoted performance rows use medians after warmup and retain the raw
   backend/memory telemetry. A fastest-only observation is not sufficient.

## Measured 7.2B/9B checkpoint

The retained 72-cell matrix predates the bsz8-only simplification and remains
useful as broader evidence: it passes `72/72` with zero red or missing rows at
the former dense `1.05x` gate. Under the new bsz8 parameter-normalized policy:

| Lane, bsz8 | Minimum prefill ratio | Minimum decode ratio | Result |
|---|---:|---:|---|
| dense RWKV / dense Qwen | `1.0537x` | `1.8924x` | decode PASS; prefill remains open vs `1.50x` |
| RWKV W8 / RWKV dense | `1.7970x` | `1.0900x` | PASS |
| RWKV W4 / RWKV dense | `1.0018x` | `1.0179x` | PASS |

The bsz8 quant routes also reduce physical memory:

| Lane | Maximum footprint ratio vs dense | Maximum peak ratio vs dense |
|---|---:|---:|
| W8 | `0.5526x` | `0.7049x` |
| W4 | `0.9727x` | `0.9896x` |

The remaining dense prefill gap is intentionally not hidden: `1.0537x` needs
about `42.4%` more throughput to reach the pair's recommended `1.50x` target.

## Correctness and quality

- BnB W8 external-token compression ratio: `1.001475` over 950 tokens.
- BnB W4 external-token compression ratio: `1.001564` over 950 tokens.
- TorchAO W4 head compression ratio: `1.004745` over 950 tokens.
- Quality gate: ratio `<=1.01`, at least 900 externally supplied tokens.
- TorchAO W4 native/HF graph checks pass for bsz2/8; minimum decode cosine is
  `0.99999797`, and prefill/decode greedy tokens match.
- Touched CUDA/unit suite: `72 passed`.

## Files

- `final_72cell_summary.{json,md}`: fail-closed broad matrix summary.
- `w8_production_default.jsonl`: no-environment-override W8 production row.
- `quant_production_correctness.jsonl`: W8/W4 correctness evidence.
- `quality_gate_summary.{json,md}` and `bnb_quality_72_*`: BnB quality gate.
- `torchao_w4_quality.{json,md}`: fp16-head TorchAO W4 quality gate.
- `torchao_w4_correctness.jsonl`: TorchAO W4 graph/HF alignment.
- `system_metadata.txt`: exact software, GPU and model metadata.
- `selected_tests.log`: selected test-suite result.

Reproduce a bsz8 pair with:

```bash
PYTHON_BIN=/path/to/python \
  bench/run_3090_qwen35_pair_acceptance.sh \
  rwkv-7.2b__qwen3.5-9b \
  /path/to/rwkv7-g1g-7.2b-hf \
  /path/to/Qwen3.5-9B \
  /tmp/rwkv7-3090-72-vs-9
```

`DENSE_PREFILL_GATE` and `DENSE_DECODE_GATE` can override the pair-specific
dense gates for an exploratory run; the legacy `DENSE_SPEEDUP_GATE` aliases
the decode override. Quantized Qwen checkpoints are not loaded by this
command. The completed 1.5B/2B and 2.9B/4B bsz8 evidence is in
[`../3090_small_bsz8_20260714/README.md`](../3090_small_bsz8_20260714/README.md).
