# Documentation map

The documentation is organized around a small canonical layer. Read that layer
first; detailed platform documents and dated benchmark artifacts provide depth
and history.

## Canonical documents

| Question | Canonical document |
|---|---|
| How do I install and run a model? | [`USER_GUIDE.md`](USER_GUIDE.md) / [`USER_GUIDE_ZH.md`](USER_GUIDE_ZH.md) |
| What is done now? | [`../HF_STATUS.md`](../HF_STATUS.md) |
| How should completion be reported? | [`../HF_STATUS.md#completion-reporting-rule`](../HF_STATUS.md#completion-reporting-rule) |
| What still needs work? | [`../HF_TODO.md`](../HF_TODO.md) |
| Do we meet the public HF requirements? | [`ACCEPTANCE.md`](ACCEPTANCE.md) |
| What are the current promoted numbers? | [`../BENCHMARK.md`](../BENCHMARK.md) |
| Which cards are validated? | [`HARDWARE_MATRIX.md`](HARDWARE_MATRIX.md) |
| What is the performance boundary? | [`PERFORMANCE.md`](PERFORMANCE.md) |
| What is the W8/W4 status? | [`QUANTIZATION.md`](QUANTIZATION.md) |
| Which training libraries and distributed paths work? | [`TRAINING.md`](TRAINING.md) |
| How do I contribute? | [`../CONTRIBUTING.md`](../CONTRIBUTING.md) |
| Where is raw evidence? | [`../bench/INDEX.md`](../bench/INDEX.md) |

## Source-of-truth order

If documents appear to conflict, use this order:

1. Raw dated artifact (`bench/<topic>_<hardware>_<date>/`), including JSONL/logs.
2. Current promoted numeric summary ([`../BENCHMARK.md`](../BENCHMARK.md)).
3. Canonical status/acceptance documents in the table above.
4. Platform detail and engineering-roadmap documents.
5. Historical prose and Git history.

A newer experiment does not automatically replace a promoted result. Promotion
requires compatible shape/reference, correctness and reproducible evidence.
Likewise, completion is reported for a named scope. The completed current
milestone must not be conflated with the still-partial universal production
scope, and roadmap checkbox counts must not be converted into a global
percentage.

## Document lifecycle

Use the title/date and the following classes when interpreting prose:

| Class | Meaning | May override current status? |
|---|---|---|
| Canonical | Root status/TODO/benchmark plus `ACCEPTANCE`, `HARDWARE_MATRIX`, `PERFORMANCE`, `QUANTIZATION`, `TRAINING` | Yes, subject to newer raw accepted evidence |
| Current engineering reference | Backend/runtime architecture and active kernel roadmaps | Only for implementation direction, not measured status |
| Dated validation snapshot | Exact-card validation documents and dated benchmark artifacts | Only for the exact recorded scope |
| Historical plan/investigation | `docs/plans`, `docs/archive`, dated live notes and superseded summaries | No; preserve for rationale and chronology |

Words such as “current”, “next” and “open” inside a historical document refer
to that document's date unless a banner explicitly promotes the statement.
Dated benchmark artifacts are evidence records and should not be rewritten to
match later outcomes.

## Platform details

| Platform | Detail document | Promoted summary |
|---|---|---|
| V100 | [`validation/V100_HF_VALIDATION.md`](validation/V100_HF_VALIDATION.md) | [`../bench/v100_production_close_20260711/README.md`](../bench/v100_production_close_20260711/README.md), [`../bench/v100_active_b1b8_20260715/README.md`](../bench/v100_active_b1b8_20260715/README.md) |
| RTX 4090 | [`../bench/4090_validation_summary.md`](../bench/4090_validation_summary.md) | [`../bench/4090_small_bsz8_20260715/README.md`](../bench/4090_small_bsz8_20260715/README.md), [`../bench/4090_g1h_7p2_bsz8_20260715/README.md`](../bench/4090_g1h_7p2_bsz8_20260715/README.md) |
| RTX 50 / Blackwell | [`hardware/BLACKWELL_50SERIES.md`](hardware/BLACKWELL_50SERIES.md) | [`../bench/5090_blackwell_production_close_20260712/README.md`](../bench/5090_blackwell_production_close_20260712/README.md) |
| A100 | [`validation/A100_HF_VALIDATION.md`](validation/A100_HF_VALIDATION.md) | [`HARDWARE_MATRIX.md`](HARDWARE_MATRIX.md) |
| A800 | [`validation/A800_HF_VALIDATION.md`](validation/A800_HF_VALIDATION.md) | [`HARDWARE_MATRIX.md`](HARDWARE_MATRIX.md) |
| Apple Silicon | [`hardware/APPLE_SILICON.md`](hardware/APPLE_SILICON.md) | [`hardware/APPLE_PRODUCTION_CLOSE.md`](hardware/APPLE_PRODUCTION_CLOSE.md) |
| Apple/Qwen methodology | [`hardware/QWEN35_APPLE_BASELINE.md`](hardware/QWEN35_APPLE_BASELINE.md) | [`hardware/APPLE_PRODUCTION_CLOSE.md`](hardware/APPLE_PRODUCTION_CLOSE.md) |

Platform detail files contain experiment chronology and may include superseded
or negative rows. Their promoted conclusion must agree with the canonical
matrix and benchmark summary.

## Engineering references

| Document | Purpose |
|---|---|
| [`BACKENDS.md`](BACKENDS.md) | Backend boundaries and rules for hardware-specific dispatch |
| [`performance/FUSED_BACKEND.md`](performance/FUSED_BACKEND.md) | Fused fp16/quant kernel roadmap and target ladder |
| [`performance/BN_TN_TUNING.md`](performance/BN_TN_TUNING.md) | Explicit CUDA block-N/thread-N terminology, sweep and promotion contract |
| [`native_fused_roadmap.md`](native_fused_roadmap.md) | Native kernel/layout/DPLR architecture notes |
| [`reference/HF_CRITERIA.md`](reference/HF_CRITERIA.md) | Low-level acceptance criteria reference |
| [`reference/MLX_RUNTIME_ARCHITECTURE.md`](reference/MLX_RUNTIME_ARCHITECTURE.md) | MLX runtime module and session boundaries |
| [`validation/math500_acceptance.md`](validation/math500_acceptance.md) | MATH500 runner and gate methodology |
| [`validation/math500_accuracy_parity.md`](validation/math500_accuracy_parity.md) | Accuracy/RNG/logit parity investigations |
| [`DOCUMENT_AUDIT_20260715.md`](DOCUMENT_AUDIT_20260715.md) | Full Markdown freshness sweep, corrected ambiguities and lifecycle rules |
| [`archive/NEXT_STEPS.md`](archive/NEXT_STEPS.md) | Historical plan only; not current TODO |

## Benchmark evidence workflow

1. Create `bench/<topic>_<hardware>_<YYYYMMDD>/`.
2. Include a concise README, exact command, environment, raw JSONL and logs.
3. Run correctness, speed and memory gates together.
4. Add the artifact to [`../bench/INDEX.md`](../bench/INDEX.md).
5. Promote only current conclusions to [`../BENCHMARK.md`](../BENCHMARK.md).
6. Update status/TODO only when the accepted state actually changes.

See [`../bench/README.md`](../bench/README.md) for the complete artifact rules.

## Scope boundaries

This repository delivers the Hugging Face adapter. HF cache helpers and
serving-like tests are in scope; native vLLM/SGLang scheduling and engine-level
PP/TP implementations are separate projects. Apple MLX/CoreML is retained as a
hardware backend lane because it validates the same converted model and HF
compatibility contract.
