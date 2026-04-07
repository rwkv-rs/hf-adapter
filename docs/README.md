# Documentation map

The documentation is organized around a small canonical layer. Read that layer
first; detailed platform documents and dated benchmark artifacts provide depth
and history.

## Canonical documents

| Question | Canonical document |
|---|---|
| How do I install and run a model? | [`USER_GUIDE.md`](USER_GUIDE.md) / [`USER_GUIDE_ZH.md`](USER_GUIDE_ZH.md) |
| How do I run inference and a tiny training demo on Windows/CPU? | [`WINDOWS_CPU.md`](WINDOWS_CPU.md) |
| Where is the tutorial for every implemented adaptation? | [`COMPLETE_ADAPTER_GUIDE.md`](COMPLETE_ADAPTER_GUIDE.md)（单一中文主索引） |
| Can an AI assistant execute and verify any documented workflow? | [`AI_ASSISTED_SETUP.md`](AI_ASSISTED_SETUP.md)（唯一 AI 操作入口） |
| How do I use speculative decoding, training, or multiple GPUs? | [`ADVANCED_USAGE.md`](ADVANCED_USAGE.md) / [`ADVANCED_USAGE_ZH.md`](ADVANCED_USAGE_ZH.md) |
| How do I use conversion, native/no-FLA, cache, batching, and chunked prefill? | [`INFERENCE_WORKFLOWS.md`](INFERENCE_WORKFLOWS.md) |
| How do I use PEFT, Trainer, resume, SFT, DPO, or GRPO? | [`TRAINING_WORKFLOWS.md`](TRAINING_WORKFLOWS.md) |
| How do I align CUDA training math and effect with official RWKV-LM train_temp? | [`TRAIN_TEMP_CUDA.md`](TRAIN_TEMP_CUDA.md) |
| How do I run the official RWKV-Gradio-3 UI with Native HF? | [`GRADIO_NATIVE_HF.md`](GRADIO_NATIVE_HF.md) |
| How do I run W8/W4 instead of only reading quantization status? | [`QUANTIZATION_USAGE.md`](QUANTIZATION_USAGE.md) |
| How do I use the promoted RTX 5090 g1h BN/TN W4 route? | [`QUANTIZATION_USAGE.md#5-rtx-5090-g1h-bntn-tensor-core-w4`](QUANTIZATION_USAGE.md#5-rtx-5090-g1h-bntn-tensor-core-w4) |
| How do I reproduce the RTX 4080 B1/B8 dense/quant acceptance? | [`QUANTIZATION_USAGE.md#6-rtx-4080-b1b8-配对验收`](QUANTIZATION_USAGE.md#6-rtx-4080-b1b8-配对验收) |
| How do I run Apple MPS, MLX, sessions, quant, or CoreML? | [`APPLE_USAGE.md`](APPLE_USAGE.md) |
| What is the one-page project and architecture summary? | [`PROJECT_SUMMARY.md`](PROJECT_SUMMARY.md) |
| What is done now? | [`../HF_STATUS.md`](../HF_STATUS.md) |
| How should completion be reported? | [`../HF_STATUS.md#completion-reporting-rule`](../HF_STATUS.md#completion-reporting-rule) |
| What can be extended after the release? | [`../HF_TODO.md`](../HF_TODO.md) |
| Do we meet the public HF requirements? | [`ACCEPTANCE.md`](ACCEPTANCE.md) |
| What are the current promoted numbers? | [`../BENCHMARK.md`](../BENCHMARK.md) |
| Where is the cross-platform evidence index? | [`RESULTS_INDEX.md`](RESULTS_INDEX.md) |
| Which cards are validated? | [`HARDWARE_MATRIX.md`](HARDWARE_MATRIX.md) |
| Are RTX 4080 defaults isolated from RTX 4090/5090? | [`validation/RTX4080_CROSS_CARD_AUDIT.md`](validation/RTX4080_CROSS_CARD_AUDIT.md) |
| What is the performance boundary? | [`PERFORMANCE.md`](PERFORMANCE.md) |
| What is the W8/W4 status? | [`QUANTIZATION.md`](QUANTIZATION.md) |
| Which training libraries and distributed paths work? | [`TRAINING.md`](TRAINING.md) |
| How do I contribute? | [`../CONTRIBUTING.md`](../CONTRIBUTING.md) |
| Where is raw evidence? | [`../bench/INDEX.md`](../bench/INDEX.md) |
| What changed between releases and current main? | [`../CHANGELOG.md`](../CHANGELOG.md) |
| How can a separate vLLM/SGLang project reuse the model, state, and quantization contracts? | [`integrations/README.md`](integrations/README.md) |

## Source-of-truth order

If documents appear to conflict, use this order:

1. Raw dated artifact (`bench/<topic>_<hardware>_<date>/`), including JSONL/logs.
2. Current promoted numeric summary ([`../BENCHMARK.md`](../BENCHMARK.md)).
3. Canonical status/acceptance documents in the table above.
4. Platform detail and engineering-roadmap documents.
5. Historical prose and Git history.

A newer experiment does not automatically replace a promoted result. Promotion
requires compatible shape/reference, correctness and reproducible evidence.
Likewise, completion is reported for a named scope. The completed `v0.6.0`
milestone is not reopened by an unbounded future-card or future-shape matrix,
and post-release project counts must not be converted into a global percentage.

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
| Windows / CPU | [`WINDOWS_CPU.md`](WINDOWS_CPU.md) | Download-free Native HF interface/update/save-reload smoke; not a quality or speed claim |
| V100 | [`validation/V100_HF_VALIDATION.md`](validation/V100_HF_VALIDATION.md) | [`../bench/v100_production_close_20260711/README.md`](../bench/v100_production_close_20260711/README.md), [`../bench/v100_active_b1b8_20260715/README.md`](../bench/v100_active_b1b8_20260715/README.md) |
| Tesla T4 | [`hardware/TURING_T4.md`](hardware/TURING_T4.md) | [`../bench/t4_production_close_20260720/README.md`](../bench/t4_production_close_20260720/README.md) (`Validated`, performance gaps retained) |
| RTX 4080 | [`HARDWARE_MATRIX.md`](HARDWARE_MATRIX.md) | [`../bench/4080_full_model_ladder_20260719/README.md`](../bench/4080_full_model_ladder_20260719/README.md), [`../bench/4080_b8_projection_bmm_20260809/README.md`](../bench/4080_b8_projection_bmm_20260809/README.md), [`../bench/4080_7p2b_fp16_state_20260809/README.md`](../bench/4080_7p2b_fp16_state_20260809/README.md) |
| RTX 4090 | [`../bench/4090_validation_summary.md`](../bench/4090_validation_summary.md) | [`../bench/4090_small_bsz8_20260715/README.md`](../bench/4090_small_bsz8_20260715/README.md), [`../bench/4090_g1h_7p2_bsz8_20260715/README.md`](../bench/4090_g1h_7p2_bsz8_20260715/README.md) |
| RTX 50 / Blackwell | [`hardware/BLACKWELL_50SERIES.md`](hardware/BLACKWELL_50SERIES.md) | [`../bench/5090_bntn_all_models_20260716/README.md`](../bench/5090_bntn_all_models_20260716/README.md), [`../bench/5090_native_official_fp16_production_20260718/README.md`](../bench/5090_native_official_fp16_production_20260718/README.md), [`../bench/5090_native_train_temp_real_minipile_20260718/README.md`](../bench/5090_native_train_temp_real_minipile_20260718/README.md), [`../bench/5090_native_hf_gradio_train_temp_20260718/README.md`](../bench/5090_native_hf_gradio_train_temp_20260718/README.md) |
| A100 | [`validation/A100_HF_VALIDATION.md`](validation/A100_HF_VALIDATION.md) | [`HARDWARE_MATRIX.md`](HARDWARE_MATRIX.md) |
| A800 | [`validation/A800_HF_VALIDATION.md`](validation/A800_HF_VALIDATION.md) | [`HARDWARE_MATRIX.md`](HARDWARE_MATRIX.md) |
| AMD gfx1100 | [`validation/AMD_ROCM_HF_VALIDATION.md`](validation/AMD_ROCM_HF_VALIDATION.md) | [`../bench/amd_gfx1100_native_20260727/README.md`](../bench/amd_gfx1100_native_20260727/README.md) |
| Moore Threads MUSA | [`hardware/MUSA.md`](hardware/MUSA.md) | Accepted exact-card legacy S70 scope with paired WKV/shift-mix evidence; later MUSA products require independent profiles |
| Huawei Ascend NPU | [`hardware/HUAWEI_ASCEND.md`](hardware/HUAWEI_ASCEND.md) | Accepted integrated compatibility scope pinned to standalone 910B3 evidence; future-main and broader profiles extend the matrix |
| Biren BR106M | [`hardware/BIREN_BR106M.md`](hardware/BIREN_BR106M.md) | Accepted BF16/FP32-state compatibility scope pinned to standalone evidence; performance, quant and multi-card profiles are extensions |
| MetaX C500 | [`hardware/METAX_C500.md`](hardware/METAX_C500.md) | Accepted native-eager compatibility scope pinned to C500 evidence; broader models, performance and quant profiles are extensions |
| Apple Silicon | [`hardware/APPLE_SILICON.md`](hardware/APPLE_SILICON.md) | [`hardware/APPLE_PRODUCTION_CLOSE.md`](hardware/APPLE_PRODUCTION_CLOSE.md) |
| Apple/Qwen methodology | [`hardware/QWEN35_APPLE_BASELINE.md`](hardware/QWEN35_APPLE_BASELINE.md) | [`hardware/APPLE_PRODUCTION_CLOSE.md`](hardware/APPLE_PRODUCTION_CLOSE.md) |
| Apple strict global audit | [`hardware/APPLE_PRODUCTION_ACCEPTANCE.md`](hardware/APPLE_PRODUCTION_ACCEPTANCE.md) | Historical `2026-07-13.2` 149-gate snapshot; it does not override the bounded current close |
| Apple/Qwen dated investigation | [`hardware/APPLE_QWEN35_LIVE_EVIDENCE_20260707.md`](hardware/APPLE_QWEN35_LIVE_EVIDENCE_20260707.md) | Historical 2026-07-07 blocker record; use the production-close page for current conclusions |

Platform detail files contain experiment chronology and may include superseded
or negative rows. Their promoted conclusion must agree with the canonical
matrix and benchmark summary.

## Engineering references

| Document | Purpose |
|---|---|
| [`architecture/REPOSITORY_LAYOUT.md`](architecture/REPOSITORY_LAYOUT.md) | Current ownership boundaries, stable remote-code surface, and the staged source-layout migration plan |
| [`architecture/NATIVE_DEFAULT_BACKEND.md`](architecture/NATIVE_DEFAULT_BACKEND.md) | Canonical Native/no-FLA backend decision, accepted migration gates, and retained FLA reference boundary |
| [`BACKENDS.md`](BACKENDS.md) | Backend boundaries and rules for hardware-specific dispatch |
| [`performance/FUSED_BACKEND.md`](performance/FUSED_BACKEND.md) | Fused fp16/quant kernel roadmap and target ladder |
| [`performance/BN_TN_TUNING.md`](performance/BN_TN_TUNING.md) | Scalar negative evidence, promoted RTX 5090 exact-model Tensor Core BN/TN matrix, and V100 packed-MM4 decode profiles |
| [`native_fused_roadmap.md`](native_fused_roadmap.md) | Native kernel/layout/DPLR architecture notes |
| [`reference/HF_CRITERIA.md`](reference/HF_CRITERIA.md) | Low-level acceptance criteria reference |
| [`reference/MLX_RUNTIME_ARCHITECTURE.md`](reference/MLX_RUNTIME_ARCHITECTURE.md) | MLX runtime module and session boundaries |
| [`architecture/RWKV7_OPERATOR_SPEC.md`](architecture/RWKV7_OPERATOR_SPEC.md) | Runtime-independent RWKV-7 token-step, state, and chunk-transition contract |
| [`integrations/VLLM_PORTING_GUIDE.md`](integrations/VLLM_PORTING_GUIDE.md) | Implementation guide for a separate vLLM/SGLang integration |
| [`integrations/RWKV7_STATE_CACHE_ABI.md`](integrations/RWKV7_STATE_CACHE_ABI.md) | Request-state pool, dynamic batching, prefix reuse, and chunked-prefill ABI |
| [`quantization/VLLM_QUANTIZATION_PORTING.md`](quantization/VLLM_QUANTIZATION_PORTING.md) | W8/W4 packed formats, kernel ABI, and serving-engine dispatch |
| [`integrations/VLLM_CHECKPOINT_MAPPING.md`](integrations/VLLM_CHECKPOINT_MAPPING.md) | Config, safetensors names, conversion, and TP/PP loading map |
| [`validation/VLLM_ACCEPTANCE.md`](validation/VLLM_ACCEPTANCE.md) | Production acceptance matrix for a future serving-engine implementation |
| [`reference/rwkv7_serving_contract.yaml`](reference/rwkv7_serving_contract.yaml) | Machine-readable serving tensor/operation/quantization contract |
| [`validation/math500_acceptance.md`](validation/math500_acceptance.md) | MATH500 runner and gate methodology |
| [`validation/math500_accuracy_parity.md`](validation/math500_accuracy_parity.md) | Accuracy/RNG/logit parity investigations |
| [`contributing/APPLE_VALIDATION.md`](contributing/APPLE_VALIDATION.md) | Full Apple contributor validation catalog split from the root contribution guide |
| [`DOCUMENT_AUDIT_20260715.md`](DOCUMENT_AUDIT_20260715.md) | Full Markdown freshness sweep, corrected ambiguities and lifecycle rules |
| [`archive/AGENTS_MILESTONES_202607.md`](archive/AGENTS_MILESTONES_202607.md) | Historical agent milestones preserved outside the active root instructions |
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
