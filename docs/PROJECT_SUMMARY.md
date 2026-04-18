# RWKV-7 Hugging Face adapter project summary

Last updated: **2026-08-12**. The released baseline was audited at `main`
`045bac1b769240facd290e1ac8232e8b1ca39778`.

## Project purpose

This repository turns official RWKV-7 checkpoints into a standard Hugging Face
/ Transformers implementation while preserving RWKV's recurrent-state model
and providing optimized paths for training, quantization and heterogeneous
hardware. Native vLLM and SGLang runtimes remain separate projects; this
repository publishes the shared model, state, checkpoint and quantization
contracts they can reuse.

The published `v0.6.0` HF milestone is complete for its declared,
evidence-backed scope. Current main continues to add exact-card performance and
hardware profiles without redefining that release boundary.

## Delivered capability

| Area | Delivered scope | Canonical evidence |
|---|---|---|
| Checkpoint conversion | Official `.pth` to HF config, tokenizer metadata and safetensors; shape inference, batch conversion and low-memory large-model conversion | [`../scripts/convert_rwkv7_to_hf.py`](../scripts/convert_rwkv7_to_hf.py), [`ACCEPTANCE.md`](ACCEPTANCE.md) |
| Transformers API | AutoConfig/AutoTokenizer/AutoModelForCausalLM, generation, labels/loss, masks, save/reload and remote-code packaging | [`../tests/test_hf_api_contract.py`](../tests/test_hf_api_contract.py), [`INFERENCE_WORKFLOWS.md`](INFERENCE_WORKFLOWS.md) |
| Recurrent state | Generation cache plus select/reorder/repeat/drop/compact, offload/restore, cache handoff and chunked prefill | [`../rwkv7_hf/model_cache.py`](../rwkv7_hf/model_cache.py), [`INFERENCE_WORKFLOWS.md`](INFERENCE_WORKFLOWS.md) |
| HF training ecosystem | PEFT LoRA, Trainer, SFT, DPO, GRPO, gradient checkpointing and checkpoint resume | [`TRAINING.md`](TRAINING.md), [`TRAINING_WORKFLOWS.md`](TRAINING_WORKFLOWS.md) |
| Distributed execution | ZeRO-2/3 bounded smoke/resume; dense HF inference `device_map` PP and Transformers-native TP | [`ACCEPTANCE.md`](ACCEPTANCE.md), [`ADVANCED_USAGE.md`](ADVANCED_USAGE.md) |
| Native performance | FLA-free Native model, JIT/graph decode, fused recurrent/output and projection paths, DPLR/WY prefill and fail-closed per-product policy | [`PERFORMANCE.md`](PERFORMANCE.md), [`architecture/NATIVE_DEFAULT_BACKEND.md`](architecture/NATIVE_DEFAULT_BACKEND.md) |
| Quantization | bitsandbytes compatibility, native MM8/MM4, A8W8, TorchAO/Marlin and Apple MLX W8/W4 with profile-bounded speed/memory gates | [`QUANTIZATION.md`](QUANTIZATION.md), [`QUANTIZATION_USAGE.md`](QUANTIZATION_USAGE.md) |
| Hardware breadth | NVIDIA Pascal through Blackwell policies, AMD ROCm, Apple MPS/MLX/CoreML, Ascend, Biren, MetaX, MUSA and CPU fallback | [`HARDWARE_MATRIX.md`](HARDWARE_MATRIX.md) |
| Speculative decoding | Initial HF/Apple target-draft path with correctness gates | [`../rwkv7_hf/model_speculative.py`](../rwkv7_hf/model_speculative.py), [`ADVANCED_USAGE.md`](ADVANCED_USAGE.md) |

## Engineering architecture

The implementation is organized around four stable layers:

1. **HF public surface:** configuration, model, tokenizer, generation and cache
   contracts consumed by Transformers and converted checkpoints.
2. **Native RWKV-7 model:** FLA-free referenceable implementation used for
   inference and training ecosystem compatibility.
3. **Optimized operators and policy:** CUDA/Triton/ROCm/MLX or vendor-specific
   kernels selected by capability and exact-product evidence, with conservative
   fallbacks for every unmeasured shape.
4. **Evidence and regression:** focused tests, benchmark harnesses, immutable
   dated artifacts and canonical promoted summaries.

The stable remote-code surface and module ownership rules are documented in
[`architecture/REPOSITORY_LAYOUT.md`](architecture/REPOSITORY_LAYOUT.md). The
runtime-independent RWKV-7 equations and state transition are documented in
[`architecture/RWKV7_OPERATOR_SPEC.md`](architecture/RWKV7_OPERATOR_SPEC.md).

## Current-main performance additions

The latest merged series by `@123123213weqw` / Wang Yue adds:

- exact RTX 4080 and V100 B8 decode tuning
  ([PR #100](https://github.com/rwkv-rs/hf-adapter/pull/100));
- grouped RTX 4080/B8 W/A/V tensor-core projections for
  0.4B/1.5B/2.9B
  ([PR #101](https://github.com/rwkv-rs/hf-adapter/pull/101));
- an exact RTX 4080 7.2B/B8 FP16-state route reaching `344.39 tok/s`,
  `1.0301x` its FP32-state route, `-123.88 MiB` median peak allocation and
  greedy `12,288/12,288`
  ([PR #102](https://github.com/rwkv-rs/hf-adapter/pull/102),
  [`4080_7p2b_fp16_state_20260809`](../bench/4080_7p2b_fp16_state_20260809/README.md)).

The implementation, route telemetry, correctness checks, process-repeated
timing and raw artifacts landed together. See
[`RESULTS_INDEX.md`](RESULTS_INDEX.md) for the cross-platform evidence map.

The 2026-08-11 post-release expansion by `@yyqdbngt` adds exact RTX 5070
Laptop Native prefill/decode shape routing and exact Tesla V100 0.4B/1.5B B8
FP16 recurrent state. Both additions are fail-closed to measured product and
model shapes; rejected launch and fusion candidates remain disabled. Evidence
is recorded in [`5070_max_perf_20260811`](../bench/5070_max_perf_20260811/README.md)
and [`v100_exact_card_20260811`](../bench/v100_exact_card_20260811/README.md).

The 2026-08-12 RTX 3090 expansion by `@123123213weqw` closes all 24
latest-checkpoint B1/B8 parameter-adjusted prefill cells against official
full-FLA Qwen3.5 and adds a 15-row prompt/cache-handoff correctness oracle.
The row-32 scan and scoped FP16-accumulation schedules remain exact-card and
exact-shape only. Evidence is recorded in
[`3090_g1i_qwen35_prefill_pd_20260812`](../bench/3090_g1i_qwen35_prefill_pd_20260812/README.md).

## Production evidence discipline

A performance claim is promoted only for its named card, runtime, checkpoint,
dtype, batch and sequence shape. A complete artifact records:

- exact environment and command;
- baseline and candidate routes;
- warmup, repeat and synchronization policy;
- logits/state/greedy correctness;
- throughput and latency;
- physical footprint or peak memory when relevant;
- fallback isolation and regression results.

New cards and shapes extend the matrix. They do not retroactively turn a
completed HF API or accepted profile into an unfinished feature.

## Contribution provenance

Wang Yue (`@123123213weqw`, with repository aliases documented in
[`../CONTRIBUTORS.md`](../CONTRIBUTORS.md)) is the lead architect and primary
implementer. External work remains separately attributed: `@yyqdbngt`
contributed the Biren BR106M integration in PR #95 and is **not** an alias of
Wang Yue; `@KakaruHayate` contributed the MUSA integration in PR #87. Detailed
work-type and evidence links are in [`../CONTRIBUTIONS.md`](../CONTRIBUTIONS.md).

## Current boundaries and next work

Remaining work is expansion rather than a reopened HF milestone. The highest
value profiles currently include:

- RTX 4080 7.2B/B8 same-card Qwen/Albatross and full-model W8/W4;
- T4 full-model all-phase quantization and broader V100 full-memory prefill;
- additional H100, MI-series and Apple product evidence;
- larger/longer SFT, DPO, GRPO and distributed ZeRO-3 convergence matrices;
- broader task-quality and long-context evaluation.

The maintained list and precise boundaries are in
[`../HF_TODO.md`](../HF_TODO.md). Release history is in
[`../CHANGELOG.md`](../CHANGELOG.md).
