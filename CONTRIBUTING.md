# Contributing to the RWKV-7 HF Adapter

Thanks for helping with the RWKV-7 Hugging Face / Transformers adapter. This
repository is focused on the **HF adapter track**: loading, conversion,
generation, PEFT, Trainer, TRL, DeepSpeed, HF state-cache helpers, quantized HF
inference, hardware/card validation, and production-readiness evidence.

vLLM, SGLang, DFlash, and standalone serving-engine integrations are separate
projects. Do not mix them into HF adapter PRs unless an issue explicitly asks
for shared helper code or documentation.

For shared serving-engine contracts, start from
[`docs/integrations/README.md`](docs/integrations/README.md). Documentation in
that directory may specify model mathematics, state-cache semantics,
checkpoint mapping, quantization layouts, and acceptance gates, but must not
add a vLLM/SGLang runtime dependency or report an unimplemented integration as
complete.

## Start here

1. Read [`HF_STATUS.md`](HF_STATUS.md) to understand what is already done.
2. Read [`HF_TODO.md`](HF_TODO.md) to pick a current task.
3. For performance or hardware work, read [`BENCHMARK.md`](BENCHMARK.md).
4. For the current V100 training/quant/ZeRO evidence, read [`docs/validation/V100_HF_VALIDATION.md`](docs/validation/V100_HF_VALIDATION.md).
5. For kernel/performance experiments, also read [`docs/performance/FUSED_BACKEND.md`](docs/performance/FUSED_BACKEND.md).
6. For backend or hardware work, read [`docs/BACKENDS.md`](docs/BACKENDS.md).
7. For Apple Silicon work, read [`docs/hardware/APPLE_SILICON.md`](docs/hardware/APPLE_SILICON.md).
8. Pick an issue, comment that you are working on it, then open a focused PR.

## Current contribution map

Do not use old issue numbers as a roadmap. Check the repository's current open
issues and [`HF_TODO.md`](HF_TODO.md) before claiming work; several historical
card tasks are already complete on current main.

| Area | Useful next contribution | Already completed; do not duplicate |
|---|---|---|
| Full-model quant performance | Close still-unmeasured T4 all-phase, broader V100 prefill, or RTX 4080 7.2B W8/W4 cells with exact fp16 pairing | Existing V100 packed-MM4, T4 output-head, 4080 output-head, 4090/5090 and Apple promoted profiles |
| Same-card references | Add a missing model/batch/shape Albatross or optimized-Qwen comparison, especially RTX 4080 7.2B/B8 | V100, RTX 3090/4090/5090 and the published RTX 4080 small-model pairs |
| New hardware products | Add independent H100/Hopper, MI-series, additional RTX 50, Apple M1-M4, Jetson or DGX Spark evidence | Existing exact-product V100/T4/3090/4080/4090/5070/5090/M5 plus integrated Ascend/Biren/MetaX/MUSA boundaries |
| Training breadth | Add larger-model or longer SFT/DPO/GRPO, distributed convergence, or broader ZeRO-3 resume evidence | Trainer/TRL compatibility, selected ZeRO-2/3 resume and RTX 5090 exact/5,000-step train_temp lanes |
| Maintenance | Reduce module duplication, add clean-install GPU lanes, or publish conversion provenance without changing stable remote-code ABI | Current conversion, Auto*, cache, PP/TP and quantization contracts |

If no current issue covers the proposed work, open a focused issue containing
the exact card/runtime, model, batch/shape, baseline, known risk and definition
of done. Link the planned evidence directory before starting a large hardware
run.

## What a good contribution looks like

A good PR is small, reproducible, and tied to one acceptance gap.

Examples:

- Add H100 or MI-series benchmark rows and update `BENCHMARK.md`.
- Extend an existing ZeRO checkpoint-resume gate to a larger model or longer run.
- Add a one-click acceptance script.
- Fix a `generate()` / `attention_mask` / cache behavior bug with a regression
  test.
- Add AMD/CPU fallback coverage to the native/no-FLA path.
- Add 8-bit/4-bit quantized inference telemetry for a new card.

Avoid large PRs that mix unrelated tasks such as docs, kernels, training, and
serving changes at the same time.

## Backend boundary rule

Cards are validation rows, not code branches. Keep exact card/chip names in
docs, tests, scripts, benchmark JSONL, and `rwkv7_hf/kernel_policy.py`. Core
model code should branch on capabilities such as backend availability,
`device.type`, dtype support, graph-capture support, or the normalized policy
family. See [`docs/BACKENDS.md`](docs/BACKENDS.md) for the full contract.

## Local setup

Typical environment variables for GPU work:

```bash
export PYTHONNOUSERSITE=1
export RWKV_V7_ON=1
export TORCHDYNAMO_DISABLE=1
export PYTHONPATH=/path/to/flash-linear-attention:/path/to/rwkv7-hf-adapter:${PYTHONPATH:-}
```

For DeepSpeed smoke on machines without a full CUDA toolkit setup, some tests
also support:

```bash
export DS_IGNORE_CUDA_DETECTION=1
```

Use the project-specific environment and model paths from your issue or PR body.
Do not hardcode private local paths in committed scripts unless they are examples
with `/path/to/...` placeholders.

## Test markers and dependency compatibility

Pytest collection uses strict, centrally enforced markers:

```text
cpu  cuda  sm70  ada  blackwell  apple  musa  slow  model_required
```

Hardware/domain markers are additive because kernel modules commonly include
CPU fallback and policy tests. Use `cpu and not model_required` for the portable
offline lane, and add a hardware marker when selecting a card-specific subset:

```bash
python -m pytest -m "cpu and not model_required"
python -m pytest -m "cuda and sm70"
python -m pytest -m "apple and not model_required"
python -m pytest -m "musa and not model_required"
```

The supported ecosystem bounds are Transformers `>=5.12.1,<6`, PEFT
`>=0.19.1,<1`, and TRL `>=1.7,<2`. Reproduce the lower-bound clean install with:

```bash
RWKV7_CPU_ONLY=1 \
RWKV7_HF_COMPAT_LANE=minimum \
RWKV7_CONSTRAINTS_FILE=configs/ci/hf-minimum.txt \
scripts/run_clean_install_tests.sh compat
```

Omit `RWKV7_CONSTRAINTS_FILE` to test the newest resolver result inside the
supported major versions. The scheduled CI workflow runs both lanes.

## Minimal no-GPU checks

For docs, conversion, and API-contract changes that do not require a live GPU,
run the relevant subset:

```bash
python tests/test_convert_config.py
python tests/test_batch_convert_manifest.py
python tests/test_result_tools.py
python tests/test_sync_hf_adapter_code.py
git diff --check
```

If dependencies are missing, mention the skip reason in the PR body.

## Minimal GPU card validation

For a card-adaptation issue, prefer the one-click wrapper first:

```bash
MODEL=/path/to/rwkv7-g1d-0.1b-hf \
DEVICE=cuda DTYPE=fp16 \
RESULTS=bench/results.jsonl \
bash scripts/run_hardware_smoke.sh
```

If the wrapper fails or you need to bisect, run the underlying commands:

```bash
python tests/smoke_hf_generate.py \
  --model /path/to/rwkv7-g1d-0.1b-hf

python tests/test_hf_api_contract.py \
  --model /path/to/rwkv7-g1d-0.1b-hf \
  --device cuda \
  --dtype fp16

python tests/test_quantized_inference.py \
  --model /path/to/rwkv7-g1d-0.1b-hf \
  --device cuda \
  --quantization 8bit \
  --optional

python tests/test_quantized_inference.py \
  --model /path/to/rwkv7-g1d-0.1b-hf \
  --device cuda \
  --quantization 4bit \
  --optional
```

Then add speed rows:

```bash
python bench/bench_speed.py \
  --hf-dir /path/to/rwkv7-g1d-0.1b-hf \
  --backend hf \
  --dtype fp16 \
  --device cuda \
  --results bench/results.jsonl

python bench/bench_batch_sweep.py \
  --hf-dir /path/to/rwkv7-g1d-0.1b-hf \
  --dtype fp16 \
  --device cuda \
  --results bench/results.jsonl
```

For training-capable cards, add:

```bash
python tests/test_peft_lora.py \
  --model /path/to/rwkv7-g1d-0.1b-hf \
  --device cuda \
  --attn-mode fused_recurrent

python tests/test_hf_training_smoke.py \
  --model /path/to/rwkv7-g1d-0.1b-hf \
  --device cuda \
  --attn-mode fused_recurrent \
  --backend both \
  --results bench/results.jsonl

python tests/test_hf_rl_training_smoke.py \
  --model /path/to/rwkv7-g1d-0.1b-hf \
  --device cuda \
  --attn-mode fused_recurrent \
  --backend dpo \
  --results bench/results.jsonl
```

For multi-GPU cards/nodes, add ZeRO smoke through the wrapper:

```bash
NPROC_PER_NODE=2 ZERO_STAGE=both \
MODEL=/path/to/rwkv7-g1d-0.1b-hf \
RESULTS=bench/results.jsonl \
bash scripts/run_zero_training_smoke.sh
```

Equivalent raw command for debugging:

```bash
torchrun --standalone --nproc_per_node=2 tests/test_deepspeed_training_smoke.py \
  --model /path/to/rwkv7-g1d-0.1b-hf \
  --zero-stage both \
  --train-dtype fp32 \
  --max-steps 1 \
  --batch-size 1 \
  --gradient-accumulation-steps 1 \
  --max-length 32 \
  --results bench/results.jsonl
```


## Minimal Apple Silicon validation

Run the smallest native/MPS smoke first:

```bash
python -m pip install -e .
MODEL=/path/to/rwkv7-g1d-0.1b-hf \
DEVICE=auto DTYPE=fp32 \
RESULTS=bench/results_apple_silicon.jsonl \
bash scripts/run_apple_silicon_smoke.sh
```

The complete MPS, MLX, Metal, quantization, session, long-context, training,
and pressure-test command catalog is maintained in
[`docs/contributing/APPLE_VALIDATION.md`](docs/contributing/APPLE_VALIDATION.md).
Use [`docs/APPLE_USAGE.md`](docs/APPLE_USAGE.md) for ordinary usage and
[`docs/hardware/APPLE_SILICON.md`](docs/hardware/APPLE_SILICON.md) for current
validated support and limitations.

## Reporting hardware results

Every hardware/card PR should include this information in the PR body or in a
linked issue comment:

````markdown
## Environment

- GPU(s):
- Driver:
- CUDA or ROCm:
- OS:
- Python:
- PyTorch:
- Transformers:
- PEFT:
- TRL:
- DeepSpeed:
- flash-linear-attention:
- Model path / size:
- dtype:

## Commands

```bash
# paste exact commands
```

## Results

- Smoke status:
- Prefill tok/s:
- Decode tok/s:
- Peak VRAM / memory:
- Quantized footprint:
- Quantized speed:
- Training loss / trainable delta, if applicable:

## Known limits

- Unsupported dtype/backend:
- Compile or kernel issues:
- Fallback path used:
````

If a benchmark writes rows to `bench/results.jsonl`, commit only rows that are
relevant to the PR. Do not mix unrelated local experiments into the same results
change.

## Documentation updates

Update docs when the PR changes public behavior, card support, or known gaps.

Common docs to update:

- `HF_STATUS.md` — if a status changes from open/partial to done.
- `HF_TODO.md` — if a TODO is completed, split, or reprioritized.
- `BENCHMARK.md` — if you add benchmark or hardware rows.
- `README.md` — if contributor-facing entry points or quickstart commands change.
- `docs/performance/FUSED_BACKEND.md` — if you change fused/native performance routes.

Keep installation audiences separate:

- ordinary users start with the pinned PyPI release, public 0.1B model,
  `rwkv7-hf-doctor`, and `rwkv7-hf-smoke`;
- `pip install -e .` appears only in clearly labeled source-development,
  conversion, repository-test, or benchmark workflows;
- the six live Hugging Face model cards and their three thin entrypoint error
  messages must recommend the same current PyPI release as `README.md`;
- immutable conversion manifests, release tags, changelog entries, and retained
  benchmark evidence keep the runtime version that actually produced them.

When the public package version changes, update the English and Chinese README,
user guides, published-model pages, kernel pages, AI setup route, Collection
description, and live model cards together. Run `tests/test_document_freshness.py`
and `tests/test_markdown_links.py` before publishing.

## Pull request checklist

Before opening a PR:

- [ ] The PR is scoped to one issue or one clear gap.
- [ ] Tests or benchmark commands are listed in the PR body.
- [ ] Hardware/software versions are listed for GPU work.
- [ ] `bench/results.jsonl` rows, if changed, are relevant and reproducible.
- [ ] Docs are updated if support status changed.
- [ ] The PR does not start vLLM/SGLang work in this HF adapter repository.

## Issue completion checklist

A card issue can usually be closed when:

- the required smoke commands pass or skips are explicitly justified;
- benchmark rows are recorded where applicable;
- `BENCHMARK.md` or the PR body summarizes the card result;
- the issue is updated with the final supported dtype/backend/model range;
- known limitations and fallback paths are documented.
