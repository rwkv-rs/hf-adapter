# AGENTS.md

## Project Mission

This repository is now scoped to the **RWKV-7 Hugging Face / Transformers adapter only**.

Historical upstream context mentioned three independent tracks: Hugging Face,
vLLM, and SGLang. For this repository and active work, do **not** build or gate
native vLLM/SGLang integrations. Any vLLM/SGLang work is a separate future
project and must not block HF deliverables.

The active reward target is the HF/Transformers track: make RWKV-7 usable from
standard HF APIs with near-production correctness, performance, memory behavior,
training compatibility, quantized inference, and reproducible benchmarks.

## Active Goal: Finish the Current HF Adapter First

Current priority: finish the RWKV-7 Hugging Face / Transformers adapter with
the hardware and evidence available now. Do not wait for H100/4090/5090/A100
access before completing the current repository deliverables. V100 remains the
active development and regression baseline; newer GPUs are follow-up validation
targets once available.

The current delivery strategy is:

- Keep the HF wrapper as the production-facing compatibility layer for
  `AutoModelForCausalLM`, `generate`, PEFT, Trainer, TRL, state cache,
  dynamic batching, chunked prefill, quantization, speculative decoding, and
  benchmark gates.
- Keep optimizing wrapper performance through `RWKV7StateCache`,
  `native_jit`, `native_graph`, cache reuse, reduced launch count, and future
  fused/native quantized kernels.
- Keep `native_model` explicitly experimental. It is the long-term base for
  removing the mandatory FLA runtime, upstream Transformers work, AMD/CPU
  fallback, and future kernels. It must not be described as replacing the
  wrapper until it proves the same HF compatibility, batching, cache semantics,
  and benchmark coverage.
- Do not merge older native branches wholesale when they would remove current
  HF training, quantization, cache, benchmark, or telemetry work. Audit those
  branches and port only the useful implementation ideas.

Near-term completion, without waiting for extra GPUs:

1. Done: preserve V100 training telemetry for HF Trainer, TRL SFT, TRL DPO,
   and TRL GRPO in the benchmark/report pipeline.
2. Done: add Albatross A/B benchmark ingestion on the same checkpoint, V100,
   dtype, batch size, prompt length, decode length, and cache policy.
3. Current: finish W8/W4 reporting and gates so the repository clearly records
   both the memory-target bnb rows and the fastest passing hybrid variants.
4. Current: keep code/tests/docs green locally without CUDA, then merge only
   changes that preserve existing HF training, cache, quantization, benchmark,
   and telemetry behavior.
5. Current no-GPU task: finish executable DeepSpeed ZeRO-2/ZeRO-3 HF Trainer
   smoke harness, analyzer/report ingestion, docs, and local tests. Real pass
   rows can wait for live GPU/DeepSpeed access, but the repository should be
   ready to run them with one command.
6. Current performance phase: follow `FUSED_BACKEND.md` for the native fused
   fp16 -> native W8/W4 backend. The analyzer must track Albatross ratio
   ladders and quantized speed/footprint gates under `fused_backend_targets`.
7. Next when GPUs return: expand V100 evidence for large-model smoke,
   speed/precision sweeps, chunked prefill, dynamic batching, state-cache reuse,
   speculative decoding, and ZeRO-2/ZeRO-3 multi-GPU smoke.
8. Later validation: run the prepared benchmark matrix on H100/4090/5090/A100.
   These newer cards are validation targets, not blockers for current progress.

Current no-GPU work mode:

- Finish everything that does not require live CUDA access first: HF API
  compatibility code, analyzers, benchmark ingestion, result gates, docs, unit
  tests, and PR hygiene.
- Treat existing V100 evidence as the active baseline until GPUs return. Do not
  block merges on new H100/4090/5090/A100 numbers.
- Keep GPU-only work as explicit follow-up rows in `BENCHMARK.md` /
  `NEXT_STEPS.md`: fresh speed sweeps, large-model runs, fused W8/W4 kernels,
  ZeRO-2/3 multi-GPU validation, and cross-card validation.
- The immediate finish line for this repository is a clean HF adapter that can
  be reviewed, installed, tested, and benchmarked reproducibly; vLLM/SGLang and
  DFlash stay outside the current merge gate.
- Do not start vLLM/SGLang work in this repository while the HF adapter still
  has open local tasks. First finish the HF adapter evidence, gates, and docs.

## Target Acceptance Criteria

Use this HF-only checklist as the authoritative target for the active
deliverable:

1. Match or approach the current RWKV-LM and Albatross training/inference
   performance, speed, precision, and memory use through HF-compatible paths
   across common batch sizes.
2. HF adaptation must work with common Transformer-based PEFT, RL, and training
   libraries, including PEFT, TRL, SFT/DPO/GRPO-style workflows, Trainer-style
   loops, gradient accumulation, and real multi-batch training smoke tests.
3. HF serving helpers must expose RWKV recurrent state cache semantics, dynamic
   batch select/reorder/drop, chunked prefill, state-cache allocation/reuse, and
   cache-reuse metrics that can later be reused by serving integrations.
4. Hardware support should cover common professional and consumer GPUs:
   NVIDIA from Pascal onward where feasible, newer NVIDIA generations, and AMD
   GPUs. HF inference should keep a path toward PP/TP, and HF training should
   support DeepSpeed ZeRO-2 and ZeRO-3 where feasible.
5. Quantized inference must support common W8 and W4 modes, reduce memory
   accordingly, and be faster than W16 on common cards. Older cards may need
   dedicated optimization. Quality should get as close as possible to
   llama.cpp-style Q*_K_M levels.
6. Add initial HF-compatible speculative decoding support, such as using a
   smaller RWKV model as the draft model. DFlash, native vLLM/SGLang adapters,
   and deeper standalone serving-engine work stay as follow-up projects.

Benchmark comparisons must separate engine performance from model quality:

- Albatross is the high-performance RWKV inference-engine reference. Compare it
  against this repository on the same checkpoint, hardware, dtype, batch size,
  prompt length, decode length, and cache policy. Track prefill tok/s, decode
  tok/s, aggregate tok/s, latency percentiles, memory footprint, peak VRAM,
  state-cache reuse/hit rate, and dynamic-batch behavior.
- Qwen3.5 is the model-quality target. The overall model-level goal is to
  exceed comparable Qwen3.5 baselines on instruction quality, reasoning, math,
  code, multilingual/Chinese, long-context, and RL/PEFT training workflows. Do
  not treat an inference-engine speed win as proof of beating Qwen3.5 quality;
  require explicit evaluation rows and reproducible prompts/datasets.

The final implementation should approach the performance, speed, precision, and memory usage of the official RWKV-LM path and Albatross path across different batch sizes.

### HF Transformers Track

Required goals:

- Convert official RWKV-7 `.pth` checkpoints to Hugging Face format.
- Provide `RWKV7Config`, `RWKV7Model`, and `RWKV7ForCausalLM`.
- Provide RWKV tokenizer support.
- Support `AutoConfig.from_pretrained`, `AutoTokenizer.from_pretrained`, and
  `AutoModelForCausalLM.from_pretrained`.
- Support `generate(..., use_cache=True)` with RWKV recurrent state cache.
- Support HF-style recurrent-state utilities for serving-like usage:
  state-cache allocation/reuse, dynamic batch select/reorder/drop/compact,
  chunked prefill, offload/restore, and cache telemetry.
- Support PEFT LoRA workflows and common HF training / RL libraries, especially
  PEFT, Trainer, TRL `SFTTrainer`, `DPOTrainer`, and `GRPOTrainer`-style flows.
- Support DeepSpeed ZeRO-2/ZeRO-3 presets where feasible through HF training
  entrypoints.
- Support 8-bit and 4-bit HF inference paths that reduce memory, preserve
  quality as much as possible, and target speed no slower than W16 on common
  cards.
- Add initial HF-compatible speculative decoding support, such as a smaller
  RWKV draft model verified by a larger HF RWKV target model.
- Keep a migration path toward an upstreamable native Transformers
  implementation without a mandatory FLA runtime dependency.

### Hardware Support

Required goals:

- Support common professional GPUs.
- Support common consumer GPUs.
- Current development server has 2 x Tesla V100-PCIE-32GB.
- V100 is acceptable for smoke tests and development, but final performance work
  should also be validated on newer cards such as A100/H100/4090/5090 where
  available.
- AMD GPU support remains a compatibility target for the HF path, preferably via
  pure PyTorch/reference paths first and optional kernels later.

### Quantized Inference

Required goals:

- Support 8-bit inference.
- Support 4-bit inference.
- Quantization must reduce memory usage.
- Quantized speed should be no slower than fp16 as much as possible.
- V100 may not be ideal for final int4/int8 speed validation because it lacks
  newer tensor core features.

## Current State

Completed first-stage HF wrapper adaptation:

- Downloaded official RWKV-7 0.1B checkpoint.
- Verified official `rwkv` package can load and generate on V100.
- Converted 0.1B checkpoint to Hugging Face-style `model.safetensors`.
- Added remote-code wrappers for config/model/tokenizer.
- Verified `AutoTokenizer` loading.
- Verified `AutoModelForCausalLM` loading.
- Verified `generate(use_cache=True)`.
- Verified PEFT LoRA forward/loss/backward smoke test.
- Compared HF logits with official RWKV path:
  - top-5 token IDs match
  - fp16 cosine similarity around `0.999996`
  - max absolute difference around `0.047`

Current implementation uses FLA (`flash-linear-attention`) as backend. This is a first-stage wrapper, not yet the final native Transformers implementation.

Recent completed evidence:

- V100 training telemetry is recorded for HF Trainer, TRL SFT, TRL DPO, and TRL
  GRPO-style smoke paths.
- Albatross A/B ingestion exists and analyzer output reports HF-vs-Albatross
  prefill/decode ratios.
- W8/W4 quantization rows record both canonical memory-target bnb behavior and
  `decode_hot` hybrid variants. The hybrid variants improve decode over generic
  bnb on V100 while remaining below fp16/native-graph speed, so fused/native
  quantized projection kernels remain the main quantization performance gap.

## Important Paths

Local GitHub checkout:

```bash
/Users/wangyue/Documents/vllmsp/rwkv7-hf-adapter
```

Server project checkout:

```bash
/home/data/wangyue/projects/rwkv7-hf-adapter
```

Server model files:

```bash
/home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-20260129-ctx8192.pth
/home/data/wangyue/models/rwkv7/rwkv_vocab_v20230424.txt
/home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf
```

Server environments:

```bash
/home/data/wangyue/envs/rwkv7
/home/data/wangyue/envs/rwkv7-cu118
```

Reference repos on server:

```bash
/home/data/wangyue/projects/RWKV-LM
/home/data/wangyue/projects/Albatross
/home/data/wangyue/projects/flash-linear-attention
```

## Development Environment

Use this for the current HF wrapper work:

```bash
source /home/wzu/anaconda3/etc/profile.d/conda.sh
conda activate /home/data/wangyue/envs/rwkv7
export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/home/data/wangyue/projects/flash-linear-attention:/home/data/wangyue/projects/rwkv7-hf-adapter:$PYTHONPATH
```

For official RWKV / CUDA extension smoke tests:

```bash
source /home/wzu/anaconda3/etc/profile.d/conda.sh
conda activate /home/data/wangyue/envs/rwkv7-cu118
export RWKV_V7_ON=1
export CUDA_VISIBLE_DEVICES=0
```

## Common Commands

### Convert checkpoint to HF format

```bash
python /home/data/wangyue/projects/rwkv7-hf-adapter/scripts/convert_rwkv7_to_hf.py \
  --input /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-20260129-ctx8192.pth \
  --output /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf \
  --vocab-file /home/data/wangyue/models/rwkv7/rwkv_vocab_v20230424.txt \
  --precision fp16 \
  --attn-mode chunk
```

### HF generate smoke test

```bash
python /home/data/wangyue/projects/rwkv7-hf-adapter/tests/smoke_hf_generate.py \
  --model /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf
```

### PEFT LoRA smoke test

```bash
export TORCHDYNAMO_DISABLE=1
python /home/data/wangyue/projects/rwkv7-hf-adapter/tests/test_peft_lora.py \
  --model /home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf \
  --attn-mode fused_recurrent
```

## Engineering Rules

- Do not commit model weights, `.pth`, `.safetensors`, `.bin`, `.gguf`, checkpoints, or generated large artifacts.
- Keep conversion scripts reproducible.
- Keep smoke tests small and runnable on V100.
- Always compare against official RWKV-LM or `rwkv` package outputs when changing math or weight mapping.
- Treat FLA backend as a temporary first-stage dependency until native Transformers implementation is ready.
- Be explicit about state cache behavior: RWKV recurrent state is not Transformer KV cache.
- For PEFT/TRL compatibility, prefer standard HF model signatures and return types.
- For HF serving-style helpers, design state cache allocator/gather/scatter/reorder/release explicitly.

## Next Milestones

1. Convert and validate larger RWKV-7 checkpoints, including the 13.3B gate.
2. Keep official RWKV vs HF logits/generation alignment tests green.
3. Keep `save_pretrained` / reload roundtrip tests green.
4. Expand PEFT / Trainer / TRL SFT/DPO/GRPO smoke tests into multi-batch and gradient-accumulation checks.
5. Continue HF decode-performance work: native graph/JIT, dynamic-batch cache, chunked prefill, and cache telemetry.
6. Finish HF quantized W8/W4 inference so memory drops and speed is competitive with W16.
7. Validate on more GPUs and larger batch sizes.
8. Start native Transformers implementation under `src/transformers/models/rwkv7/` style layout.
9. Remove mandatory FLA dependency from the final HF implementation.
