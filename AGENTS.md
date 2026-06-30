# AGENTS.md

## Project Mission

This repository is for the RWKV-7 ecosystem adaptation work.

The upstream task is to build production-quality adapters for RWKV-7:

1. Hugging Face Transformers adaptation
2. vLLM adaptation
3. SGLang adaptation

Original reward target: each track is worth 300,000 RMB if completed to near-production quality.

Current repository focus: **Hugging Face / Transformers adaptation first**.

## Target Acceptance Criteria

The final implementation should approach the performance, speed, precision, and memory usage of the official RWKV-LM path and Albatross path across different batch sizes.

### 1. HF Transformers Track

Required goals:

- Convert official RWKV-7 `.pth` checkpoints to Hugging Face format.
- Provide `RWKV7Config`.
- Provide `RWKV7Model`.
- Provide `RWKV7ForCausalLM`.
- Provide RWKV tokenizer support.
- Support `AutoConfig.from_pretrained`.
- Support `AutoTokenizer.from_pretrained`.
- Support `AutoModelForCausalLM.from_pretrained`.
- Support `generate()` with recurrent state cache.
- Support PEFT LoRA workflows.
- Support common HF training / RL libraries, especially PEFT, TRL, SFTTrainer-style flows.
- Eventually upstream or package as a native Transformers-style implementation.

### 2. vLLM Track

Required goals:

- Support RWKV-7 serving in vLLM.
- Implement recurrent state cache instead of assuming Transformer KV cache.
- Support dynamic batching.
- Support chunked prefill.
- Support request-level state cache allocation, gather/scatter, reorder, release, and reuse.
- Match or approach official RWKV-LM / Albatross throughput and memory behavior.

### 3. SGLang Track

Required goals:

- Support RWKV-7 serving in SGLang.
- Implement RWKV recurrent state cache integration.
- Support dynamic batching.
- Support chunked prefill.
- Support stateful decoding across requests.
- Preserve compatibility with SGLang serving APIs and scheduler behavior.

### 4. Hardware Support

Required goals:

- Support common professional GPUs.
- Support common consumer GPUs.
- Current development server has 2 x Tesla V100-PCIE-32GB.
- V100 is acceptable for smoke tests and development, but final performance work should also be validated on newer cards such as A100/H100/4090/5090 where available.

### 5. Quantized Inference

Required goals:

- Support 8-bit inference.
- Support 4-bit inference.
- Quantization must reduce memory usage.
- Quantized speed should be no slower than fp16 as much as possible.
- V100 may not be ideal for final int4/int8 speed validation because it lacks newer tensor core features.

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
/home/data/wangyue/projects/vllm
/home/data/wangyue/projects/sglang
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
- For serving frameworks, design state cache allocator/gather/scatter/reorder/release explicitly.

## Next Milestones

1. Convert and validate larger RWKV-7 checkpoints.
2. Add official RWKV vs HF automated logits comparison test.
3. Add `save_pretrained` / reload roundtrip test.
4. Add TRL `SFTTrainer` smoke test.
5. Start native Transformers implementation under `src/transformers/models/rwkv7/` style layout.
6. Remove mandatory FLA dependency from the final HF implementation.
7. Begin vLLM state-cache design document.
8. Begin SGLang state-cache design document.
