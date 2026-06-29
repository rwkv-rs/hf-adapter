# RWKV-7 Hugging Face Reference

[English](README.md) | [中文](README_ZH.md)

A readable, pure-PyTorch RWKV-7 implementation for Hugging Face Transformers.
Version 0.9 makes compatibility and reproducibility the default: the model
architecture is visible in one `modeling_rwkv7.py`, recurrent math has one
small boundary in `ops_rwkv7.py`, and each converted model is self-contained.
CUDA/JIT/graph/quantization kernels remain on the long-lived
`perf/native-kernels-v0.8` branch and are not part of this reference line.

## Install and use a published model

```bash
python -m pip install "torch" "transformers>=4.48,<6"
```

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "wangyue114514/rwkv7-g1d-0.1b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    dtype=torch.float16,
).cuda().eval()

inputs = tokenizer("User: Hello! Assistant:", return_tensors="pt").to("cuda")
with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=32)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

The model repository contains its configuration, cache, PyTorch operator,
modeling code, tokenizer, vocabulary, and safetensors. Loading it does not
require `rwkv7-hf`, FLA, Triton, a compiler, or a kernel wheel.

## Convert an official checkpoint

```bash
python -m pip install "rwkv7-hf==0.9.0"
rwkv7-hf convert \
  --input /path/to/model.pth \
  --output ./rwkv7-model-hf \
  --vocab-file /path/to/rwkv_vocab_v20230424.txt \
  --precision fp16 \
  --adapter-layout reference \
  --no-fuse-norm \
  --low-memory
```

`reference` is the default. `thin` remains only as a legacy package-backed
layout for older deployment workflows.

## Public architecture

- `RWKV7Config` with `model_type = "rwkv7"`
- `RWKV7Cache`: canonical `[B,H,K,V]` state plus TMix/CMix shifts
- `RWKV7TimeMix`, `RWKV7ChannelMix`, `RWKV7Block`
- `RWKV7PreTrainedModel`, `RWKV7Model`, `RWKV7ForCausalLM`
- standard loss, cache, generation, save/reload, gradient checkpointing, PEFT
  and Trainer/TRL surfaces

Historical `NativeRWKV7*` class names are 0.9 compatibility aliases.

## Reproduction

- [Architecture](docs/ARCHITECTURE.md)
- [Conversion](docs/CONVERSION.md)
- [Evaluation](docs/EVALUATION.md)
- [LoRA SFT, DPO, and GRPO](docs/FINETUNING.md)
- [Reproducibility artifacts](docs/REPRODUCIBILITY.md)
- [Published models](docs/PUBLISHED_MODELS.md)

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```
