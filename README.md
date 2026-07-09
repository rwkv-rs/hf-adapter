# RWKV-7 Hugging Face Reference

[English](README.md) | [中文](README_ZH.md)

A readable, pure-PyTorch RWKV-7 implementation for Hugging Face Transformers.
Version 1.0 makes readability and reproducibility the default: the model
architecture is visible in one `modeling_rwkv7.py`, recurrent math has one
small boundary in `ops_rwkv7.py`, and each converted model is self-contained.
The `rwkv7_hf` package contains model code only; conversion and smoke-test
commands live in the separate `rwkv7_hf_tools` package.
Optional CUDA Graph and Triton work remains on
`perf/optional-native-backend-v0.10`; older CUDA/JIT/quantization and KV-v2
experiments remain archived on `perf/native-kernels-v0.8`. Neither performance
branch is part of this reference line.

## Install and use a published model

```bash
python -m pip install "torch" "transformers>=4.48,<6"
```

Install the PyTorch build that matches the GPU before installing the adapter.
In particular, current default CUDA 13 wheels may omit `sm_70`; V100 users
should select a compatible CUDA 12.x wheel from the official PyTorch index.
Once PyTorch is present, `pip install rwkv7-hf==1.0.0` keeps that installation.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "wangyue114514/rwkv7-g1d-0.1b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=torch.float16,
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
python -m pip install "torch"  # choose the wheel for your CUDA/GPU first
python -m pip install "rwkv7-hf==1.0.0"
rwkv7-hf convert \
  --input /path/to/model.pth \
  --output ./rwkv7-model-hf \
  --vocab-file /path/to/rwkv_vocab_v20230424.txt \
  --precision fp16 \
  --low-memory
```

The converter always writes the complete, self-contained reference layout.

## Public architecture

- `RWKV7Config` with `model_type = "rwkv7"`
- `RWKV7Cache`: canonical `[B,H,K,V]` state plus TMix/CMix shifts
- `RWKV7TimeMix`, `RWKV7ChannelMix`, `RWKV7Block`
- `RWKV7PreTrainedModel`, `RWKV7Model`, `RWKV7ForCausalLM`
- standard loss, cache, generation, save/reload, gradient checkpointing, PEFT
  and Trainer/TRL surfaces

The public API uses only the canonical `RWKV7*` class names.

## Source packages

- `rwkv7_hf/` contains only the HF configuration, cache, operator boundary,
  modeling, tokenizer, and chat template.
- `rwkv7_hf_tools/` contains the CLI, checkpoint converter, manifest helpers,
  and public-model smoke test.

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
