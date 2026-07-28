# RWKV-7 Hugging Face Reference

[English](README.md) | [中文](README_ZH.md)

A readable, pure-PyTorch RWKV-7 implementation for Hugging Face Transformers.
Version 1.0 follows the clean Mamba-style source separation used by readable HF
state-space models: configuration, cache, operators, and modeling each have one
obvious owner. This is an organizational convention, not a change to RWKV-7
math. The full architecture stays visible in `modeling_rwkv7.py`, recurrent
math has one small boundary in `ops_rwkv7.py`, and each converted model is
self-contained.
The `rwkv7_hf` package contains model code only; conversion and smoke-test
commands live in the separate `rwkv7_hf_tools` package.
The optional `rwkv7-kernels` companion distribution contains the complete
NVIDIA performance implementation behind one stable API-v4 facade.
Installing it does not replace the readable model, config, cache, tokenizer or
checkpoint layout. Historical development remains archived on
`perf/native-kernels-v0.8`; users do not install code from that branch.

Publication status and immutable-artifact gates are tracked in
[`HF_STATUS.md`](HF_STATUS.md). Exact `==1.0.0` commands below apply once the
matching artifacts are listed there as published.

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

## Add the optional NVIDIA backend

After installing the PyTorch build for the target GPU:

```bash
python -m pip install "rwkv7-hf==1.0.0" "rwkv7-kernels==1.0.0"
```

The equivalent single requirement is
`python -m pip install "rwkv7-hf[kernels]==1.0.0"`. In both forms the model
API stays unchanged; uninstalling `rwkv7-kernels` restores package-free
reference execution.

No model-code change is required. The default `RWKV7_BACKEND=auto` uses only
device/dtype/shape routes accepted by the release matrix and otherwise runs the
same reference body. `RWKV7_BACKEND=reference` disables the plugin;
`RWKV7_BACKEND=optimized` is the strict diagnostic mode and raises instead of
hiding an unsupported route.

The companion wheel owns recurrent, fused prefill/decode, CUDA Graph/state
pools, SM70/Ada/Blackwell policies, quantization adapters and training
autograd. Its only public execution entry point is `execute_optional_v4`, with
five operation kinds: `training_program`, `model_forward`, `linear_training`,
`mix6_training`, and `recurrent`. A negative capability decision is
side-effect-free and returns a normalized result with `result=None`, which lets
`auto` take the unchanged reference path. `model_forward` passes the caller's
canonical cache to the wheel without cloning. Once positive execution starts,
an exception or malformed payload fails closed even in `auto`; it is never
recomputed through reference code after the cache may have been bound or
updated by a CUDA Graph.
Training keeps the same readable HF layer loop. Setting
`RWKV7_TRAINING_KERNEL_IMPL=adaptive` asks API v4 for one atomic certificate;
the currently certified dense B4/T128 program uses the factorized recurrent,
bounded FFN-linear, and explicit-shift Mix6 leaves. Every leaf revalidates its
concrete tensors against that certificate. Other explicitly adaptive requests
use the individually gated exact-matrix/reference fallbacks; PEFT requests
whose frozen embeddings cannot prove an autograd input select one complete
reference program. Strict `RWKV7_BACKEND=optimized` still requires the atomic
certificate and fails at the model boundary outside its domain.

The kernel package top level exports only `__version__`,
`RWKV7_KERNEL_API_VERSION`, and `execute_optional_v4`; historical v1 helpers
remain internal.
The API-v4 operation set and envelope are frozen for the 1.0 line. See the
[kernel plugin contract](docs/KERNEL_PLUGIN_API.md). Alternative backends plug
into that one entrypoint without replacing the HF model or canonical cache.
The exact 1.0.0 distribution inputs are SHA-256 locked by
[`RELEASE_SOURCE_FREEZE.json`](RELEASE_SOURCE_FREEZE.json); changing them
requires a new versioned freeze and the complete hardware matrix.
It never adds a hardware field to `RWKV7Config` or a private layout to
`RWKV7Cache`. Native W8/W4/A8W8, BN/TN, BitsAndBytes, Marlin and TorchAO remain
explicit quantization choices through `rwkv7_kernels.quantization`.

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
- one immutable `RWKV7ExecutionContext`, resolved once and passed explicitly
  through the readable non-linear layer boundaries and checkpoint replay; two
  narrow lexical routing bridges preserve decoder-to-LM-head context transfer
  and the standard `nn.Linear.forward(x)` contract, while two separate
  context-local snapshots record evidence only
- standard loss, cache, generation, save/reload, gradient checkpointing, PEFT
  and Trainer/TRL surfaces

The public API uses only the canonical `RWKV7*` class names.

## Source packages

- `rwkv7_hf/` contains only the HF configuration, cache, reference recurrence
  plus minimal API-v4 facade, modeling, tokenizer, and chat template.
- `rwkv7_hf_tools/` contains the CLI, checkpoint converter, manifest helpers,
  and public-model smoke test.
- `kernels/rwkv7_kernels/` contains only the optional versioned protocol,
  NVIDIA implementations, graph/state pools, quantizers and training ops.

## Reproduction

- [Architecture](docs/ARCHITECTURE.md)
- [Conversion](docs/CONVERSION.md)
- [Evaluation](docs/EVALUATION.md)
- [LoRA SFT, DPO, and GRPO](docs/FINETUNING.md)
- [Reproducibility artifacts](docs/REPRODUCIBILITY.md)
- [NVIDIA migration and capability audit](docs/NVIDIA_MIGRATION_AUDIT.md)
- [Published models](docs/PUBLISHED_MODELS.md)

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```
