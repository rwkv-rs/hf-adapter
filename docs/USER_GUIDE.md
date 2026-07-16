# RWKV-7 HF Adapter User Guide

This guide is for users who want to run an official RWKV-7 checkpoint through
the standard Hugging Face `transformers` API. Benchmark development and kernel
tuning are not required for normal inference.

Chinese version: [`USER_GUIDE_ZH.md`](USER_GUIDE_ZH.md)

## Choose a path

- Follow this page step by step for a manual setup.
- Give [`AI_ASSISTED_SETUP.md`](AI_ASSISTED_SETUP.md) to a terminal-capable AI
  assistant if you want it to perform and verify the setup.
- After the first generation, use
  [`COMPLETE_ADAPTER_GUIDE.md`](COMPLETE_ADAPTER_GUIDE.md) to find the tutorial
  and PASS gate for every other implemented adaptation.
- If you already have a converted HF model directory, skip to
  [Run generation](#3-run-generation).

Setup is complete only when the environment doctor reports `RESULT: READY`,
the model-directory check passes, and `examples/generate.py` exits with code 0
and prints newly generated text.

## What you need

- Python 3.10 or newer.
- A converted RWKV-7 Hugging Face model directory, or an official `.pth`
  checkpoint that you can convert.
- Enough RAM or VRAM for the selected model. A dense fp16 checkpoint uses
  roughly 2 bytes per parameter before runtime buffers: approximately 0.8 GB
  for 0.4B, 3 GB for 1.5B, 5.8 GB for 2.9B, 14.4 GB for 7.2B, and 26.6 GB for
  13.3B. Leave additional memory for activations, cache, and temporary buffers.

Start with 0.1B or 0.4B to verify the installation.

## 1. Install

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/rwkv-rs/hf-adapter.git
cd hf-adapter
python -m venv .venv
```

Activate it on Linux or macOS:

```bash
source .venv/bin/activate
```

Activate it in Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install one of these profiles:

```bash
# Portable base install: CPU, MPS, or CUDA through the native backend.
python -m pip install -U pip
python -m pip install -e .

# Optimized NVIDIA CUDA/FLA path, normally on Linux.
python -m pip install -e ".[cuda]"

# Optional training or quantization dependencies.
python -m pip install -e ".[train]"
python -m pip install -e ".[quant]"

# Apple Silicon MLX tools.
python -m pip install -e ".[mlx]"
```

Install a PyTorch build appropriate for your hardware before these commands if
the default PyPI build does not provide the CUDA or platform support you need.
If `flash-linear-attention` cannot be installed, the base profile and native
backend remain usable.

Verify the base environment before downloading a model:

```bash
python examples/check_environment.py
```

Fix the first `FAIL` and rerun the command. Continue only after it prints
`RESULT: READY`.

## 2. Get and convert a model

If you already have a converted model directory containing `config.json`,
tokenizer files, remote-code Python files, and safetensors weights, skip to
[Run generation](#3-run-generation).

Official RWKV-7 checkpoints are published in
[`BlinkDL/rwkv7-g1`](https://huggingface.co/BlinkDL/rwkv7-g1). The example
below downloads the 0.4B checkpoint with the Hugging Face CLI installed by the
following command:

```bash
python -m pip install -U huggingface_hub
```

Download the checkpoint:

```bash
mkdir -p models/source
hf download BlinkDL/rwkv7-g1 \
  rwkv7-g1d-0.4b-20260210-ctx8192.pth \
  --local-dir models/source
```

In PowerShell, create the directory with
`New-Item -ItemType Directory -Force models/source` and either put each command
on one line or replace Bash's trailing `\` with PowerShell's backtick.

Download the official tokenizer vocabulary from
[`RWKV-LM/RWKV-v7`](https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/rwkv_vocab_v20230424.txt)
and save it as `models/source/rwkv_vocab_v20230424.txt`.

```bash
curl -L \
  https://raw.githubusercontent.com/BlinkDL/RWKV-LM/main/RWKV-v7/rwkv_vocab_v20230424.txt \
  -o models/source/rwkv_vocab_v20230424.txt
```

Use `curl.exe` instead of `curl` in Windows PowerShell if `curl` is configured
as a PowerShell alias.

Convert the checkpoint:

```bash
python scripts/convert_rwkv7_to_hf.py \
  --input models/source/rwkv7-g1d-0.4b-20260210-ctx8192.pth \
  --output models/rwkv7-g1d-0.4b-hf \
  --vocab-file models/source/rwkv_vocab_v20230424.txt \
  --precision fp16 \
  --attn-mode fused_recurrent \
  --no-fuse-norm
```

For 7.2B and 13.3B checkpoints, reduce conversion RAM and bound output shard
size:

```bash
python scripts/convert_rwkv7_to_hf.py \
  --input /path/to/model.pth \
  --output /path/to/model-hf \
  --vocab-file /path/to/rwkv_vocab_v20230424.txt \
  --precision fp16 \
  --attn-mode fused_recurrent \
  --no-fuse-norm \
  --low-memory \
  --max-shard-size 5GB
```

`--low-memory` lowers conversion RAM. It does not reduce inference VRAM.

Validate the converted directory before generation:

```bash
python examples/check_environment.py --model models/rwkv7-g1d-0.4b-hf
```

The result must include `[PASS] Model directory` and `RESULT: READY`.

## 3. Run generation

The included example automatically selects CUDA, MPS, or CPU. It uses FLA on
CUDA when available and otherwise selects the native backend:

```bash
python examples/generate.py \
  --model models/rwkv7-g1d-0.4b-hf \
  --prompt "User: Write a short greeting. Assistant:" \
  --max-new-tokens 8
```

Useful explicit configurations:

```bash
# NVIDIA CUDA with the optimized FLA backend.
python examples/generate.py --model /path/to/model-hf \
  --prompt "Hello" --device cuda --backend fla --dtype fp16

# CPU fallback. Start with a small checkpoint.
python examples/generate.py --model /path/to/model-hf \
  --prompt "Hello" --device cpu --backend native --dtype fp32

# Apple MPS fallback.
python examples/generate.py --model /path/to/model-hf \
  --prompt "Hello" --device mps --backend native --dtype fp16

# Sampling instead of deterministic greedy generation.
python examples/generate.py --model /path/to/model-hf \
  --prompt "Once upon a time" --temperature 0.8 --top-p 0.9

# Do not access the network after the model is prepared locally.
python examples/generate.py --model /path/to/model-hf \
  --prompt "Hello" --local-files-only
```

Run `python examples/generate.py --help` for all options.

The first-run gate checks execution, not model quality: the command must exit
with code 0 and print new text after the loading message.

## 4. Use the Transformers API

The direct API does not require `accelerate` or `device_map` for a single
device:

```python
import importlib.util
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "models/rwkv7-g1d-0.4b-hf"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16 if device.type == "cuda" else torch.float32

# Enable the native backend for CPU, MPS, or a CUDA installation without FLA.
if device.type != "cuda" or importlib.util.find_spec("fla") is None:
    os.environ["RWKV7_NATIVE_MODEL"] = "1"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    dtype=dtype,
).eval().to(device)

inputs = tokenizer(
    "User: Explain recurrent language models.\n\nAssistant:",
    return_tensors="pt",
)
inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

with torch.inference_mode():
    output = model.generate(
        **inputs,
        max_new_tokens=64,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
    )

new_tokens = output[0, inputs["input_ids"].shape[1]:]
print(tokenizer.decode(new_tokens, skip_special_tokens=True))
```

`trust_remote_code=True` is required because converted directories include the
RWKV-7 adapter classes. Only enable it for a model directory or Hub repository
you trust.

## 5. Verify the installation

Check the example and focused tests without loading a large checkpoint:

```bash
python examples/generate.py --help
python examples/check_environment.py
python -m pytest tests/test_user_quickstart.py -q
```

After conversion, run a real generation smoke:

```bash
python examples/generate.py \
  --model /path/to/model-hf \
  --prompt "User: Hello! Assistant:" \
  --max-new-tokens 8
```

## 6. Let an AI assistant do the setup

Use the copy-ready prompt and fail-closed checklist in
[`AI_ASSISTED_SETUP.md`](AI_ASSISTED_SETUP.md). It tells the assistant to
inspect the real machine, request approval before a large download, avoid
global package installation, and prove success with command exit status and
generated output. Do not give an assistant account tokens or SSH credentials
for this public-model setup.

## Common problems

### `No module named 'fla'`

Use `--backend native`, or install the CUDA profile on a supported NVIDIA
environment with `python -m pip install -e ".[cuda]"`.

### CUDA is unavailable

Confirm `python -c "import torch; print(torch.cuda.is_available())"`. If it is
false, install a CUDA-enabled PyTorch build that matches the host driver.

### Out of memory

Use a smaller checkpoint first. Close other GPU processes and remember that
conversion's `--low-memory` option does not lower inference VRAM. W8/W4 can
reduce model footprint, but speed and support are card-dependent; read
[`QUANTIZATION.md`](QUANTIZATION.md) before choosing a quantized path.

### The first run is slow

CUDA/Triton kernels and graph paths may compile or warm up on first use.
Measure steady-state performance after the first generation.

### Output quality is not chat-like

The adapter preserves the checkpoint; it does not turn a base model into an
instruction model. Use the prompt format and checkpoint variant appropriate
for the model you downloaded.

### Windows CUDA installation is difficult

Start with the base package and `--backend native`. The optimized FLA/Triton
path is primarily validated on Linux. WSL2 is another option for a Linux CUDA
environment.

## Next steps

- Visual speculative decoding, training, and multi-GPU workflows:
  [`ADVANCED_USAGE.md`](ADVANCED_USAGE.md)
- Training and PEFT/TRL: [`TRAINING.md`](TRAINING.md)
- Quantized inference: [`QUANTIZATION.md`](QUANTIZATION.md)
- Validated cards and limitations: [`HARDWARE_MATRIX.md`](HARDWARE_MATRIX.md)
- Performance backends: [`PERFORMANCE.md`](PERFORMANCE.md)
- Developer and benchmark documentation: [`README.md`](README.md)
