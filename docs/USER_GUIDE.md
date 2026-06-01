# RWKV-7 HF Adapter User Guide

This guide is for users who want to run an official RWKV-7 checkpoint through
the standard Hugging Face `transformers` API. Benchmark development and kernel
tuning are not required for normal inference.

Chinese version: [`USER_GUIDE_ZH.md`](USER_GUIDE_ZH.md)

## Choose a path

- **Normal first run:** complete [Install](#1-install), then run the public-model
  smoke. No repository clone or checkpoint conversion is required.
- **Python integration:** after the smoke passes, continue to
  [Use the Transformers API](#4-use-the-transformers-api).
- **New/custom checkpoint:** use the optional
  [conversion workflow](#2-optional-get-and-convert-a-model).
- **AI-assisted setup:** give [`AI_ASSISTED_SETUP.md`](AI_ASSISTED_SETUP.md) to
  a terminal-capable assistant.
- **Advanced task:** after the first generation, choose a bounded workflow from
  [`COMPLETE_ADAPTER_GUIDE.md`](COMPLETE_ADAPTER_GUIDE.md).

For a normal installation, setup is complete only when:

1. `rwkv7-hf-doctor` prints `RESULT: READY`;
2. `rwkv7-hf-smoke` prints `RESULT: PASS` and writes a valid report; and
3. the report says `rwkv7_hf: 0.8.0` and contains generated tokens.

Conversion and repository-development workflows have additional model-directory
and test gates described in their own sections. A downloaded file or a command
that merely started is not a pass.

## What you need

- Python 3.10 or newer.
- A published RWKV-7 Hugging Face model ID for normal use. Start with
  `wangyue114514/rwkv7-g1d-0.1b-hf`.
- Enough RAM or VRAM for the selected model. See the
  [published model matrix](PUBLISHED_MODELS.md); leave headroom beyond the
  stored fp16 weight size for runtime buffers.
- A repository checkout only if you are converting a new checkpoint, modifying
  code, running repository tests, or reproducing a benchmark.

## 1. Install

Create and activate an isolated environment:

```bash
python -m venv .venv
source .venv/bin/activate                 # Windows: .venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install "rwkv7-hf==0.8.0"
```

Install a PyTorch build appropriate for your hardware first when the default
PyPI resolver does not provide the CUDA or vendor runtime you need. Ordinary
RWKV inference does not require Flash Linear Attention.

Inspect the installed runtime without downloading model weights:

```bash
rwkv7-hf-doctor
rwkv7-hf-kernels status
rwkv7-hf-kernels recommend
```

`rwkv7-hf-doctor` must finish with `RESULT: READY`. On Linux NVIDIA, run the
next command only when the recommendation reports one exact compatible wheel:

```bash
rwkv7-hf-kernels install
rwkv7-hf-doctor
```

The installer matches the Python ABI, platform, PyTorch series and C++ ABI,
CUDA runtime, GPU compute capability, and adapter series. It never installs a
nearby build. Installing `rwkv7-hf` itself does not silently install a GPU
binary. At runtime the default `auto` policy selects a compatible prebuilt
extension, then JIT where allowed, then the existing correct portable fallback.
See [prebuilt CUDA kernel wheels](KERNEL_WHEELS.md).

Now run the public 0.1B acceptance command:

```bash
rwkv7-hf-smoke \
  --model wangyue114514/rwkv7-g1d-0.1b-hf \
  --revision v0.7.0 \
  --device auto \
  --output rwkv7-smoke.json
```

The first run downloads the public model. `RESULT: PASS` means loading, prefill,
greedy decode, finite logits, and runtime reporting all completed. Smoke timing
is installation telemetry, not a general performance claim.

Install optional published extras only when a later workflow needs them:

```bash
python -m pip install "rwkv7-hf[cuda]==0.8.0"    # Linux NVIDIA JIT helper
python -m pip install "rwkv7-hf[train]==0.8.0"   # PEFT/TRL/DeepSpeed
python -m pip install "rwkv7-hf[quant]==0.8.0"   # bitsandbytes
python -m pip install "rwkv7-hf[mlx]==0.8.0"     # Apple Silicon MLX
```

## 2. Optional: get and convert a model

If you already have a converted model directory containing `config.json`,
tokenizer files, remote-code Python files, and safetensors weights, skip to
[Run generation](#3-run-generation).

The converter and repository checks are development tools rather than PyPI
console commands. Clone the source tree and install it in editable mode before
continuing:

```bash
git clone https://github.com/rwkv-rs/hf-adapter.git
cd hf-adapter
python -m pip install -e .
```

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
  --adapter-layout thin \
  --no-fuse-norm
```

`thin` is the default and matches the published Hub repositories: the output
contains three small remote-code entrypoints backed by the `rwkv7-hf` package.
Use `--adapter-layout bundled` only when an offline or archival model directory
must carry a complete runtime-code snapshot. The weights are identical in both
layouts.

For 7.2B and 13.3B checkpoints, reduce conversion RAM and bound output shard
size:

```bash
python scripts/convert_rwkv7_to_hf.py \
  --input /path/to/model.pth \
  --output /path/to/model-hf \
  --vocab-file /path/to/rwkv_vocab_v20230424.txt \
  --precision fp16 \
  --attn-mode fused_recurrent \
  --adapter-layout thin \
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

The included example automatically selects CUDA, MPS, or CPU and always loads
the canonical native backend:

```bash
python examples/generate.py \
  --model models/rwkv7-g1d-0.4b-hf \
  --prompt "User: Write a short greeting. Assistant:" \
  --max-new-tokens 8
```

Useful explicit configurations:

```bash
# NVIDIA CUDA with native fused kernels selected by the card policy.
python examples/generate.py --model /path/to/model-hf \
  --prompt "Hello" --device cuda --backend native --dtype fp16

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
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "wangyue114514/rwkv7-g1d-0.1b-hf"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16 if device.type == "cuda" else torch.float32

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

`trust_remote_code=True` is required because the published thin repositories
and converted directories contain small adapter entrypoints; the maintained
implementation comes from the installed `rwkv7-hf`. Only enable it for a model
directory or Hub repository you trust.

### Public argument and config names

The causal-LM API uses inspectable Transformers-style argument names such as
`input_ids`, `attention_mask`, `inputs_embeds`, `past_key_values`, `labels`,
`use_cache`, `output_hidden_states`, `return_dict`, `logits_to_keep`,
`position_ids`, and `cache_position`. The optional FLA reference wrapper also
keeps `**kwargs` for version-specific Transformers arguments. Use
`logits_to_keep`; the deprecated `num_logits_to_keep` spelling remains a
compatibility alias.

RWKV checkpoints and kernels historically use `num_heads`, while Transformers
tools commonly inspect `num_attention_heads`. Both native and FLA-reference
configs accept either spelling and expose both attributes with the same value:

```python
from transformers import AutoConfig

config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
assert config.num_heads == config.num_attention_heads
```

Existing `num_heads`-only configs remain valid. New code may supply either
name, but supplying different non-null values is an error. Both fields are
written during config serialization. Internal parameter names, state-dict
keys, and kernel-local RWKV notation are intentionally unchanged.

## 5. Verify the installation

For an ordinary PyPI installation, the complete gate is:

```bash
rwkv7-hf-doctor
rwkv7-hf-smoke --model wangyue114514/rwkv7-g1d-0.1b-hf \
  --revision v0.7.0 --device auto --output rwkv7-smoke.json
```

Inside a cloned source tree, additionally check repository examples and the
focused quick-start tests:

```bash
python examples/generate.py --help
python examples/check_environment.py --model /path/to/model-hf
python -m pytest tests/test_user_quickstart.py -q
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

Normal inference does not require FLA. Confirm that `rwkv7-hf==0.8.0` is
installed and use the canonical native model. Install `fla-reference` only for
an explicitly named comparison benchmark.

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
Run `rwkv7-hf-kernels recommend`; an exact prebuilt wheel avoids local JIT for
the extensions it contains. Measure steady-state performance only after the
first generation.

### Output quality is not chat-like

The adapter preserves the checkpoint; it does not turn a base model into an
instruction model. Use the prompt format and checkpoint variant appropriate
for the model you downloaded.

### Windows CUDA installation is difficult

Start with the base package and `--backend native`. The native CUDA/Triton
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
