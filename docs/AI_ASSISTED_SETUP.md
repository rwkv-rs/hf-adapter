# Use an AI assistant to install and run RWKV-7

This page is written for both users and terminal-capable AI assistants such as
Codex, Claude Code, Cursor, and similar tools. It covers the first successful
local generation only. It does not ask the assistant to run benchmarks, train a
model, tune kernels, or change repository code.

中文版操作说明和可直接粘贴的提示词见下文。普通手动安装请看
[`USER_GUIDE_ZH.md`](USER_GUIDE_ZH.md)。

## Before giving an AI terminal access

- Open the cloned `hf-adapter` repository as the AI workspace.
- Do not paste passwords, Hugging Face tokens, SSH keys, or other secrets into
  the prompt. Public RWKV-7 files used by this guide do not require a token.
- Review commands before approving them. The assistant should ask before a
  large download, a system-wide package change, or deleting files.
- Keep the first run on the 0.4B checkpoint. Larger models consume much more
  disk, RAM, and VRAM and are not useful for checking whether installation works.

## Copy this prompt to the AI

Replace nothing unless you want to use a model that has already been converted.

```text
Set up this RWKV-7 Hugging Face adapter and prove that one local generation works.

Read README.md, docs/USER_GUIDE_ZH.md, and docs/AI_ASSISTED_SETUP.md before
running commands. Work only inside this repository and its models/ directory.

Rules:
1. First report the operating system, active shell, Python version, free disk
   space, PyTorch version, and detected GPU. Do not guess them.
2. Use Python 3.10 or newer and create an isolated .venv in this repository.
3. Default to the public BlinkDL/rwkv7-g1 0.4B checkpoint. Before downloading,
   state the file name, destination, and approximate disk requirement, then ask
   me to approve the download.
4. Start with the portable native backend. On Linux with a supported NVIDIA
   GPU, the optional [cuda] profile may be attempted, but a failed FLA install
   must fall back to native instead of blocking the first generation.
5. Do not install packages globally. Do not delete existing files. Do not run
   benchmark, training, quantization, or multi-GPU scripts.
6. After installation, run: python examples/check_environment.py
7. Download the official checkpoint and vocabulary using the exact commands in
   docs/USER_GUIDE_ZH.md. Convert them with scripts/convert_rwkv7_to_hf.py.
8. Validate the converted directory with:
   python examples/check_environment.py --model models/rwkv7-g1d-0.4b-hf
9. Run a deterministic 8-token smoke generation with examples/generate.py.
10. Stop on a non-zero exit code, explain the first real error, fix only that
    error, and rerun the failed command. Never claim success from file existence
    alone.
11. Use trust_remote_code=True only for this trusted local converted directory.
12. Finish with the exact model path, backend, device, dtype, generation output,
    and every validation command's exit status.

Acceptance requires all three:
- examples/check_environment.py prints RESULT: READY;
- the model-directory check prints [PASS] Model directory;
- examples/generate.py exits with code 0 and prints generated text.
```

## What the AI should do

The setup state machine is intentionally small and fail-closed:

| State | Required evidence | Next state |
|---|---|---|
| Inspect | Real OS, shell, Python, disk, and accelerator output | Create `.venv` |
| Install | `examples/check_environment.py` prints `RESULT: READY` | Download |
| Download | Checkpoint and vocabulary exist at the documented paths | Convert |
| Convert | Converter exits `0`; output contains config, tokenizer, and weights | Validate model |
| Validate model | Doctor prints `[PASS] Model directory` | Generate |
| Generate | Command exits `0` and prints new text | Done |

An AI must not replace a failed state with prose such as "the setup should
work." The command must be rerun successfully.

## Exact first-run acceptance command

After the model is converted, the AI should run this command without sampling:

```bash
python examples/generate.py \
  --model models/rwkv7-g1d-0.4b-hf \
  --prompt "User: Say hello in one sentence. Assistant:" \
  --max-new-tokens 8
```

PowerShell uses the same arguments with backticks instead of trailing `\`, or
places the command on one line.

## Use the model from an AI application

Once setup passes, an application can load the converted model through standard
Hugging Face APIs. The complete Python example is in
[`USER_GUIDE.md#4-use-the-transformers-api`](USER_GUIDE.md#4-use-the-transformers-api)
and [`USER_GUIDE_ZH.md#5-python-api`](USER_GUIDE_ZH.md#5-python-api). Keep
`use_cache=True`; RWKV uses its recurrent state through the HF cache contract.

This repository provides a model adapter, not a hosted chat service. An app is
responsible for its own prompt template, conversation history, request limits,
and process lifecycle.

## Safe troubleshooting handoff

When asking another AI for help, provide only:

```text
OS and shell:
GPU name and VRAM:
Python version:
Command that failed:
Complete error text from the first failure:
Output of python examples/check_environment.py:
Model path (no credentials):
```

Do not provide account tokens or private SSH credentials. For CUDA errors, also
include `python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"`.
