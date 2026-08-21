# `rwkv7-hf` command-line reference

`rwkv7-hf==0.8.1` installs one unified command that works from any directory:

```bash
python -m pip install "rwkv7-hf==0.8.1"
rwkv7-hf --help
```

The package contains adapter code and conversion tools, not model weights.
Use a published Hugging Face model directly, or convert an official `.pth`
checkpoint that you already downloaded.

## Commands

| Command | Purpose | Success gate |
|---|---|---|
| `rwkv7-hf doctor` | Inspect Python, PyTorch, accelerators, kernel candidates, and fallback policy | `RESULT: READY` |
| `rwkv7-hf kernels` | Show status, recommend an exact wheel, list the release index, or install an exact wheel | command exits with code 0 |
| `rwkv7-hf smoke` | Load a Hub ID or local HF directory and run prefill plus greedy decode | `RESULT: PASS` |
| `rwkv7-hf convert` | Convert one official RWKV-7 `.pth` checkpoint to an HF directory | output is saved and command exits with code 0 |

Every subcommand has its own help:

```bash
rwkv7-hf doctor --help
rwkv7-hf kernels --help
rwkv7-hf smoke --help
rwkv7-hf convert --help
```

## Convert a checkpoint

```bash
rwkv7-hf convert \
  --input /path/to/model.pth \
  --output /path/to/model-hf \
  --vocab-file /path/to/rwkv_vocab_v20230424.txt \
  --precision fp16 \
  --attn-mode chunk \
  --adapter-layout thin \
  --no-fuse-norm \
  --low-memory
```

`thin` is the default. It writes three small remote-code entrypoints that use
the installed, version-pinned `rwkv7-hf` runtime. `bundled` copies a complete
runtime-code snapshot for offline or archival use. Both layouts write the same
model weights. For large checkpoints, add `--max-shard-size 5GB` together with
`--low-memory`.

Verify a converted directory:

```bash
rwkv7-hf doctor
rwkv7-hf smoke --model /path/to/model-hf --device auto
```

The legacy absolute script remains usable from a source checkout:

```bash
python /absolute/path/to/hf-adapter/scripts/convert_rwkv7_to_hf.py --help
```

## Compatibility aliases

Existing automation does not need an immediate migration. These aliases remain
installed:

```text
rwkv7-hf-convert  -> rwkv7-hf convert
rwkv7-hf-doctor   -> rwkv7-hf doctor
rwkv7-hf-kernels  -> rwkv7-hf kernels
rwkv7-hf-smoke    -> rwkv7-hf smoke
```

## 中文速查

安装 wheel 后无需 clone 仓库，可在任意目录执行：

```bash
rwkv7-hf doctor
rwkv7-hf smoke --model wangyue114514/rwkv7-g1d-0.1b-hf \
  --revision v0.7.0 --device auto
rwkv7-hf convert --help
```

基础 wheel 不包含任何模型权重。直接推理时，`smoke`/Transformers 会从 Hub
下载所选模型；自行转换时，需要另外准备官方 `.pth` 和词表文件。单模型转换已包含
在 wheel 中，只有批量工具、开发、仓库测试和 benchmark 复现才需要 clone 源码。
