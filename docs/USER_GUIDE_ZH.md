# RWKV-7 HF Adapter 普通用户指南

本文档面向希望用 Hugging Face `transformers` 直接运行 RWKV-7 的用户。
英文版见 [`USER_GUIDE.md`](USER_GUIDE.md)。

## 最短使用流程

1. 安装仓库。
2. 下载官方 `.pth` 权重和词表。
3. 转换为 Hugging Face 模型目录。
4. 运行 `examples/generate.py`。

建议先用 0.1B 或 0.4B 验证环境。fp16 权重大约占用每参数 2 字节，
推理时还需要额外的激活、缓存和临时显存。

## 1. 安装

```bash
git clone https://github.com/rwkv-rs/hf-adapter.git
cd hf-adapter
python -m venv .venv
```

Linux/macOS 激活环境：

```bash
source .venv/bin/activate
```

Windows PowerShell 激活环境：

```powershell
.venv\Scripts\Activate.ps1
```

选择安装方式：

```bash
# 基础安装：CPU、MPS，或使用 native 后端的 CUDA。
python -m pip install -U pip
python -m pip install -e .

# Linux NVIDIA 优化 CUDA/FLA 后端。
python -m pip install -e ".[cuda]"

# 可选：训练、量化、Apple MLX。
python -m pip install -e ".[train]"
python -m pip install -e ".[quant]"
python -m pip install -e ".[mlx]"
```

如果 FLA 安装失败，基础安装仍可配合 `--backend native` 使用。

## 2. 下载并转换模型

官方权重位于
[`BlinkDL/rwkv7-g1`](https://huggingface.co/BlinkDL/rwkv7-g1)。下载 0.4B
示例：

```bash
mkdir -p models/source
hf download BlinkDL/rwkv7-g1 \
  rwkv7-g1d-0.4b-20260210-ctx8192.pth \
  --local-dir models/source
```

PowerShell 使用
`New-Item -ItemType Directory -Force models/source` 创建目录，并把命令写在一行，
或把 Bash 末尾的 `\` 换成 PowerShell 反引号。

再从
[`RWKV-LM/RWKV-v7`](https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/rwkv_vocab_v20230424.txt)
下载 `rwkv_vocab_v20230424.txt`，保存到 `models/source/`。

```bash
curl -L \
  https://raw.githubusercontent.com/BlinkDL/RWKV-LM/main/RWKV-v7/rwkv_vocab_v20230424.txt \
  -o models/source/rwkv_vocab_v20230424.txt
```

Windows PowerShell 如果把 `curl` 配置成别名，请改用 `curl.exe`。

转换模型：

```bash
python scripts/convert_rwkv7_to_hf.py \
  --input models/source/rwkv7-g1d-0.4b-20260210-ctx8192.pth \
  --output models/rwkv7-g1d-0.4b-hf \
  --vocab-file models/source/rwkv_vocab_v20230424.txt \
  --precision fp16 \
  --attn-mode fused_recurrent \
  --no-fuse-norm
```

7.2B、13.3B 等大模型转换时增加：

```text
--low-memory --max-shard-size 5GB
```

`--low-memory` 只降低转换时的内存，不会降低推理显存。

## 3. 直接生成文本

默认会自动选择 CUDA、MPS 或 CPU；CUDA 已安装 FLA 时自动使用 FLA，
否则自动使用 native 后端：

```bash
python examples/generate.py \
  --model models/rwkv7-g1d-0.4b-hf \
  --prompt "User: 你好，请做一个简短的自我介绍。 Assistant:" \
  --max-new-tokens 64
```

常用配置：

```bash
# NVIDIA CUDA + FLA。
python examples/generate.py --model /path/to/model-hf \
  --prompt "你好" --device cuda --backend fla --dtype fp16

# CPU。建议只先试小模型。
python examples/generate.py --model /path/to/model-hf \
  --prompt "你好" --device cpu --backend native --dtype fp32

# Apple MPS。
python examples/generate.py --model /path/to/model-hf \
  --prompt "你好" --device mps --backend native --dtype fp16

# 开启采样。
python examples/generate.py --model /path/to/model-hf \
  --prompt "从前有一座山" --temperature 0.8 --top-p 0.9
```

查看全部参数：

```bash
python examples/generate.py --help
```

## 4. Python API

```python
import importlib.util
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "models/rwkv7-g1d-0.4b-hf"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16 if device.type == "cuda" else torch.float32

# CPU、MPS 或没有安装 FLA 的 CUDA 环境使用 native 后端。
if device.type != "cuda" or importlib.util.find_spec("fla") is None:
    os.environ["RWKV7_NATIVE_MODEL"] = "1"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    dtype=dtype,
).eval().to(device)

inputs = tokenizer("User: 你好！\n\nAssistant:", return_tensors="pt")
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

转换后的模型目录包含 remote-code 适配文件，因此必须设置
`trust_remote_code=True`。只对可信的本地目录或 Hugging Face 仓库使用该选项。

## 常见问题

- **提示缺少 `fla`**：增加 `--backend native`，或者在支持的 Linux CUDA
  环境安装 `.[cuda]`。
- **CUDA 不可用**：运行
  `python -c "import torch; print(torch.cuda.is_available())"` 检查 PyTorch。
- **显存不足**：先换小模型。量化可以省显存，但不同显卡的速度和支持情况
  不同，请阅读 [`QUANTIZATION.md`](QUANTIZATION.md)。
- **第一次运行很慢**：CUDA/Triton 内核可能需要首次编译和预热。
- **输出不像聊天模型**：适配器不会改变模型训练性质。基础模型并不会因为接入 HF
  自动变成指令模型，请选择合适的 checkpoint 和提示格式。
- **Windows CUDA/FLA 安装困难**：先使用基础安装和 native 后端；优化后端主要在
  Linux 上验证，也可以考虑 WSL2。

更多文档：训练见 [`TRAINING.md`](TRAINING.md)，硬件支持见
[`HARDWARE_MATRIX.md`](HARDWARE_MATRIX.md)，性能后端见
[`PERFORMANCE.md`](PERFORMANCE.md)。
