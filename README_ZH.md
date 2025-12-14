# RWKV-7 HF Adapter

[English](README.md) | [**中文**](README_ZH.md)

这是面向官方 RWKV-7 `.pth` 权重的 Hugging Face / Transformers 适配器。你可以
使用标准 `AutoModelForCausalLM` API 完成生成、状态缓存、PEFT/Trainer/TRL 训练、
W8/W4 量化、投机解码和多卡运行，并按设备选择原生或融合后端。

## 五分钟开始

新用户建议先使用 0.1B 或 0.4B 模型。下面的命令会创建独立环境、安装仓库并检查
Python、PyTorch、Transformers 和可用设备。

### Linux / macOS

```bash
git clone https://github.com/rwkv-rs/hf-adapter.git
cd hf-adapter
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
python examples/check_environment.py
```

### Windows PowerShell

```powershell
git clone https://github.com/rwkv-rs/hf-adapter.git
Set-Location hf-adapter
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e .
python examples/check_environment.py
```

Linux NVIDIA 用户可以安装包含 CUDA 优化依赖的版本：

```bash
python -m pip install -e ".[cuda]"
```

看到下面的输出说明基础环境已经可用：

```text
RESULT: READY
```

## 准备模型

如果你已经有转换好的 Hugging Face 模型目录，可以直接进入下一节。模型目录至少应
包含 `config.json`、tokenizer 文件和 `.safetensors` 或 `.bin` 权重。

如果你只有官方 `.pth` 权重，请按照
[下载与转换逐步教程](docs/USER_GUIDE_ZH.md#2-下载并转换模型)操作。该教程包含：

- Hugging Face 官方模型下载位置截图；
- GitHub tokenizer 下载位置截图；
- Windows 与 Linux 可复制命令；
- 大模型低内存转换和断点恢复方法；
- 模型目录的明确 `PASS` 标准。

也可以先让环境检查脚本验证模型目录：

```bash
python examples/check_environment.py --model /path/to/rwkv7-model-hf
```

通过时会看到：

```text
MODEL DIRECTORY: PASS
RESULT: READY
```

## 运行第一次生成

```bash
python examples/generate.py \
  --model /path/to/rwkv7-model-hf \
  --prompt "User: 你好！请用一句话介绍 RWKV。 Assistant:" \
  --max-new-tokens 64
```

Windows PowerShell 可以使用反引号换行，也可以写成一行：

```powershell
python examples/generate.py --model D:\models\rwkv7-model-hf --prompt "User: 你好！ Assistant:" --max-new-tokens 64
```

完成标志：命令退出码为 `0`，输出中显示所用 `device` 和 `dtype`，并在输入提示词后
生成新文本。示例会自动选择 CUDA、MPS 或 CPU，并始终使用仓库原生后端。

> 转换后的模型使用仓库代码，因此需要 `trust_remote_code=True`。只加载你信任的
> 本地目录或 Hugging Face 仓库。

RTX 5090 的同精度 Native 证据使用官方 RWKV-Gradio-3 的 FP16 权重、状态和输入输出
作为对照。g1h 7.2B 缓存解码在 B1/B8 达到官方 `1.0010x/1.0104x`，并通过
logits、状态和 greedy 门禁；g1h 2.9B/13.3B 在 B1/B8、提示长度
128/512/2048 的 12 个 prefill 单元全部通过。仓库只会对这些精确卡、模型和形状
自动选择已验证策略，证据见
[`bench/5090_native_official_fp16_production_20260718/README.md`](bench/5090_native_official_fp16_production_20260718/README.md)。

## 使用标准 Transformers API

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/path/to/rwkv7-model-hf"
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
dtype = torch.float16 if device.type in {"cuda", "mps"} else torch.float32

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    dtype=dtype,
).eval().to(device)

inputs = tokenizer("User: 你好！ Assistant:", return_tensors="pt")
inputs = {name: value.to(device) for name, value in inputs.items()}
output = model.generate(
    **inputs,
    max_new_tokens=32,
    do_sample=False,
    use_cache=True,
    pad_token_id=tokenizer.pad_token_id,
)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

通过标准：输出 tensor 的 batch 与输入一致，生成长度增加，并且文本解码成功。

## 按目标选择教程

| 你要完成的任务 | 从这里开始 |
|---|---|
| 零基础安装、下载、转换与生成 | [中文逐步指南](docs/USER_GUIDE_ZH.md) |
| 全部功能导航 | [全功能使用指南](docs/COMPLETE_ADAPTER_GUIDE.md) |
| 批量转换、HF API、缓存和分块 prefill | [推理工作流](docs/INFERENCE_WORKFLOWS.md) |
| PEFT LoRA、Trainer、SFT、DPO、GRPO | [训练工作流](docs/TRAINING_WORKFLOWS.md) |
| 对齐官方 train_temp 数学和训练效果 | [train_temp CUDA 训练对齐教程](docs/TRAIN_TEMP_CUDA.md) |
| 在官方 RWKV-Gradio-3 网页使用 Native HF | [Gradio Native HF 教程](docs/GRADIO_NATIVE_HF.md) |
| 投机解码、`device_map`、DeepSpeed 多卡 | [高级使用教程](docs/ADVANCED_USAGE_ZH.md) |
| bitsandbytes W8/W4、原生 MM8/MM4 | [量化使用教程](docs/QUANTIZATION_USAGE.md) |
| Apple MPS、MLX、CoreML | [Apple 使用教程](docs/APPLE_USAGE.md) |
| 让 AI 帮你安装、运行或排错 | [统一 AI 操作入口](docs/AI_ASSISTED_SETUP.md) |
| 选择显卡与后端配置 | [硬件矩阵](docs/HARDWARE_MATRIX.md) |
| 查看性能结果与复现命令 | [性能指南](docs/PERFORMANCE.md) 与 [benchmark 索引](bench/INDEX.md) |

## 后端怎么选

- **NVIDIA CUDA：** 先使用自动选择。需要已验证的融合性能路线时，再按硬件矩阵
  启用对应环境变量。
- **Apple Silicon：** 普通 Transformers 工作流使用 MPS；追求 Apple 原生性能时
  使用 MLX 教程。
- **CPU 或便携环境：** 原生后端可以完成转换、接口检查和小模型生成。
- **量化：** 显存优先时查看 W8/W4 footprint；速度优先时选择与你的显卡、模型和
  batch 完全一致的配对结果。

## RTX 5090 最新 W4 生产路径

RTX 5090 上的官方 g1h 1.5B、2.9B、7.2B 和 13.3B BF16 模型已有精确卡、
精确模型的 Tensor Core W4 配置。运行时会自动选择 group-128 Marlin FFN、
是否量化 `lm_head` 以及是否保留最后一层 FFN；用户不需要手工设置 BN/TN。

在 prompt128/decode128 的配对 hot-BF16 B1/B8 验收中，四个模型的最差
prefill/decode 分别为：1.5B `1.0097x/1.1854x`、2.9B
`1.0092x/1.2222x`、7.2B `1.0010x/1.4978x`、13.3B
`1.0153x/1.4670x`。footprint 为 BF16 的 `0.6250x/0.5776x/0.5298x/0.5347x`，
所有新增 profile 的 prompt/final cosine 均不低于 `0.9995`，next-token 一致。

扩展契约覆盖四档模型的八种 FFN GEMM shape：group-128 为 `280/280`
通过、bit-exact、错误 BN fail-close；group-32 实验网格另有 `48/48`。g1d
0.4B 全 FFN 候选因 decode 和 cosine 未过门而保持 fallback。使用方法见
[量化教程的 RTX 5090 专节](docs/QUANTIZATION_USAGE.md#5-rtx-5090-g1h-bntn-tensor-core-w4)，
原始结果见
[`bench/5090_bntn_all_models_20260716/`](bench/5090_bntn_all_models_20260716/README.md)。

## 常见问题恢复

### `RESULT: NEEDS ATTENTION`

重新运行环境检查并按 `ERROR` 行处理第一个缺失项：

```bash
python examples/check_environment.py
```

### 模型目录检查失败

确认 `--model` 指向转换后的目录而不是单个 `.pth` 文件，再执行：

```bash
python examples/check_environment.py --model /path/to/rwkv7-model-hf
```

### CUDA 显存不足

先把 `--max-new-tokens` 降到 `8`，使用 0.1B/0.4B 验证流程，然后在
[量化教程](docs/QUANTIZATION_USAGE.md)中选择 W8/W4 路线。转换大型 `.pth` 时使用
`--low-memory`。

### 旧模型提示缺少 FLA

普通用户不需要安装 FLA。运行
`python scripts/sync_hf_adapter_code.py /path/to/rwkv7-model-hf` 更新旧模型目录，
再按 [中文逐步指南](docs/USER_GUIDE_ZH.md)使用原生后端。

### 下载中断

保留已下载文件并使用教程中的可续传命令继续；转换成功后再运行模型目录检查，
不需要重新创建 Python 环境。

## 让 AI 代你执行

所有 AI 操作集中在 [`docs/AI_ASSISTED_SETUP.md`](docs/AI_ASSISTED_SETUP.md)。打开后
选择 `TASK_ID`，填写模型路径、设备和 dtype，即可让 AI 执行安装、推理、缓存、
投机解码、训练、多卡、量化或 Apple 工作流。该入口还规定了退出码、通过标记和
失败恢复的统一返回格式。

请把密码、SSH 密钥和私有 token 保留在本机，只向 AI 提供任务需要的普通路径与
公开环境信息。

## 更多资料

- [完整英文工程说明](README.md)
- [文档目录](docs/README.md)
- [贡献归属](CONTRIBUTIONS.md)
- [贡献者名单](CONTRIBUTORS.md)
- [许可证](LICENSE)
