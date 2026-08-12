# RWKV-7 HF Adapter

[English](README.md) | [**中文**](README_ZH.md)

这是面向官方 RWKV-7 `.pth` 权重的 Hugging Face / Transformers 适配器。你可以
使用标准 `AutoModelForCausalLM` API 完成生成、状态缓存、PEFT/Trainer/TRL 训练、
W8/W4 量化、投机解码和多卡运行，并按设备选择原生或融合后端。

## 项目状态

已发布的 **RWKV-7 HF `v0.7.0` 正式版交付范围已经完成**。它保留 v0.6
适配器里程碑的转换、Transformers 标准接口、PEFT/TRL、状态缓存、
Native/no-FLA、W8/W4、PP/TP 和 ZeRO，并加入 Ascend、Biren、MetaX、MUSA
边界以及最新的 NVIDIA 精确显卡性能路线。

完成状态按模型、显卡、精度、batch 和序列形状声明；增加新的显卡或测试形状属于
发布后的矩阵扩展，不会重新打开已经完成的版本里程碑。当前结论见
[HF_STATUS.md](HF_STATUS.md)，正式验收映射见
[docs/ACCEPTANCE.md](docs/ACCEPTANCE.md)，后续扩展项目见
[HF_TODO.md](HF_TODO.md)。

当前主线还已合入 V100/RTX 4080 的精确 B8 解码调优、RTX 4080
0.4B/1.5B/2.9B 的 B8 分组 W/A/V 投影，以及 RTX 4080 7.2B/B8
FP16-state 路径。这些均已有配对性能、正确性和回退边界证据，不再列为待完成缺口；
具体结果见 [BENCHMARK.md](BENCHMARK.md)。

## 五分钟开始

新用户建议先使用 0.1B 或 0.4B 模型。下面的命令会创建独立环境、安装仓库并检查
Python、PyTorch、Transformers 和可用设备。

直接从 PyPI 安装正式版：

```bash
python -m pip install -U rwkv7-hf
```

需要开发当前源码时再克隆仓库：

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

华为昇腾用户需先安装与当前 CANN 精确匹配的 PyTorch/torch-npu wheel，再安装
适配器入口：

```bash
python -m pip install -e ".[ascend]"
```

壁仞 BR106M 用户需先加载匹配的 BIRENSUPA/torch_br 环境，再安装适配器入口：

```bash
python -m pip install -e ".[biren]"
```

沐曦 MetaX C500 用户需先进入官方 MXMACA/PyTorch 环境，再安装适配器入口：

```bash
python -m pip install -e ".[metax]"
```

看到下面的输出说明基础环境已经可用：

```text
RESULT: READY
```

没有 GPU 或模型时，可以先运行不下载权重的 Windows/CPU tiny 演示。它会验证
标准生成、CPU 反向传播、参数更新和保存重载：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_cpu_demo.ps1 -Install
```

完整命令、通过标记和失败恢复见 [Windows 与 CPU 教程](docs/WINDOWS_CPU.md)。

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

## RTX 5090 已验证结果

- **Native 对 Albatross/v3a：**官方 g1h 7.2B FP16 缓存解码在 B1/B8 达到
  `146.42/899.51 tok/s`，v3a 对照为 `146.28/890.21`，即
  `1.0010x/1.0104x`；logits、循环状态、top-1 和 greedy token 全部通过。
- **Prefill：**官方 g1h 2.9B/13.3B 在 B1/B8、prompt128/512/2048 的
  12 个单元全部通过，Native 速度为 v3a 的 `1.0029x–1.5690x`，并通过
  tensor、state 和 token 对齐。
- **完整 FLA Qwen3.5：**B1/B8 共 8 组模型对、144/144 性能单元通过。
  dense prefill/decode 最低为 `1.0226x/2.8130x`；RWKV-7 7.2B 对
  Qwen3.5-9B 的 B1/B8 最低 prefill 为 `1.1739x/1.0309x`，decode 为
  `2.8934x/2.8130x`。
- **最新检查点 Qwen3.5 Prefill PD：**g1d/g1i 0.4B/1.5B/2.9B/7.2B
  对官方 Qwen3.5 0.8B/2B/4B/9B，在 B1/B8、P128/512/2048 的 24 个
  dense-FP16 单元全部通过。参数规模校正 Prefill 最低/中位为
  `1.072987x/1.317515x`，原始 Prefill 最低为 `1.347871x`；Qwen
  full-FLA contract 为 24/24，P2048 graph/eager 正确性为 8/8。
- **Tensor Core W4：**官方 g1h 1.5B/2.9B/7.2B/13.3B 的 B1/B8 共
  8 个 all-phase 单元全部通过；footprint 为 `0.5298x–0.6250x`，最低
  prefill/decode 为 `1.0010x/1.1854x`，cosine 高于 `0.9995`，next token
  8/8 一致，group-128 物理网格 280/280 通过。
- **训练：**Native B16/T512 BF16 train_temp 通过 399/399 梯度和参数更新对齐、
  3 个配对 real-MiniPile seed、连续 5,000 steps 和 2,500+2,500 断点恢复；
  配对中位速度为官方 `1.00049x`，5,000-step 速度为官方 `1.00255x`。
- **MATH500：**完整 `500 x 64` 运行达到 pass@64 `0.38`，对仓库 Albatross
  reference 的 summary/decode 速度为 `4.336x/4.871x`。

证据：
[`Native/v3a`](bench/5090_native_official_fp16_production_20260718/README.md)、
[`Qwen3.5`](bench/5090_g1h_qwen35_b1_b8_20260715/README.md)、
[`最新检查点 Qwen3.5 Prefill PD`](bench/5090_g1i_qwen35_prefill_pd_sota_20260811/README.md)、
[`W4`](bench/5090_bntn_all_models_20260716/README.md)、
以及 [`训练`](bench/5090_native_train_temp_real_minipile_20260718/README.md)。

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

### 公开参数与配置命名

公开的因果语言模型 `forward` 使用 Transformers 风格的显式参数名，并保留
`**kwargs` 接收不同 Transformers 版本的扩展参数。配置同时接受 `num_heads` 和
`num_attention_heads`，两者冲突时会报错，保存时会同时写出；RWKV checkpoint 与
kernel 内部变量名保持不变。完整规则见
[中文普通用户指南](docs/USER_GUIDE_ZH.md#公开参数与配置命名)。

## 按目标选择教程

| 你要完成的任务 | 从这里开始 |
|---|---|
| 零基础安装、下载、转换与生成 | [中文逐步指南](docs/USER_GUIDE_ZH.md) |
| 全部功能导航 | [全功能使用指南](docs/COMPLETE_ADAPTER_GUIDE.md) |
| Windows/CPU 无模型下载推理与微型训练 | [Windows 与 CPU 教程](docs/WINDOWS_CPU.md) |
| 批量转换、HF API、缓存和分块 prefill | [推理工作流](docs/INFERENCE_WORKFLOWS.md) |
| PEFT LoRA、Trainer、SFT、DPO、GRPO | [训练工作流](docs/TRAINING_WORKFLOWS.md) |
| 对齐官方 train_temp 数学和训练效果 | [train_temp CUDA 训练对齐教程](docs/TRAIN_TEMP_CUDA.md) |
| 在官方 RWKV-Gradio-3 网页使用 Native HF | [Gradio Native HF 教程](docs/GRADIO_NATIVE_HF.md) |
| 投机解码、`device_map`、DeepSpeed 多卡 | [高级使用教程](docs/ADVANCED_USAGE_ZH.md) |
| bitsandbytes W8/W4、原生 MM8/MM4 | [量化使用教程](docs/QUANTIZATION_USAGE.md) |
| 华为昇腾 NPU、NPUGraph 与 W8 | [华为昇腾使用与验收边界](docs/hardware/HUAWEI_ASCEND.md) |
| 壁仞 BR106M、BIRENSUPA | [壁仞 BR106M 使用与验收边界](docs/hardware/BIREN_BR106M.md) |
| 沐曦 MetaX C500、MXMACA | [MetaX C500 使用与验收边界](docs/hardware/METAX_C500.md) |
| Apple MPS、MLX、CoreML | [Apple 使用教程](docs/APPLE_USAGE.md) |
| RWKV-7 与 Qwen3.5 谁更快、如何复现 | [完整参数、速度对比及 GPU 实测复现](docs/QWEN35_SPEED_COMPARISON_ZH.md)（[English](docs/QWEN35_SPEED_COMPARISON.md)） |
| 让 AI 帮你安装、运行或排错 | [统一 AI 操作入口](docs/AI_ASSISTED_SETUP.md) |
| 选择显卡与后端配置 | [硬件矩阵](docs/HARDWARE_MATRIX.md) |
| 查看性能结果与复现命令 | [性能指南](docs/PERFORMANCE.md) 与 [benchmark 索引](bench/INDEX.md) |
| 在独立项目中实现 vLLM/SGLang 适配 | [Serving-engine 实现规范](docs/integrations/README.md) |

## 后端怎么选

- **NVIDIA CUDA：** 先使用自动选择。需要已验证的融合性能路线时，再按硬件矩阵
  启用对应环境变量。
- **华为昇腾：** 先安装与 CANN 精确匹配的 PyTorch/torch-npu，再从
  [Ascend 文档](docs/hardware/HUAWEI_ASCEND.md) 的 BF16 eager 验收开始；
  NPUGraph 和 W8 只继承精确 910B3/软件栈/模型/batch 的证据。
- **壁仞 BR106M：** 使用 `torch_br` 注册的 `supa` 设备，模型使用 BF16、
  recurrent state 使用 FP32，并保持 native eager；FP16、图执行、量化与性能
  结论需按 [BR106M 文档](docs/hardware/BIREN_BR106M.md) 的边界验收。
- **沐曦 MetaX C500：** 使用 MXMACA 提供的 `torch.cuda` 兼容接口，但不会
  继承 NVIDIA Ampere 算子策略；先按 [C500 文档](docs/hardware/METAX_C500.md)
  运行 native eager FP16 验收。
- **Apple Silicon：** 普通 Transformers 工作流使用 MPS；追求 Apple 原生性能时
  使用 MLX 教程。
- **CPU 或便携环境：** 原生后端可以完成转换、接口检查、真实小模型生成和无下载
  tiny 训练演示；从 [Windows 与 CPU 教程](docs/WINDOWS_CPU.md) 开始。
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
- [项目与架构总览](docs/PROJECT_SUMMARY.md)
- [跨平台结果与证据索引](docs/RESULTS_INDEX.md)
- [版本变更记录](CHANGELOG.md)
- [贡献归属](CONTRIBUTIONS.md)
- [贡献者名单](CONTRIBUTORS.md)
- [许可证](LICENSE)
