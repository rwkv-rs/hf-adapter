# Windows 与 CPU 推理/微型训练

本教程提供一条不需要 NVIDIA GPU、CUDA、FLA 或模型下载的最小路线。它使用
RWKV-7 Native eager 后端和 CPU FP32，依次验证标准 `generate(use_cache=True)`、
反向传播、参数更新以及 SafeTensors 保存重载。Windows 用户可以直接运行
PowerShell 脚本；Linux 和 macOS 用户可以运行同一个 Python 示例。

## 1. 前置条件和支持环境

- Windows 10/11、PowerShell 5.1 或更高版本、64 位 Python 3.10 或更高版本；
- 推荐至少 2 GB 可用磁盘空间，用于虚拟环境和 PyTorch CPU wheel；
- Linux 和 macOS 也可使用下面的直接 Python 命令；
- 本示例不需要 CUDA、FLA、bitsandbytes 或 Hugging Face token。

一键脚本会把依赖安装在仓库内的 `.venv-cpu-demo`，不会修改系统 Python。已有
可用环境时可以不安装，直接运行示例。

## 2. 最小安全模型和输入

默认示例在内存中创建一个随机初始化的两层 RWKV-7：`hidden_size=16`、
`intermediate_size=32`、`vocab_size=32`。训练数据是固定生成的整数序列，默认
batch 为 4、序列长度为 16、训练 12 步。整个流程只使用 CPU FP32。

这是接口、梯度、参数更新和保存重载的可执行演示。随机 tiny 模型只输出 token
ID，不会生成有意义的自然语言；短 smoke 也不代表模型质量、长期收敛、训练容量或
CPU 性能。

## 3. 可直接复制的命令和 API

### Windows 一键安装并运行

在仓库根目录打开 PowerShell：

一键入口文件是 `scripts/run_cpu_demo.ps1`。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\run_cpu_demo.ps1 -Install
```

已经安装过本仓库和 PyTorch 时：

```powershell
.\scripts\run_cpu_demo.ps1
```

只运行推理或训练：

```powershell
.\scripts\run_cpu_demo.ps1 -Mode infer
.\scripts\run_cpu_demo.ps1 -Mode train -Steps 12
```

### Linux、macOS 或已有 Python 环境

```bash
python examples/cpu_tiny_demo.py --mode all --steps 12 --threads 4
```

需要保留训练后 tiny checkpoint 时：

```powershell
python examples/cpu_tiny_demo.py --mode all --output-dir artifacts\cpu-tiny-demo
```

### 使用真实的已转换模型推理

先从 0.1B 模型开始，并明确指定 CPU FP32：

```powershell
python examples/generate.py --model D:\models\rwkv7-model-hf --prompt "User: Hello. Assistant:" --device cpu --dtype fp32 --max-new-tokens 8
```

标准 Transformers API 如下：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = r"D:\models\rwkv7-model-hf"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    dtype=torch.float32,
).eval().to("cpu")

inputs = tokenizer("User: Hello. Assistant:", return_tensors="pt")
output = model.generate(
    **inputs,
    max_new_tokens=8,
    do_sample=False,
    use_cache=True,
    pad_token_id=tokenizer.pad_token_id,
)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

只对自己信任的本地目录或 Hugging Face 仓库使用 `trust_remote_code=True`。

## 4. 精确且可观察的通过标准

命令退出码必须为 `0`，并且终端依次出现：

```text
CPU INFERENCE PASS
CPU TRAINING PASS
CPU SAVE/RELOAD PASS
CPU_DEMO_RESULT={...}
CPU DEMO PASS
```

`CPU_DEMO_RESULT` 中还必须满足：

- `device` 为 `cpu`、`dtype` 为 `float32`、`backend` 为 `native_eager`；
- 生成 token 数等于 `--max-new-tokens`；
- `final_loss < initial_loss`；
- `max_grad_l1 > 0` 且 `parameter_changed_l1 > 0`；
- `save_reload.max_abs == 0.0`。

不同 CPU 和 PyTorch 版本的具体 loss、耗时和 token ID 可以不同，不应把某一台机器
的精确数值写成门槛。

## 5. 失败恢复方法和当前限制

- PowerShell 报脚本执行被禁用时，使用上面的
  `powershell -ExecutionPolicy Bypass -File ...` 命令；它只对这次进程生效。
- 找不到 `python` 时，安装 64 位 Python 3.10+，重新打开 PowerShell，并运行
  `python --version` 检查 PATH。
- 缺少 `torch`、`transformers` 或 `rwkv7_hf` 时，重新加 `-Install` 运行一键脚本。
- 安装中断后可再次运行同一条 `-Install` 命令；pip 会复用已经下载的文件。
- 真实模型内存不足时先使用 0.1B、把 `--max-new-tokens` 降到 8，并关闭其他占用
  内存的程序。CPU 训练真实大模型会很慢且需要大量 RAM。
- 需要重建演示环境时，可以在确认没有重要文件后删除 `.venv-cpu-demo` 再运行
  `-Install`。训练 checkpoint 只在显式传入 `--output-dir` 时保留。

本教程没有证明 CPU 上的生产吞吐、长时间训练稳定性、模型质量或全模型 W8/W4
加速；它验证的是最小 Native HF 接口和参数更新合同。

### 与官方 train_temp 脚本的区别

本页的 CPU tiny 训练是 Windows 也能运行的最小演示。维护者提到的
`demo-training-prepare.sh` 和 `demo-training-run.sh` 属于官方 RWKV-LM 仓库的
`RWKV-v7/train_temp/`，不是本仓库的 Windows CPU 命令：前者只在 CPU 上创建初始
checkpoint，后者使用 Linux、NVIDIA CUDA 和 DeepSpeed 运行正式训练。配置含义、
固定版本和安全复现命令见 [`TRAIN_TEMP_CUDA.md`](TRAIN_TEMP_CUDA.md)。

## 6. 让 AI 执行

AI 的唯一操作入口是 [`AI_ASSISTED_SETUP.md`](AI_ASSISTED_SETUP.md)。在其中选择
`windows-cpu` 任务，并让 AI 读取本页后执行。AI 会返回完整命令、退出码和验收结果；
本页不复制第二套 AI 指令。
