# RWKV-7 Hugging Face

[English](README.md) | [发布状态与待办](HF_STATUS.md)

**普通用户用已发布的 0.9.0。当前 main 的 1.0 是冻结候选版，尚未完成正式发布。**
源码合并、PyPI 发包、HF 模型仓库更新是三个不同步骤。

## 现在就能使用：稳定版

先安装适合自己设备的 PyTorch，再安装：

```bash
python -m pip install "rwkv7-hf==0.9.0"
rwkv7-hf --version
```

从 Hub 使用已发布的参考模型时，并不必须安装 `rwkv7-hf`；它主要提供转换和
CLI。模型加载需要 Torch、Transformers 及其依赖，不需要 FLA 或可选 kernel 包。
为避免随 Hub `main` 变化，tokenizer 和 model 固定同一个发布 tag：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "wangyue114514/rwkv7-g1d-0.1b-hf"
revision = "v0.9.0"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

tokenizer = AutoTokenizer.from_pretrained(
    model_id, revision=revision, trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    model_id, revision=revision, trust_remote_code=True, torch_dtype=dtype
).to(device).eval()

inputs = tokenizer("User: Hello! Assistant:", return_tensors="pt").to(device)
with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=32)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

其他尺寸见[模型列表](docs/PUBLISHED_MODELS.md)。CPU 路径是便于使用的参考实现，
不代表具有 GPU 后端的速度。

### 转换自己的 checkpoint

```bash
rwkv7-hf convert \
  --input /absolute/path/model.pth \
  --output /absolute/path/model-hf \
  --vocab-file /absolute/path/rwkv_vocab_v20230424.txt \
  --precision fp16 \
  --low-memory
```

请把路径换成真实文件。无需寻找源码目录里的转换脚本；统一入口就是
`rwkv7-hf convert`。0.9.0 与 1.0 候选版的模型代码/目录细节不同，见
[转换说明](docs/CONVERSION.md)。

## 开发者：1.0 候选版与可选后端

候选版的模型结构保持纯 PyTorch、可阅读、可替换后端：

```text
rwkv7_hf/          # config、cache、ops、modeling、tokenizer、初始化与聊天模板
rwkv7_hf_tools/    # CLI、转换、manifest、smoke
kernels/           # 独立 rwkv7-kernels 包：可选硬件后端
examples/finetune/ # SFT、DPO、GRPO
evaluation/        # 数值、HF 生态、正式 lm_eval 与发布审计
```

`modeling_rwkv7.py` 中可以直接看到 TMix、CMix、残差、归一化、层循环和 loss。
HF cache 使用 `[B,H,K,V]`；后端只通过 `rwkv7_kernels.execute_optional_v4`
接入，不替换 HF 模型类、config、tokenizer 或公开 cache。

源码安装候选版的步骤和固定 SHA 见 [HF_STATUS.md](HF_STATUS.md#inspect-or-reproduce-the-10-candidate)。
**现在不要把 `pip install rwkv7-hf==1.0.0` 当成已发布命令。** 源码安装也不能冒充
正式评测使用的不可变 wheel；旧版 Hub 模型不会仅因安装插件就自动变成 API-v4 模型。

后端模式：

| 模式 | 行为 |
|---|---|
| `RWKV7_BACKEND=reference` | 不使用插件，走 PyTorch 参考实现 |
| `RWKV7_BACKEND=auto` | 能力检查接受才用后端，否则走参考实现 |
| `RWKV7_BACKEND=optimized` | 严格诊断；不支持的请求报错，不静默回退 |

已经开始更新 cache 的后端执行若失败，不会再回退重算。训练保留可读 HF 层循环；
加速叶子是否执行取决于形状、dtype 和 autograd 条件。
**能运行 SFT/DPO/GRPO，不等于全部训练都经过高性能算子。**
详见[能力矩阵](docs/NVIDIA_MIGRATION_AUDIT.md#user-facing-capability-matrix)。

## 当前验证状态

9 月 1 日按用户要求停止时：Optimized 47 项成功、1 项 OOM；FLA 48 项成功；
Reference 43 项成功、1 项中止、4 项未开始。不是三路 144 项全部通过。
9 月 8 日已按用户要求继续验收：本地完整测试 470/470、上游 main 的 CPU/HF CI
通过；4080 离线、4090 SSH 超时，尚未恢复 GPU 任务。停止记录、证据限制和剩余发布条件见
[状态页](HF_STATUS.md#paused-formal-evaluation)。

## 文档入口

- [架构](docs/ARCHITECTURE.md) / [插件接口](docs/KERNEL_PLUGIN_API.md)
- [转换](docs/CONVERSION.md) / [模型列表](docs/PUBLISHED_MODELS.md)
- [评测](docs/EVALUATION.md) / [SFT、DPO、GRPO](docs/FINETUNING.md)
- [复现与发布](docs/REPRODUCIBILITY.md)
- [NVIDIA 迁移与能力矩阵](docs/NVIDIA_MIGRATION_AUDIT.md)

模型核心和发行输入继续冻结，不为了整理文档增加桥接模块、重复模型类或新 cache。
