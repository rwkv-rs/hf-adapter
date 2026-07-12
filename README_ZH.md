# RWKV-7 Hugging Face 参考实现

[English](README.md) | [中文](README_ZH.md)

1.0 是可阅读、可复现的纯 PyTorch RWKV-7 HF 实现。TMix、
CMix、残差、归一化、层循环、loss 与 cache 都能直接从
`rwkv7_hf/modeling_rwkv7.py` 看懂；WKV 递推只通过
`rwkv7_hf/ops_rwkv7.py` 这一处明确边界。`rwkv7_hf/` 只放模型代码，转换与
smoke 命令统一放在同级 `rwkv7_hf_tools/`。独立的 `rwkv7-kernels` 可选包通过
两个版本化算子边界提供完整 NVIDIA 高性能实现；安装它不会替换可读模型、
config、cache、tokenizer 或 checkpoint 布局。历史开发代码继续归档在
`perf/native-kernels-v0.8`，用户不需要从旧分支安装。

## 直接使用 HF 模型

```bash
python -m pip install "torch" "transformers>=4.48,<6"
```

请先安装与显卡匹配的 PyTorch。当前部分默认 CUDA 13 wheel 已不包含
`sm_70`，V100 应从 PyTorch 官方索引选择兼容的 CUDA 12.x wheel。PyTorch
已经存在时，再执行 `pip install rwkv7-hf==1.0.0` 不会替换它。

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "wangyue114514/rwkv7-g1d-0.1b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, torch_dtype=torch.float16
).cuda().eval()

inputs = tokenizer("User: Hello! Assistant:", return_tensors="pt").to("cuda")
with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=32)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

新版模型仓库自带 config、cache、PyTorch 算子、modeling、tokenizer、词表和
safetensors。普通加载不要求安装 `rwkv7-hf`、FLA、Triton、编译器或
kernel wheel。

## 安装可选 NVIDIA 高性能后端

先安装与目标显卡匹配的 PyTorch，再执行：

```bash
python -m pip install "rwkv7-hf==1.0.0" "rwkv7-kernels==1.0.0"
```

也可以写成一个等价依赖：
`python -m pip install "rwkv7-hf[kernels]==1.0.0"`。两种安装方式都不会改变
模型 API；卸载 `rwkv7-kernels` 后仍可直接走无需插件的参考实现。

模型调用方式完全不变。默认 `RWKV7_BACKEND=auto` 只选择已经通过发布矩阵的
设备、dtype 和 shape 路由，不支持时执行同一份 reference 主体。
`RWKV7_BACKEND=reference` 可关闭插件；`RWKV7_BACKEND=optimized` 是严格诊断
模式，遇到不支持的路由会直接报错，不会静默回退。

可选 wheel 负责 recurrent、融合 prefill/decode、CUDA Graph/state pool、
SM70/Ada/Blackwell 策略、量化适配和训练 autograd；它不会向 `RWKV7Config`
加入硬件字段，也不会改变 `RWKV7Cache` 的公开布局。W8/W4/A8W8、BN/TN、
BitsAndBytes、Marlin 和 TorchAO 仍由 `rwkv7_kernels.quantization` 显式选择。

## 转换官方 .pth

```bash
python -m pip install "torch"  # 先选择与 CUDA/显卡匹配的 wheel
python -m pip install "rwkv7-hf==1.0.0"
rwkv7-hf convert \
  --input /path/to/model.pth \
  --output ./rwkv7-model-hf \
  --vocab-file /path/to/rwkv_vocab_v20230424.txt \
  --precision fp16 \
  --low-memory
```

转换器始终输出完整、自包含的 reference 布局。

## 公开结构

- `RWKV7Config`，`model_type = "rwkv7"`
- `RWKV7Cache`：每层 `[B,H,K,V]` state、TMix shift、CMix shift
- `RWKV7TimeMix`、`RWKV7ChannelMix`、`RWKV7Block`
- `RWKV7PreTrainedModel`、`RWKV7Model`、`RWKV7ForCausalLM`
- 标准 loss、cache、generation、save/reload、gradient checkpointing、PEFT、
  Trainer 与 TRL 接口

公开 API 只使用规范的 `RWKV7*` 类名。

## 源码包

- `rwkv7_hf/`：只包含 config、cache、算子边界、modeling、tokenizer 和聊天模板。
- `rwkv7_hf_tools/`：包含 CLI、checkpoint 转换器、manifest 工具和模型 smoke 验证。
- `kernels/rwkv7_kernels/`：只包含可选版本化协议、NVIDIA 实现、Graph/state
  pool、量化器和训练算子。

- [架构](docs/ARCHITECTURE.md)
- [转换](docs/CONVERSION.md)
- [评测](docs/EVALUATION.md)
- [SFT / DPO / GRPO](docs/FINETUNING.md)
- [可复现产物](docs/REPRODUCIBILITY.md)
- [NVIDIA 迁移与能力审计](docs/NVIDIA_MIGRATION_AUDIT.md)
- [模型列表](docs/PUBLISHED_MODELS.md)
