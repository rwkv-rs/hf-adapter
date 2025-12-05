# train_temp CUDA 训练对齐教程

本教程用于在 Hugging Face 模型表面下启用官方 RWKV-LM `train_temp` 数学边界，
适合需要复现官方全参数训练行为的 CUDA 用户。它是**显式启用**的训练后端；普通
Trainer、PEFT 和 TRL 教程继续使用 [`TRAINING_WORKFLOWS.md`](TRAINING_WORKFLOWS.md)。

## 1. 前置条件和支持环境

需要 Linux、NVIDIA CUDA、本地 CUDA toolkit、PyTorch 和 FLA：

```bash
python -m pip install -e ".[cuda,train]"
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())"
```

当前运行合同是：

- Ampere 或更新的 NVIDIA GPU，BF16；
- FLA-backed RWKV-7 HF 模型，K/V head size 为 64；
- dense batch，所有样本等长，不含 padding 或 `-100` label；
- `use_cache=False`，序列长度是 16 的倍数；
- 首次调用会用本机 toolkit 编译已固定来源的 CUDA 扩展。

精确生产证据来自 RTX 5090、batch 1、T512、12x768、191M 参数配置。其他 Ampere+
设备可以按同一接口试跑，但必须通过自己的精确卡验收后才能继承生产结论。

## 2. 最小安全输入

第一次运行请使用已经转换的可信 HF 模型和一条不含 padding 的 16-token 输入。
开始正式验收时使用 T512；不要直接从大模型、动态 padding 或多卡启动。

```bash
export MODEL=/path/to/converted-rwkv7-hf
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
```

`MODEL` 必须是完整 HF 模型目录，不是单个 `.pth` 文件。转换方法见
[`USER_GUIDE.md`](USER_GUIDE.md)。

## 3. 可直接复制的 API

下面是一轮完整的 dense causal-LM 更新。`train_temp_causal_cross_entropy`
会自动完成 next-token shift，不需要手工切 labels：

```python
import os
import torch
from transformers import AutoModelForCausalLM

from rwkv7_hf.train_temp_cuda import (
    enable_train_temp_cuda_backend,
    train_temp_causal_cross_entropy,
)

model_path = os.environ["MODEL"]
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    dtype=torch.bfloat16,
).cuda().train()
model.config.use_cache = False

backend = enable_train_temp_cuda_backend(model)
input_ids = torch.randint(
    0, model.config.vocab_size, (1, 512), device="cuda", dtype=torch.long
)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
optimizer.zero_grad(set_to_none=True)
outputs = model(input_ids=input_ids, use_cache=False, return_dict=True)
loss = train_temp_causal_cross_entropy(outputs.logits, input_ids)
loss.backward()
optimizer.step()

grad_count = sum(
    p.grad is not None and bool(torch.isfinite(p.grad).all())
    for p in model.parameters()
)
print({"backend": backend, "loss": float(loss), "finite_grad_tensors": grad_count})
```

这个短例子证明后端可用，但不会自动复现官方优化器分组、FusedAdam、裁剪顺序和
学习率 schedule。需要严格对齐时使用 `bench/bench_train_temp_alignment.py`；
机器证据和 runner 顺序见
[`5090_train_temp_alignment_20260717`](../bench/5090_train_temp_alignment_20260717/README.md)。
该目录同时包含官方/HF 收敛曲线 PNG、单步对比 CSV 和 cohort CSV，便于直接在
GitHub PR 中查看或下载复核。

## 4. 精确通过标准

普通 API 试跑必须同时满足：

1. 首次扩展编译退出码为 0，没有 missing operator；
2. `backend` 中 attention/FFN 数量相等且大于 0；
3. loss 有限，`finite_grad_tensors > 0`，优化器执行后至少一个参数变化；
4. 没有 padding、cache、dtype、head size 或序列长度合同错误。

官方训练效果验收使用更严格的两层门禁：

- 单次 T512 反向和 FusedAdam step 必须逐张量通过 cosine、relative-L2、loss、
  参数组和 post-step loss 比较；
- 长期训练至少三组相同 seed/样本序列，全部 finite/complete，并通过 cohort 的
  成功率、loss AUC、中位最小验证 loss 和梯度比例门禁。

RTX 5090 留存结果为反向 400/400、step 800 tensors 全部数值完全一致；三组
1,000-step 运行的成功数为官方 `2/3`、HF `2/3`，
[`compare_convergence_cohort.json`](../bench/5090_train_temp_alignment_20260717/compare_convergence_cohort.json)
状态为 `pass`。

## 5. 失败恢复和当前限制

- `CUDA_HOME` 或 `nvcc` 不存在：设置与 PyTorch CUDA 主版本兼容的 toolkit，删除
  这次失败的 torch extension build 目录后重试；不要删除模型或证据目录。
- 提示 BF16、head size 或 T 不符合：改用 BF16、head64 模型和 16 的倍数长度。
- 提示 padding/`-100`/cache：先把样本 pack 成等长 dense token，关闭 cache；
  当前后端不会静默忽略这些输入。
- 编译中断：保留 PyTorch extension cache，重新执行同一命令会继续复用已完成对象。
- 长程曲线与另一轮同 seed 不逐点相等：先检查单步严格门禁；长期结果必须重新收集
  完整三-seed cohort，不能挑选单条最好曲线。

当前不把该后端描述为 FP16/FP32、padding、可变长度、ZeRO、多卡或所有 GPU 的
生产实现。默认 HF/FLA 行为没有改变，调用 `enable_train_temp_cuda_backend` 才会启用。

## 6. 让 AI 执行

AI 操作只从唯一入口 [`AI_ASSISTED_SETUP.md`](AI_ASSISTED_SETUP.md) 选择
`TASK_ID=train-temp-alignment`。不要在其他专题文档复制或改写提示词。
