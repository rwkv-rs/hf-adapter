# train_temp CUDA 训练对齐教程

本教程用于在 Hugging Face 模型表面下启用官方 RWKV-LM `train_temp` 数学边界，
适合需要复现官方全参数训练行为的 CUDA 用户。它是**显式启用**的训练后端；普通
Trainer、PEFT 和 TRL 教程继续使用 [`TRAINING_WORKFLOWS.md`](TRAINING_WORKFLOWS.md)。

## 1. 前置条件和支持环境

需要 Linux、NVIDIA CUDA、本地 CUDA toolkit 和 PyTorch。Native 训练后端不依赖
FLA：

```bash
python -m pip install -e ".[cuda,train]"
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())"
```

当前运行合同是：

- Ampere 或更新的 NVIDIA GPU，BF16；
- Native RWKV-7 HF 模型，K/V head size 为 64；
- dense batch，所有样本等长，不含 padding 或 `-100` label；
- `use_cache=False`，序列长度是 16 的倍数；
- 首次调用会用本机 toolkit 编译已固定来源的 CUDA 扩展。

精确证据来自 RTX 5090 的两条 12x768/有效 FFN3072/T512 路线：B1 用于原始 HF
对齐，Native/no-FLA B16 用于官方 shell 形状的逐 tensor、真实 MiniPile 三 seed、
5,000-step 长程、断点恢复和显存稳定性验收。其他 Ampere+ 设备可以按同一接口试跑，
但必须通过自己的精确卡验收后才能继承结论。

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
该目录同时包含注明筛选口径的最佳观测配对主图、完整三-seed 收敛曲线 PNG、
单步对比 CSV 和 cohort CSV，便于直接在 GitHub PR 中查看或下载复核。主图不替代
完整 cohort，正式验收仍以三-seed 报告为准。

官方 shell 形状的 Native B16 证据、完整真实数据三 seed 图、5,000-step 官方对比图、
2,500+2,500 恢复哈希和稳态显存采样见
[`5090_native_train_temp_real_minipile_20260718`](../bench/5090_native_train_temp_real_minipile_20260718/README.md)。

### 官方 shell 配方门槛

官方标准固定为 RWKV-LM commit `e6f74b6` 的
`RWKV-v7/train_temp/demo-training-prepare.sh` 和
`RWKV-v7/train_temp/demo-training-run.sh`。前者在 CPU 用 B1 创建初始化；后者在单卡
使用 B16、BF16、T512、有效 FFN3072、DeepSpeed ZeRO-2、`kernel=@rwkv3` 训练。两条脚本
不是同一个阶段，不能用 prepare 的 B1 替代 run 的 B16。

下面的安全 runner 会先核对 commit、脚本 SHA256、Minipile 大小/哈希和
`magic_prime`，把输出限制在独立目录，并在不执行官方清理命令的情况下复现脚本
参数。`--max-steps 1` 只给正式训练增加可审计的有界停止点：

```bash
export OFFICIAL=/path/to/RWKV-LM
export DATA=/path/to/minipile
export OUT=/path/to/isolated/train-temp-acceptance

python scripts/run_train_temp_official_recipe.py verify \
  --official-checkout "$OFFICIAL" --data-prefix "$DATA" \
  --output-dir "$OUT/out" --artifact "$OUT/verify.json"

python scripts/run_train_temp_official_recipe.py prepare \
  --official-checkout "$OFFICIAL" --data-prefix "$DATA" \
  --output-dir "$OUT/out" --artifact "$OUT/prepare.json" \
  --log "$OUT/prepare.log"

python scripts/run_train_temp_official_recipe.py run \
  --official-checkout "$OFFICIAL" --data-prefix "$DATA" \
  --output-dir "$OUT/out" --artifact "$OUT/run.json" \
  --log "$OUT/run.log" --max-steps 1
```

通过标准是三个 JSON 均为 `status=pass`、两个执行阶段 `exit_code=0`、输出目录出现
`rwkv-init.pth`，且 checkpoint 中 FFN key/value 形状为 `3072x768` / `768x3072`，
run 日志显示 B16/BF16/T512、单 GPU、`deepspeed_stage_2` 和至少一个有限 loss。
中断后保留相同输出和 extension cache
原地重跑；若脚本哈希或数据哈希变化则停止，不能把不同配方结果合并。

注意：pinned `train.py` 会在参数摘要中打印通用 3.5x 默认值
`dim_ffn=2688`，但生产 fast `RWKV_CMix_x070` 明确创建 4x 矩阵。验收以实际源码和
checkpoint 形状 FFN3072 为准，不能把日志字段误写成有效模型宽度。

RTX 5090 的独立复测已经直接执行未修改的两条 shell 命令。`run` 仅通过临时 PATH
wrapper 追加 `--max_steps 1` 形成有界验收，没有修改脚本内容。官方一轮报告
B16/BF16/T512、ZeRO-2、loss `11.20`；等价 Native runner 报告 loss
`11.249235`、`399/399` 有限 ZeRO 梯度、模型 hash 变化和 4,355.95 MiB 峰值。
脚本哈希、数据哈希、原始日志和结果见
[`5090_native_hf_gradio_train_temp_20260718`](../bench/5090_native_hf_gradio_train_temp_20260718/README.md)。

### Native 断点恢复

严格 runner 会把模型、优化器、Python/NumPy/torch CPU/CUDA RNG、学习率进度和
曲线原子写入同一个 checkpoint。以下命令展示生产证据使用的 2,500+2,500 恢复；
`SEQ` 必须包含 5,000 步，`INIT_SHA` 是转换前官方初始化的 SHA256：

```bash
export SEQ=/path/to/sequence-5000.safetensors
export VAL=/path/to/validation.safetensors
export CKPT=/path/to/native-resume.pt
export INIT_SHA=5fcb1f16231626f0fde51c30c2d51994ef1ec80e6f737735afe83093c253b943

COMMON=(
  --model "$MODEL" --checkpoint-sha256 "$INIT_SHA" --native
  --train-temp-cuda --gradient-checkpointing --sequence "$SEQ"
  --validation-batch "$VAL" --seed 131 --precision bf16 --device cuda
  --learning-rate 0.0006 --learning-rate-final 0.00006
  --schedule-total-steps 182888 --warmup-steps 10 --weight-decay 0.001
  --beta1 0.9 --beta2 0.99 --adam-eps 1e-18 --grad-clip 1.0
  --eval-interval 50 --optimizer fused_adam
)

python bench/bench_train_temp_alignment.py converge-hf "${COMMON[@]}" \
  --output-json partial.json --checkpoint-out "$CKPT" \
  --checkpoint-every 500 --stop-after-step 2500

python bench/bench_train_temp_alignment.py converge-hf "${COMMON[@]}" \
  --output-json resumed.json --resume-from "$CKPT" \
  --checkpoint-out "$CKPT" --checkpoint-every 500
```

第一条命令的可观察通过标准是 `status=partial`、`failure=null`、
`steps_completed=2500`；第二条必须为 `status=pass`、`steps_completed=5000`，且
`resumed_from` 中 model/optimizer/RNG 三个 `*_restored` 都为 `true`。runner 会拒绝
不同 seed、序列、初始化、后端、优化器、schedule 或 gradient-checkpointing 的
checkpoint，不能强行合并。

## 4. 精确通过标准

普通 API 试跑必须同时满足：

1. 首次扩展编译退出码为 0，没有 missing operator；
2. `backend` 中 attention/FFN 数量相等且大于 0；
3. loss 有限，`finite_grad_tensors > 0`，优化器执行后至少一个参数变化；
4. 没有 padding、cache、dtype、head size 或序列长度合同错误。

官方训练效果验收使用更严格的三层门禁：

- 单次 T512 反向和 FusedAdam step 必须逐张量通过 cosine、relative-L2、loss、
  参数组和 post-step loss 比较；
- 长期训练至少三组相同 seed/样本序列，全部 finite/complete，并通过 cohort 的
  成功率、loss AUC、中位最小验证 loss 和梯度比例门禁；
- 至少一条长程 Native 运行必须验证断点前后模型/优化器/RNG 哈希，并记录多个稳态
  显存采样点，不能只看最终一步没有 NaN。

RTX 5090 留存结果为反向 400/400、step 800 tensors 全部数值完全一致；三组
1,000-step 运行的成功数为官方 `2/3`、HF `2/3`，
[`compare_convergence_cohort.json`](../bench/5090_train_temp_alignment_20260717/compare_convergence_cohort.json)
状态为 `pass`。新的 Native B16/T512 路线为 399/399 梯度和 399/399 参数更新完全
一致；真实 MiniPile 的官方/Native 三个配对 seed 都是 `3/3` finite，最终验证 loss
都不高于 `4.8`，Native 中位吞吐为官方 `1.00049x`。连续 5,000-step Native/官方
吞吐比为 `1.00255x`，最终验证 loss 为 `3.80373/3.81245`；2,500+2,500 恢复会校验
模型、优化器和全部 RNG 摘要，并以 `0.99822x` 连续 Native 的累计吞吐完成。当前
结论只覆盖这张 RTX 5090、这个 12x768/BF16/B16/T512 配方。

## 5. 失败恢复和当前限制

- `CUDA_HOME` 或 `nvcc` 不存在：设置与 PyTorch CUDA 主版本兼容的 toolkit，删除
  这次失败的 torch extension build 目录后重试；不要删除模型或证据目录。
- 提示 BF16、head size 或 T 不符合：改用 BF16、head64 模型和 16 的倍数长度。
- 提示 padding/`-100`/cache：先把样本 pack 成等长 dense token，关闭 cache；
  当前后端不会静默忽略这些输入。
- 编译中断：保留 PyTorch extension cache，重新执行同一命令会继续复用已完成对象。
- 训练中断：保留同一 checkpoint、序列和输出目录，用 `--resume-from` 原地恢复；
  provenance 或 payload 哈希不一致时停止，不要换文件名绕过检查。
- 长程曲线与另一轮同 seed 不逐点相等：先检查单步严格门禁；长期结果必须重新收集
  完整三-seed cohort，不能挑选单条最好曲线。

当前不把该后端描述为 FP16/FP32、padding、可变长度、多卡、其他模型宽度或所有
GPU 的生产实现。
普通 HF 运行默认使用 Native 模型；调用 `enable_train_temp_cuda_backend` 才会切换到
官方 full-sequence CUDA 训练边界。

## 6. 让 AI 执行

AI 操作只从唯一入口 [`AI_ASSISTED_SETUP.md`](AI_ASSISTED_SETUP.md) 选择
`TASK_ID=train-temp-alignment`。不要在其他专题文档复制或改写提示词。
