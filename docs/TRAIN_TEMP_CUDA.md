# train_temp CUDA 训练对齐教程

本教程把官方 RWKV-LM `train_temp` 训练数学接入 Hugging Face 模型表面，让 CUDA
用户可以继续使用标准模型目录和 API，同时获得与官方训练配方对齐的 Native/no-FLA
全参数训练路线。RTX 5090 上已经完成逐 tensor、真实 MiniPile 三 seed、5,000 步训练、
断点恢复和吞吐对比。普通 Trainer、PEFT 和 TRL 用法见
[`TRAINING_WORKFLOWS.md`](TRAINING_WORKFLOWS.md)。

## 1. 前置条件和支持环境

需要 Linux、NVIDIA CUDA、本地 CUDA toolkit 和 PyTorch。Native 训练后端不依赖
FLA：

```bash
python -m pip install -e ".[cuda,train]"
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())"
```

推荐运行配置：

- Ampere 或更新的 NVIDIA GPU，BF16；
- Native RWKV-7 HF 模型，K/V head size 为 64；
- dense batch，所有样本等长，不含 padding 或 `-100` label；
- `use_cache=False`，序列长度是 16 的倍数；
- 首次调用会用本机 toolkit 编译已固定来源的 CUDA 扩展。

RTX 5090 的 12x768/有效 FFN3072/T512 结果已经覆盖 B1 原始 HF 对齐，以及
Native/no-FLA B16 官方 shell 形状的逐 tensor、真实 MiniPile 三 seed、5,000-step
长程、断点恢复和显存稳定性。Ampere 及更新的 NVIDIA GPU 可以直接沿用同一接口；
记录本机 GPU 和运行结果后，也能方便地加入硬件矩阵。

## 2. 推荐起步配置

第一次运行建议使用已经转换的可信 HF 模型和一条不含 padding 的 16-token 输入，
确认扩展编译和反向传播正常后，再切换到官方 T512 配方或更长训练任务。

```bash
export MODEL=/path/to/converted-rwkv7-hf
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
```

`MODEL` 必须是完整 HF 模型目录，不是单个 `.pth` 文件。转换方法见
[`USER_GUIDE.md`](USER_GUIDE.md)。

## 3. 可直接复制的 API

### 可配置 Native 训练命令

普通用户可以直接使用 `scripts/train_native.py`。下面的命令读取 MiniPile 风格的
`DATA.bin` / `DATA.idx`，按指定 batch、ctx 和步数生成可复现的 packed sequence，
然后运行 Native/no-FLA `train_temp_cuda`：

```bash
python scripts/train_native.py \
  --model "$MODEL" \
  --dataset /path/to/minipile \
  --output-dir outputs/my-run \
  --batch-size 4 \
  --seq-length 1024 \
  --steps 10000 \
  --learning-rate 3e-4 \
  --learning-rate-final 3e-5 \
  --warmup-steps 100 \
  --checkpoint-every 500
```

`--seq-length` 可以选择任意 16 的倍数，实际上限由模型、显存和训练数据决定。命令会
生成 `resolved_config.json`、`result.json`、`checkpoint.pt`、训练/验证 loss 曲线和
显存记录。中断后追加下面的参数即可恢复：

```bash
python scripts/train_native.py \
  --config outputs/my-run/resolved_config.json \
  --resume-from outputs/my-run/checkpoint.pt
```

也可以从 [`train_native_example.json`](../configs/train_native_example.json) 开始，并用
CLI 覆盖其中任意软参数：

```bash
python scripts/train_native.py \
  --config configs/train_native_example.json \
  --model "$MODEL" --dataset /path/to/minipile --output-dir outputs/my-run \
  --batch-size 8 --steps 5000
```

需要复现官方示例参数时选择 preset；`--steps` 仍由用户决定：

```bash
python scripts/train_native.py \
  --preset official-x070-12x768-b16 \
  --model "$MODEL" --dataset /path/to/minipile \
  --output-dir outputs/official-shape --steps 1000
```

已有自己的 packed token 数据时，传入包含同形状 `input_ids` / `targets` 的
`[steps,batch,tokens]` 训练文件和 `[batch,tokens]` 验证文件：

```bash
python scripts/train_native.py \
  --model "$MODEL" --output-dir outputs/packed-run \
  --sequence train.safetensors --validation-batch validation.safetensors
```

先检查配置而不启动 CUDA 时添加 `--dry-run`；它不会扫描整份模型权重或生成 packed
数据。运行 `python scripts/train_native.py --help` 可以查看全部可调参数。参数解析顺序为
preset、JSON、CLI，后者优先。熟悉官方 shell 的用户也可以直接使用 `--micro-bsz`、
`--ctx-len`、`--lr-init`、`--lr-final` 和 `--max-steps` 别名；官方固定配置文件保持不变。

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

这个短例子会完成一次真实的 forward、loss、backward 和 optimizer step。要同时复现
官方优化器分组、FusedAdam、裁剪顺序和学习率 schedule，可以直接使用
`bench/bench_train_temp_alignment.py`。机器证据和 runner 顺序见
[`5090_train_temp_alignment_20260717`](../bench/5090_train_temp_alignment_20260717/README.md)。
该目录同时包含注明筛选口径的最佳观测配对主图、完整三-seed 收敛曲线 PNG、
单步对比 CSV 和 cohort CSV，可以直接在 GitHub 中查看或下载复核。

官方 shell 形状的 Native B16 证据、完整真实数据三 seed 图、5,000-step 官方对比图、
2,500+2,500 恢复哈希和稳态显存采样见
[`5090_native_train_temp_real_minipile_20260718`](../bench/5090_native_train_temp_real_minipile_20260718/README.md)。

### 官方 shell 运行方式

以下示例对应 RWKV-LM commit `e6f74b6` 中的
`RWKV-v7/train_temp/demo-training-prepare.sh` 和
`RWKV-v7/train_temp/demo-training-run.sh`。前者在 CPU 用 B1 创建初始化；后者在单卡
使用 B16、BF16、T512、有效 FFN3072、DeepSpeed ZeRO-2、`kernel=@rwkv3` 训练。按
prepare、run 的顺序执行即可得到与官方示例一致的初始化和训练配置。

这两个 `.sh` 文件位于官方 RWKV-LM checkout。进入对应的 `RWKV-v7/train_temp`
目录即可查看或运行，源码位置是：

```text
/path/to/RWKV-LM/RWKV-v7/train_temp/demo-training-prepare.sh
/path/to/RWKV-LM/RWKV-v7/train_temp/demo-training-run.sh
```

Windows/CPU 用户可以先运行 [`WINDOWS_CPU.md`](WINDOWS_CPU.md) 的 tiny 演示；
Linux、NVIDIA CUDA 和 DeepSpeed 环境可以继续使用官方 `run.sh`。本仓库提供下面的
runner，负责定位两条官方脚本并把输出写入单独目录。

runner 会读取对应 commit、脚本和 MiniPile 数据，并按 shell 中的参数启动训练。
`--max-steps 1` 表示这里只运行一个 optimizer step，便于先确认环境能够正常启动；
它不会修改官方脚本文件。

```bash
export OFFICIAL=/path/to/RWKV-LM
export DATA=/path/to/minipile
export OUT=/path/to/train-temp-output

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

三个命令分别生成 `verify.json`、`prepare.json`、`run.json` 和相应日志。训练中断时
保留输出目录与 extension cache，再从相同目录重新运行即可。

注意：pinned `train.py` 会在参数摘要中打印通用 3.5x 默认值
`dim_ffn=2688`，但生产 fast `RWKV_CMix_x070` 明确创建 4x 矩阵。实际模型使用
checkpoint 中的 FFN3072，日志里的通用字段不是有效模型宽度。

RTX 5090 的独立复测已经直接执行未修改的两条 shell 命令。`run` 仅通过临时 PATH
wrapper 追加 `--max_steps 1`，没有修改脚本内容。官方一轮报告
B16/BF16/T512、ZeRO-2、loss `11.20`；等价 Native runner 报告 loss
`11.249235`、`399/399` 有限 ZeRO 梯度、模型 hash 变化和 4,355.95 MiB 峰值。
脚本哈希、数据哈希、原始日志和结果见
[`5090_native_hf_gradio_train_temp_20260718`](../bench/5090_native_hf_gradio_train_temp_20260718/README.md)。

### Native 断点恢复

Native runner 会把模型、优化器、Python/NumPy/torch CPU/CUDA RNG、学习率进度和
曲线原子写入同一个 checkpoint。以下命令展示已验证的 2,500+2,500 恢复；
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

第一条命令运行到 2,500 步并保存 checkpoint，第二条恢复到 5,000 步。完成后
`resumed_from` 会显示 model、optimizer 和 RNG 三个 `*_restored=true`。runner 还会
自动核对 seed、序列、初始化、后端、优化器、schedule 和 gradient-checkpointing，
帮助避免恢复到不匹配的训练任务。

## 4. 已验证结果

普通 API 试跑正常完成时会看到：

1. 首次扩展编译退出码为 0，没有 missing operator；
2. `backend` 中 attention/FFN 数量相等且大于 0；
3. loss 有限，`finite_grad_tensors > 0`，优化器执行后至少一个参数变化；
4. 没有 padding、cache、dtype、head size 或序列长度合同错误。

RTX 5090 官方对齐结果覆盖三个层次：

- 单次 T512 反向和 FusedAdam step 的 tensor、loss、参数组和 post-step loss 对齐；
- 三组相同 seed/样本序列的长期训练、loss AUC、验证 loss 和梯度统计；
- 5,000-step 连续训练及 2,500+2,500 恢复的模型、优化器、RNG 和稳态显存记录。

RTX 5090 留存结果为反向 400/400、step 800 tensors 全部数值完全一致；三组
1,000-step 运行的成功数为官方 `2/3`、HF `2/3`，
[`compare_convergence_cohort.json`](../bench/5090_train_temp_alignment_20260717/compare_convergence_cohort.json)
状态为 `pass`。新的 Native B16/T512 路线为 399/399 梯度和 399/399 参数更新完全
一致；真实 MiniPile 的官方/Native 三个配对 seed 都是 `3/3` finite，最终验证 loss
都不高于 `4.8`，Native 中位吞吐为官方 `1.00049x`。连续 5,000-step Native/官方
吞吐比为 `1.00255x`，最终验证 loss 为 `3.80373/3.81245`；2,500+2,500 恢复会校验
模型、优化器和全部 RNG 摘要，并以 `0.99822x` 连续 Native 的累计吞吐完成。这组
结果可作为其他 Ampere+ 显卡复现同一配方时的直接参考。

## 5. 常见问题与扩展

- `CUDA_HOME` 或 `nvcc` 不存在：设置与 PyTorch CUDA 主版本兼容的 toolkit，删除
  这次失败的 torch extension build 目录后重试；不要删除模型或证据目录。
- 提示 BF16、head size 或 T 不符合：改用 BF16、head64 模型和 16 的倍数长度。
- 提示 padding/`-100`/cache：先把样本 pack 成等长 dense token，关闭 cache；
  当前后端不会静默忽略这些输入。
- 编译中断：保留 PyTorch extension cache，重新执行同一命令会继续复用已完成对象。
- 训练中断：保留同一 checkpoint、序列和输出目录，用 `--resume-from` 原地恢复；
  provenance 或 payload 哈希不一致时停止，不要换文件名绕过检查。
- 长程曲线与另一轮同 seed 不逐点相等：先比较单步 tensor，再查看完整三-seed
  cohort，CUDA 长程训练允许存在正常的运行间波动。

当前优化配置优先覆盖 BF16、dense batch、head64 和单卡 full-sequence CUDA 训练。
其他精度、padding、可变长度和多卡任务仍可使用普通 Native HF 训练路线，并可在此
基础上继续扩展。调用 `enable_train_temp_cuda_backend` 即可启用官方 full-sequence
CUDA 训练边界。

## 6. 让 AI 执行

AI 操作只从唯一入口 [`AI_ASSISTED_SETUP.md`](AI_ASSISTED_SETUP.md) 选择
`TASK_ID=train-temp-alignment`。不要在其他专题文档复制或改写提示词。
