# RWKV-7 投机解码、训练与多卡图文指南

本文从 [`USER_GUIDE_ZH.md`](USER_GUIDE_ZH.md) 的首次生成已经成功开始，提供
投机解码、单卡训练、多卡推理和 DeepSpeed 多卡训练的可复制命令与验收标准。
英文版见 [`ADVANCED_USAGE.md`](ADVANCED_USAGE.md)。

转换/缓存、完整训练生态、量化和 Apple 教程统一从
[`COMPLETE_ADAPTER_GUIDE.md`](COMPLETE_ADAPTER_GUIDE.md) 进入。

这些命令主要是短 smoke：通过只能证明当前路径在本机完成了指定操作，不能自动证明
加速、生产训练收敛、原生张量并行或长期稳定性。

## 共同准备

第一次仍然使用转换好的 0.1B 或 0.4B 模型。激活仓库 `.venv` 后检查模型和显卡：

```bash
python examples/check_environment.py --model /path/to/model-hf
python -c "import torch; print(torch.__version__, torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
```

只有训练和多卡训练需要安装训练依赖：

```bash
python -m pip install -e ".[train]"
```

## 1. 投机解码

投机解码先让较小的 draft 模型提出一组 token，再由 target 模型验证；不匹配时由
target 纠正。贪心输出必须与 target 普通生成完全一致。

先用 target 自己同时充当 draft，验证 API 和正确性：

```bash
python tests/test_speculative_decode.py \
  --model /path/to/target-model-hf \
  --device cuda \
  --dtype fp16 \
  --max-new-tokens 8 \
  --draft-tokens 4
```

Windows PowerShell 可以把命令写成一行：

```powershell
python tests\test_speculative_decode.py --model C:\path\to\target-model-hf --device cuda --dtype fp16 --max-new-tokens 8 --draft-tokens 4
```

成功时应打印 `speculative_stats`、生成文本和 `PASS`；同模型 smoke 还应满足
`acceptance_rate=1.0` 且没有 correction。完成正确性检查后，再运行配对 benchmark
确认实际加速效果。

再换成同一词表、同一适配器协议的较小 RWKV-7 draft 模型：

```bash
python tests/test_speculative_decode.py \
  --model /path/to/target-model-hf \
  --draft-model /path/to/smaller-draft-model-hf \
  --device cuda \
  --dtype fp16 \
  --max-new-tokens 32 \
  --draft-tokens 4
```

较小 draft 仍需输出与 target 贪心生成一致。要声明加速，还必须在同一显卡、同一
prompt 和生成长度上，与 target 普通生成进行配对计时。

### 可选：对齐较小 draft

准备 UTF-8 文本，每行一个代表性 prompt。冻结 target，只训练较小 draft，并保存
合并后的 draft：

```bash
python scripts/train_spec_draft.py \
  --target /path/to/target-model-hf \
  --draft /path/to/smaller-draft-model-hf \
  --prompts /path/to/prompts.txt \
  --output /path/to/aligned-draft-hf \
  --device cuda --dtype fp16 --epochs 1 --gen-tokens 64
```

训练必须退出 0、打印有限 loss 遥测和 `saved_aligned_draft`。这只是训练产物
追踪，不是投机解码验收。继续与普通 target generation 配对测试：

```bash
python bench/bench_speculative_decode.py \
  --target-model /path/to/target-model-hf \
  --draft-model /path/to/aligned-draft-hf \
  --draft-tag trained --device cuda --dtype fp16 \
  --max-new-tokens 32 --draft-tokens 4
```

只有 `status: pass`、与 target greedy 完全一致且配对
`speedup_vs_target_generate > 1` 时，才能写训练 draft 带来加速。保留原始
off-the-shelf draft 作为 A/B 对照。

## 2. 单卡 PEFT LoRA 和 Trainer

先跑短 backward，再接真实数据。这能用较低成本发现后端不支持、梯度为零和显存不足。

运行 LoRA backward smoke：

```bash
python tests/test_peft_lora.py \
  --model /path/to/model-hf \
  --device cuda \
  --attn-mode fused_recurrent
```

成功要求 loss 有限、`nonzero_grad_count` 大于 0，且退出码为 0。FLA backward
不可用时，直接验证 native Trainer 路径：

```bash
python tests/test_native_trainer_smoke.py \
  --model /path/to/model-hf \
  --dtype fp32 \
  --max-steps 2 \
  --batch-size 2 \
  --length 32
```

成功时打印 `NATIVE TRAINER PASS`，短 loss 历史下降，并且至少一个可训练参数发生
更新。显存不足先换 0.1B，或降低 batch size 和 length，不要直接跳过验收。

这些 smoke 使用固定小样本，不会产出可用的生产 adapter。正式训练还需要：审核后的
数据集、训练/验证切分、输出目录、定期保存、断点恢复、评估指标和 loss 日志。
已验证的 Trainer、PEFT、TRL、ZeRO 范围见 [`TRAINING.md`](TRAINING.md)。

## 3. 多卡推理：HF `device_map`

仓库当前提供的是 HF 分层放置 / pipeline 方向 smoke。模型层分布到两张可见 CUDA
显卡，并可与单卡输出对比。

Linux 或 WSL2：

```bash
CUDA_VISIBLE_DEVICES=0,1 python tests/test_device_map_generate.py \
  --model /path/to/model-hf \
  --dtype fp16 \
  --attn-mode fused_recurrent \
  --max-new-tokens 4 \
  --compare-single-device
```

Windows PowerShell：

```powershell
$env:CUDA_VISIBLE_DEVICES="0,1"
python tests\test_device_map_generate.py --model C:\path\to\model-hf --dtype fp16 --attn-mode fused_recurrent --max-new-tokens 4 --compare-single-device
```

成功时打印 `PASS`，表示测试模型的 HF 分层放置和输出一致。该路线按层分配模型；
性能评估请同时记录单卡参考和跨卡传递开销，小模型通常优先使用单卡。

## 4. 多卡训练：DeepSpeed ZeRO-2/3

ZeRO-2 切分 optimizer state 和 gradient，ZeRO-3 进一步切分 parameter。该流程请在
Linux 或 WSL2、至少两张可见 CUDA 显卡上运行。

先检查仓库配置文件：

```bash
python tests/test_deepspeed_configs.py
```

再运行 ZeRO-2 和 ZeRO-3 各一步 smoke：

```bash
NPROC_PER_NODE=2 \
ZERO_STAGE=both \
MODEL=/path/to/model-hf \
TRAIN_DTYPE=fp16 \
RESULTS=bench/results.jsonl \
bash scripts/run_zero_training_smoke.sh
```

成功要求退出码为 0、请求的 stage 均有 `PASS`，并在 `bench/results.jsonl` 写入结果。
ZeRO 是训练状态切分，不是多卡推理 tensor parallel。一步 smoke 也不能证明长期收敛、
checkpoint 连续性或 optimizer/scheduler/RNG 完整恢复。

## 5. 交给 AI 执行

需要 AI 协助时，请打开 [`AI_ASSISTED_SETUP.md`](AI_ASSISTED_SETUP.md)，选择
“投机解码”“多卡推理”或“DeepSpeed 训练”。AI 会返回完整命令、退出码和验收结果。
