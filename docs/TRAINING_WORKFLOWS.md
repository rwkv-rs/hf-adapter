# PEFT、Trainer 和 TRL 教学

本页覆盖当前已经验证的全部 HF 训练集成。命令有意设计为短 smoke：在真实
数据集消耗大量时间之前，先证明接口、梯度、序列化和断点恢复可以工作。

前置条件：

```bash
python -m pip install -e ".[train]"
python examples/check_environment.py --model MODEL
```

第一次使用 0.1B 或 0.4B，把 `MODEL` 替换为已转换模型目录。训练时关闭循环
缓存（`use_cache=False`）。

![PEFT、Trainer、恢复、TRL 和分布式训练生命周期](assets/tutorials/08-training-ecosystem.png)

## 1. 证明 PEFT LoRA 梯度

```bash
python tests/test_peft_lora.py --model MODEL \
  --device cuda --attn-mode fused_recurrent
```

通过条件是 `loss` 有限、`nonzero_grad_count > 0` 且退出码为 0。smoke 默认
目标模块为 `r_proj`、`k_proj`、`v_proj`、`o_proj`、`key` 和 `value`。真实
实验要根据任务和可训练参数量重新审查，不能不加判断地照抄。

## 2. 保存、加载和合并 LoRA adapter

原生无 FLA round-trip 会训练一个很小的 adapter，保存后加载到新的 base，验证
merge/unmerge、`merge_and_unload`，并比较 logits 和 greedy token：

```bash
python tests/test_native_peft_save_load_merge.py \
  --model MODEL --device cuda --dtype fp32 --steps 2
```

必须打印 `NATIVE PEFT SAVE/LOAD/MERGE PASS`，表示测试模型的序列化和函数一致性
已经通过。随后使用真实验证集评估 adapter 的任务效果。

普通 PEFT 部署写法：

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base = AutoModelForCausalLM.from_pretrained("MODEL", trust_remote_code=True)
tuned = PeftModel.from_pretrained(base, "ADAPTER_DIR").eval()
merged = tuned.merge_and_unload()
merged.save_pretrained("MERGED_DIR", safe_serialization=True)
```

需要切换 adapter 或继续训练时保留未合并 adapter；固定推理部署可以使用合并目录。

## 3. HF Trainer 和 TRL SFT

在固定小文本上同时运行标准 Trainer 和 `SFTTrainer`：

```bash
python tests/test_hf_training_smoke.py --model MODEL \
  --device cuda --train-dtype bf16 --max-steps 1 --batch-size 1 \
  --gradient-accumulation-steps 1 --max-length 64 --backend both
```

通过时，每个 JSON 行都有 `status: pass`、有限 loss 和正的参数变化，最后打印
`PASS`。不支持 bf16 的硬件，第一次兼容测试使用 `--train-dtype fp32`。

单独验证原生无 FLA SFT：

```bash
python tests/test_native_sft_smoke.py --model MODEL \
  --device cuda --dtype fp32 --max-steps 2 --batch-size 1 --max-length 48
```

必须打印 `NATIVE SFT PASS`。

## 4. 恢复 Trainer checkpoint

```bash
python tests/test_native_trainer_resume_smoke.py --model MODEL \
  --device cuda --dtype fp32 --first-steps 2 --resume-steps 3 \
  --batch-size 2 --length 32
```

必须打印 `NATIVE TRAINER RESUME PASS`；前后两阶段都要更新参数，最终
global step 必须等于 `--resume-steps`。

这个便携 smoke 验证 model/adapter 和 Trainer state。正式训练还应按实际库版本
逐项保存并核验 optimizer、scheduler、RNG、dataloader 和分布式状态。

## 5. TRL DPO 和 GRPO

同时运行标准 HF adapter 路线：

```bash
python tests/test_hf_rl_training_smoke.py --model MODEL \
  --device cuda --train-dtype bf16 --max-steps 1 --batch-size 2 \
  --gradient-accumulation-steps 1 --max-length 64 --backend both
```

DPO/GRPO 都必须输出 `status: pass`、有限 loss 和参数更新，最后打印 `PASS`。

单独验证原生无 FLA：

```bash
python tests/test_native_dpo_smoke.py --model MODEL \
  --dtype fp32 --max-steps 3 --batch-size 2 --max-length 48

python tests/test_native_grpo_smoke.py --model MODEL \
  --dtype fp32 --max-steps 2 --batch-size 2 --max-completion-length 8
```

必须分别打印 `NATIVE DPO PASS` 和 `NATIVE GRPO PASS`。仓库内 prompt、偏好对和
reward 只是确定性测试数据，真实对齐训练必须换成经过审查的数据和评估方案。

## 6. 运行可重复训练矩阵

在 Linux/WSL2 上对一个或多个模型运行 PEFT、Trainer/SFT、DPO/GRPO：

```bash
DEVICE=cuda TRAIN_DTYPE=bf16 RESULTS=bench/results.jsonl \
  bash scripts/run_hf_training_matrix.sh MODEL_A MODEL_B
```

设置 `RUN_RESUME=1` 加入恢复。只有至少两张 CUDA 卡并安装 DeepSpeed 时才设置
`RUN_DEEPSPEED=1`。完整 ZeRO-2/3 命令和边界见
[`ADVANCED_USAGE_ZH.md`](ADVANCED_USAGE_ZH.md)。

## 7. 从 smoke 进入真实微调

开始长训练前，至少固定：

- base checkpoint、tokenizer、adapter config、代码 revision 和依赖锁；
- 合法数据集、train/eval 切分、格式化和截断策略；
- 序列长度、microbatch、梯度累积、学习率、schedule、精度、seed、保存频率；
- 独立质量指标以及 loss、吞吐、峰值显存和 NaN 日志；
- 在一次可丢弃短运行中验证过的保存/加载/恢复流程；
- 包含未合并 adapter 和运行配置的回滚产物。

状态和证据见 [`TRAINING.md`](TRAINING.md)。生产级 Tensor Parallel 训练不是本
仓库已经完成的结论。

## 8. 交给 AI 执行

需要 AI 协助时，请打开 [`AI_ASSISTED_SETUP.md`](AI_ASSISTED_SETUP.md)，选择
“PEFT/Trainer”“TRL”或“DeepSpeed 训练”。AI 会返回完整命令、退出码和验收结果。
