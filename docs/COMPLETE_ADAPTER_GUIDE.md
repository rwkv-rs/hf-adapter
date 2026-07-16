# RWKV-7 HF Adapter 全功能使用指南

本页帮助你按目标找到对应教程。第一次使用建议先完成
[`USER_GUIDE_ZH.md`](USER_GUIDE_ZH.md)，确认模型能够生成 8 个新 token，
再继续训练、量化、多卡或部署流程。

## 教程总表

| 用户目标 | 教程 | 完成标志 | 使用建议 |
|---|---|---|---|
| 安装、检查环境、下载、转换和生成 | [`USER_GUIDE_ZH.md`](USER_GUIDE_ZH.md) | `RESULT: READY`、模型目录 `PASS`、输出新文本 | 新环境建议从 0.1B/0.4B 开始；CUDA、MPS 和 CPU 都有对应路线 |
| 单个/批量/大模型转换、保存重载、离线运行 | [`INFERENCE_WORKFLOWS.md`](INFERENCE_WORKFLOWS.md) | 转换退出码 0、manifest 成功、重载测试打印 `PASS` | 转换大模型时可使用 `--low-memory` 降低主机内存占用 |
| 使用 `AutoModelForCausalLM`、loss、mask 和无 FLA 原生后端 | [`INFERENCE_WORKFLOWS.md`](INFERENCE_WORKFLOWS.md) | API 命令退出码 0；原生 smoke 打印对应通过标记 | 原生后端适合便携运行；CUDA 优化后端适合已验证的 NVIDIA 环境 |
| 复用循环状态、批量缓存、动态批处理和分块 prefill | [`INFERENCE_WORKFLOWS.md`](INFERENCE_WORKFLOWS.md) | 每个缓存/prefill 测试打印 `PASS` | 可直接用于构建 HF serving 的状态与批处理层 |
| 投机解码或对齐较小 draft | [`ADVANCED_USAGE_ZH.md`](ADVANCED_USAGE_ZH.md) | 与 target greedy 完全一致并打印 `PASS`；配对 benchmark 给出接受率和速度 | 先验证 token 对齐，再根据目标模型选择 draft 大小 |
| 多卡 `device_map` 推理 | [`ADVANCED_USAGE_ZH.md`](ADVANCED_USAGE_ZH.md) | 分卡输出与单卡参考一致并打印 `PASS` | 适合按层分配模型；可用 `max_memory` 控制每张卡的预算 |
| PEFT LoRA、adapter 保存/加载/合并、Trainer 和断点恢复 | [`TRAINING_WORKFLOWS.md`](TRAINING_WORKFLOWS.md) | loss/梯度有限且打印对应 `PASS` | 先运行小模型 smoke，再替换为自己的数据和训练配置 |
| TRL SFT、DPO 和 GRPO | [`TRAINING_WORKFLOWS.md`](TRAINING_WORKFLOWS.md) | `NATIVE SFT/DPO/GRPO PASS` | 示例提供兼容性起点，正式训练时按数据规模调整 batch 和 checkpoint |
| DeepSpeed ZeRO-2/ZeRO-3 | [`ADVANCED_USAGE_ZH.md`](ADVANCED_USAGE_ZH.md) | 所请求的结果行为 `PASS` | 推荐 Linux/WSL2 和至少两张 CUDA 卡，并为断点恢复保留输出目录 |
| bitsandbytes W8/W4 或原生 MM8/MM4 | [`QUANTIZATION_USAGE.md`](QUANTIZATION_USAGE.md) | 生成、有限 logits、对齐和 footprint 检查通过 | 可先选择省显存路线；追求速度时使用硬件矩阵中对应显卡的已验证配置 |
| Apple MPS、MLX、packed W8/W4、会话和 CoreML | [`APPLE_USAGE.md`](APPLE_USAGE.md) | 对应命令输出 JSON/通过标记 | 按具体 Apple 芯片、模型和 batch shape 选择证据行 |
| 让 AI 编程助手执行某项任务 | [`AI_ASSISTED_SETUP.md`](AI_ASSISTED_SETUP.md) | AI 报告完整命令、退出码、设备/模型路径和通过标记 | 只需选择任务并填写模型与设备信息；密码和私有 token 留在本机 |
| 复现硬件和性能结论 | [`HARDWARE_MATRIX.md`](HARDWARE_MATRIX.md)、[`PERFORMANCE.md`](PERFORMANCE.md)、[`../bench/INDEX.md`](../bench/INDEX.md) | 同卡、同 shape 原始证据和文档门槛 | 精确卡数据可直接用于选择 backend、dtype、batch 和量化策略 |

## 如何选择合适路线

1. **先完成首次生成。** 使用 0.1B 或 0.4B 模型确认环境、模型目录和 tokenizer
   都能通过，再换成目标模型。
2. **按任务选择多卡方案。** 推理时从 `device_map` 开始；Trainer/TRL 训练时从
   DeepSpeed ZeRO-2 或 ZeRO-3 开始。
3. **按目标选择量化方案。** 显存优先时查看 footprint；速度优先时查看同显卡、
   同模型和同 batch 的配对 fp16/bf16 结果。
4. **按设备选择后端。** NVIDIA CUDA、Apple MPS/MLX 和 CPU 均有入口；具体优化
   参数以 [`HARDWARE_MATRIX.md`](HARDWARE_MATRIX.md) 的精确卡记录为准。
5. **逐步扩大规模。** smoke 通过后，再使用真实数据、目标上下文长度和计划运行时长
   执行训练或部署验收。

## 需要 AI 帮你执行

打开唯一的 [`AI_ASSISTED_SETUP.md`](AI_ASSISTED_SETUP.md)，选择任务编号并填写
模型路径、设备和 dtype。AI 会按照统一格式执行命令、检查可观察结果，并在失败时
从最近的安全步骤继续。
