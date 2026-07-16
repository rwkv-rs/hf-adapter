# 完整适配教学索引

本页是 RWKV-7 Hugging Face 适配器的教学覆盖合同。每一项已经实现或正式
发布的用户能力，都必须有可复制的教程、可观察的通过标准和明确的适用边界。

English version: [`COMPLETE_ADAPTER_GUIDE.md`](COMPLETE_ADAPTER_GUIDE.md)

第一次使用请先完成 [`USER_GUIDE_ZH.md`](USER_GUIDE_ZH.md)。普通的 8 token
生成成功后，再从下表选择后续任务。

## 教程总表

| 用户目标 | 教程 | 验收证据 | 当前边界 |
|---|---|---|---|
| 安装、检查环境、下载、转换和生成 | [`USER_GUIDE_ZH.md`](USER_GUIDE_ZH.md) | `RESULT: READY`、模型目录 `PASS`、输出新文本 | 建议从 0.1B/0.4B 开始；FLA 不是必需依赖 |
| 单个/批量/大模型转换、保存重载、离线运行 | [`INFERENCE_WORKFLOWS_ZH.md`](INFERENCE_WORKFLOWS_ZH.md) | 转换退出码 0、manifest 成功、重载测试打印 `PASS` | `--low-memory` 只降低转换内存，不降低推理显存 |
| 使用 `AutoModelForCausalLM`、loss、mask 和无 FLA 原生后端 | [`INFERENCE_WORKFLOWS_ZH.md`](INFERENCE_WORKFLOWS_ZH.md) | API 命令退出码 0；原生 smoke 打印对应通过标记 | 原生后端是便携兼容路线，性能取决于具体显卡 |
| 复用循环状态、批量缓存、动态批处理和分块 prefill | [`INFERENCE_WORKFLOWS_ZH.md`](INFERENCE_WORKFLOWS_ZH.md) | 每个缓存/prefill 测试打印 `PASS` | 这是 HF serving 基础能力，不是完整推理服务引擎 |
| 投机解码或对齐较小 draft | [`ADVANCED_USAGE_ZH.md`](ADVANCED_USAGE_ZH.md) | 与 target greedy 完全一致并打印 `PASS`；训练 draft 还需配对速度 | 同模型 draft 和训练成功都不证明加速 |
| 多卡 `device_map` 推理 | [`ADVANCED_USAGE_ZH.md`](ADVANCED_USAGE_ZH.md) | 分卡输出与单卡参考一致并打印 `PASS` | 层切分不是原生 Tensor Parallel |
| PEFT LoRA、adapter 保存/加载/合并、Trainer 和断点恢复 | [`TRAINING_WORKFLOWS_ZH.md`](TRAINING_WORKFLOWS_ZH.md) | loss/梯度有限且打印对应 `PASS` | 小型 smoke 不证明生产训练收敛 |
| TRL SFT、DPO 和 GRPO | [`TRAINING_WORKFLOWS_ZH.md`](TRAINING_WORKFLOWS_ZH.md) | `NATIVE SFT/DPO/GRPO PASS` | 固定小数据只用于兼容验收，不是训练配方 |
| DeepSpeed ZeRO-2/ZeRO-3 | [`ADVANCED_USAGE_ZH.md`](ADVANCED_USAGE_ZH.md) | 所请求的结果行为 `PASS` | 需要 Linux/WSL2 和至少两张 CUDA 卡；恢复覆盖取决于矩阵 |
| bitsandbytes W8/W4 或原生 MM8/MM4 | [`QUANTIZATION_USAGE_ZH.md`](QUANTIZATION_USAGE_ZH.md) | 生成、有限 logits、对齐和 footprint 检查通过 | 能加载或更省显存不代表更快 |
| Apple MPS、MLX、packed W8/W4、会话和 CoreML | [`APPLE_USAGE_ZH.md`](APPLE_USAGE_ZH.md) | 对应命令输出 JSON/通过标记 | 已发布性能只适用于明确的 M5 和 shape，不代表所有 Apple 设备 |
| 让 AI 编程助手执行某项任务 | [`AI_ASSISTED_SETUP.md`](AI_ASSISTED_SETUP.md) | 报告完整命令、退出码、设备/模型路径和通过标记 | 不要提供密码、私有 token 或 SSH 密钥 |
| 复现硬件和性能结论 | [`HARDWARE_MATRIX.md`](HARDWARE_MATRIX.md)、[`PERFORMANCE.md`](PERFORMANCE.md)、[`../bench/INDEX.md`](../bench/INDEX.md) | 同卡、同 shape 原始证据和文档门槛 | 性能证据的范围比 API 兼容范围窄 |

## 这些内容不能写成“已完成适配”

- 原生 vLLM/SGLang 不属于本仓库的 HF-only 范围。
- `device_map` 和 DeepSpeed ZeRO 不能证明生产级 Tensor/Pipeline Parallel。
- Turing、Hopper、AMD 只有策略条目但没有精确卡证据时，只算路由准备。
- 全模型 W8/W4 在所有显卡都不慢于 fp16 仍未完成。没有精确卡验收时，只能
  当作兼容/省内存路线。
- smoke 只证明接口执行并满足局部合同，不证明模型质量、长期收敛、容量或加速。

## 每个新增教程必须包含六项内容

1. 前置条件和支持环境；
2. 最小安全模型或输入；
3. 可直接复制的用户命令或 API；
4. 精确且可观察的通过标准；
5. 失败恢复方法和当前限制；
6. 禁止猜测、必须给证据的 AI 助手指令。

如果一项实现只藏在源码、测试、benchmark 日志或 PR 中，普通用户无法找到和
验收，就不能算教学文档已经完成。
