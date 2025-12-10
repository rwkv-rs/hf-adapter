# AI 统一操作入口

这是本仓库**唯一的 AI 操作说明**。Codex、Claude Code、Cursor 或其他能使用
终端的 AI，只需要先定位本页，再按任务路由读取一个用户教程。不要从多个文档中
拼装提示词，也不要一次执行多个工作流。

本页可以覆盖安装、转换、推理、缓存、投机解码、训练、多卡、量化和 Apple
部署。它不会要求 AI 修改内核、运行全量 benchmark 或自动下载大模型。

## 使用前的安全规则

- 把已经 clone 的 `hf-adapter` 仓库作为 AI 工作区。
- 不要在提示词里提供密码、Hugging Face 私有 token、SSH key 或云平台密钥。
- 公开的 BlinkDL 模型和词表不需要 token。
- AI 必须在下载、安装系统软件、删除文件或使用多张 GPU 前请求确认。
- 第一次安装固定使用 0.4B；不要用 7.2B/13.3B 验证环境。
- AI 只能报告真实命令输出。文件存在、脚本已启动或“理论上可用”都不是通过。

## 任务路由

先在表中选择**一个**任务 ID。专题文档提供人类可复制的命令，本页统一规定 AI
应该如何检查、执行、停止和汇报。

| 任务 ID | 用户目标 | 只读这份教程 | AI 必须观察的通过证据 |
|---|---|---|---|
| `first-run` | 安装、下载 0.4B、转换并生成 | [`USER_GUIDE_ZH.md`](USER_GUIDE_ZH.md) | `RESULT: READY`、`[PASS] Model directory`、生成退出 0 且有新文本 |
| `inference` | 转换、HF API、保存重载、离线/native | [`INFERENCE_WORKFLOWS.md`](INFERENCE_WORKFLOWS.md) | 所选章节的退出码 0、`PASS` 或有限 loss/logits |
| `gradio-native-hf` | 在官方 RWKV-Gradio-3 网页运行 Native HF | [`GRADIO_NATIVE_HF.md`](GRADIO_NATIVE_HF.md) | B1/B8 生成、切换后复用、速度标签、截图和进程显存全部可观察 |
| `cache` | 循环状态、动态 batch、chunked prefill | [`INFERENCE_WORKFLOWS.md`](INFERENCE_WORKFLOWS.md) | 所选 cache 测试所有 mode/shape 打印 `PASS` |
| `speculative` | 投机解码 | [`ADVANCED_USAGE_ZH.md`](ADVANCED_USAGE_ZH.md) | target greedy 完全一致、draft/target 调用计数有效、`PASS` |
| `training` | PEFT LoRA、Trainer、保存合并、恢复 | [`TRAINING_WORKFLOWS.md`](TRAINING_WORKFLOWS.md) | 有限 loss、非零梯度/参数变化、所选精确 `PASS` |
| `train-temp-alignment` | 在 CUDA 上复现官方 RWKV-LM train_temp 数学与训练效果 | [`TRAIN_TEMP_CUDA.md`](TRAIN_TEMP_CUDA.md) | 单步逐张量 `pass`；至少 3-seed cohort `pass`；长程恢复的模型/优化器/RNG 哈希和稳态显存门通过；精确 GPU/模型/commit/hash 齐全 |
| `trl` | SFT、DPO 或 GRPO | [`TRAINING_WORKFLOWS.md`](TRAINING_WORKFLOWS.md) | 所选流程 `status: pass` 或 `NATIVE ... PASS` |
| `multi-gpu-inference` | HF `device_map` 层切分 | [`ADVANCED_USAGE_ZH.md`](ADVANCED_USAGE_ZH.md) | 至少两张可见 GPU、单卡参考一致、`PASS` |
| `deepspeed` | ZeRO-2/ZeRO-3 smoke | [`ADVANCED_USAGE_ZH.md`](ADVANCED_USAGE_ZH.md) | 至少两张 CUDA 卡、请求的 stage 全部 `PASS`、结果行落盘 |
| `quantization` | bnb W8/W4、原生 MM8/MM4 或 RTX 5090 BN/TN Marlin W4 | [`QUANTIZATION_USAGE.md`](QUANTIZATION_USAGE.md) | 功能、质量、footprint、配对端到端速度四类结论分别给证据 |
| `apple` | MPS、MLX、packed W8/W4 或 CoreML | [`APPLE_USAGE.md`](APPLE_USAGE.md) | 精确 runtime、生成/对齐、内存和所选章节通过标记 |

全部适配及边界见 [`COMPLETE_ADAPTER_GUIDE.md`](COMPLETE_ADAPTER_GUIDE.md)。
它是给人选任务的索引，不是第二份 AI 提示词。

## 复制这份完整任务模板

只替换尖括号内容。`MODEL` 不确定时填写 `AUTO`，让 AI 先检查而不是猜路径。

```text
在当前 RWKV-7 Hugging Face adapter 仓库中执行一个任务并给出可复核证据。

TASK_ID: <从 docs/AI_ASSISTED_SETUP.md 的任务路由中选择一个>
MODEL: <已经转换的 HF 模型目录，首次安装或未知时填 AUTO>
DEVICE: <auto/cpu/cuda/mps>
DTYPE: <auto/fp32/fp16/bf16>
RESULT_DIR: <需要保存证据时填写仓库内目录，否则填 NONE>

唯一 AI 规则源是 docs/AI_ASSISTED_SETUP.md。先阅读它，再根据 TASK_ID 只读取
路由表指定的一份教程。不要搜索或拼接其他文档中的旧提示词。

严格执行：
1. 先输出当前仓库路径、git 状态、操作系统、shell、Python、可用磁盘、
   PyTorch、加速器数量/名称和可用显存。所有值来自命令，不允许猜。
2. 检查 TASK_ID、MODEL、DEVICE、DTYPE 是否与真实机器匹配。MODEL=AUTO 且
   TASK_ID=first-run 时，使用公开 0.4B；其他任务找不到模型时停止并报告。
3. 给出本次只执行的教程章节、最终命令、预计下载空间和风险。模型下载、系统级
   安装、删除/覆盖文件、多 GPU 运行前必须等我确认。
4. 使用仓库内 .venv，不全局安装包，不删除现有文件，不覆盖无关改动。
5. 一次只运行一个 TASK_ID 和其中一个编号章节。不要顺便跑 benchmark、训练、
   量化或内核调优。
6. 遇到第一个非零退出码立即停止。解释第一处真实错误，只修复这一处，然后重新
   运行失败命令。不得用“应该可以”代替重跑。
7. trust_remote_code=True 只用于我确认可信的本地转换模型目录。
8. 按教程的精确 PASS 门槛验收。smoke 只能称为兼容 smoke；不得扩写成质量、
   收敛、生产稳定性、Tensor Parallel 或速度结论。
9. 量化任务必须分开报告功能、质量、内存和配对端到端速度。缺少哪类证据就明确
   写未验收，不能用模型文件变小或 microbench 代替速度。
10. 最后严格使用本文的统一汇报格式；不输出密码、token、SSH key 或私密路径。
```

## AI 执行状态机

AI 必须按顺序完成，不允许跳过失败状态：

| 状态 | 动作 | 进入下一状态的证据 |
|---|---|---|
| `inspect` | 检查仓库、系统、Python、磁盘、GPU、模型路径 | 命令真实输出齐全 |
| `plan` | 锁定一个任务和一个教程章节，列最终命令与空间 | 用户能判断要执行什么 |
| `approve` | 对下载、覆盖、系统变更和多卡运行请求确认 | 用户明确同意 |
| `execute` | 在 `.venv` 中执行，非零退出立即停 | 命令退出 0 |
| `verify` | 执行该章节定义的对齐/PASS 检查 | 精确通过标记和关键数值存在 |
| `report` | 用统一格式汇报并复述边界 | 证据与结论一一对应 |

## 首次安装的固定下载规则

`TASK_ID=first-run` 时，AI 必须使用
`BlinkDL/rwkv7-g1/rwkv7-g1d-0.4b-20260210-ctx8192.pth`，下载前报告目标为
`models/source/`，并等待确认。安装顺序固定为：

1. 创建 `.venv` 并安装基础包；
2. `python examples/check_environment.py` 得到 `RESULT: READY`；
3. 下载官方 0.4B 权重和官方词表；
4. 按 [`USER_GUIDE_ZH.md`](USER_GUIDE_ZH.md) 转换；
5. 带 `--model` 再检查，得到 `[PASS] Model directory`；
6. 用 `examples/generate.py` 生成 8 token。

固定生成命令为：

```bash
python examples/generate.py --model models/rwkv7-g1d-0.4b-hf \
  --prompt "User: Say hello in one sentence. Assistant:" --max-new-tokens 8
```

Linux NVIDIA 可以在首次 native 通过后安装 `.[cuda]` 启用原生融合 kernel。
普通 RWKV 任务不得因为 FLA 未安装而失败；`.[fla-reference]` 只用于明确的参考
benchmark。

## 统一汇报格式

AI 最终回答必须包含以下字段，不要只写“完成”：

```text
TASK_ID:
教程与章节:
仓库 revision / git 状态:
模型路径:
设备 / GPU / dtype / backend:
实际执行命令:
每条命令退出码:
PASS 证据或关键数值:
结果文件路径:
未通过或未执行项:
结论边界:
```

量化任务在 `PASS 证据或关键数值` 中额外分四行：`functional`、`quality`、
`memory`、`paired speed`。训练任务额外给出可训练参数、loss、梯度/参数变化；
多卡任务额外给出可见 GPU 和实际 world size。

## 安全排错交接

向另一个 AI 求助时只提供：OS/shell、GPU、Python、失败命令、第一段完整错误、
模型是否公开、输出目录是否可删除。不要提供任何密码、token、SSH key 或云密钥。

AI 在以下情况必须停下而不是猜：模型路径不存在、磁盘不足、硬件不满足教程前置
条件、下载需要私有认证、命令会覆盖现有结果、或教程没有为所请求结论定义验收门槛。
