# RWKV-7 HF Adapter 状态

本页是 **Hugging Face / Transformers 适配**这条线的贡献者状态入口。仓库范围严格限定在 HF 加载/生成/训练、PEFT/TRL 兼容、HF state-cache helper、量化推理、可复现 benchmark。

vLLM、SGLang、DFlash 与独立服务引擎是后续项目,不得阻塞 HF 适配交付。

> 本页只放「状态快照 + 硬件矩阵」。**已完成进展详见 [`HF_CRITERIA.md`](HF_CRITERIA.md) §2、当前缺口详见 §3、验收门禁详见 §1;性能数字详见 [`BENCHMARK.md`](BENCHMARK.md);性能 kernel 路线详见 [`FUSED_BACKEND.md`](FUSED_BACKEND.md)。**

## 当前状态摘要

| 领域 | 状态 | 说明 |
|---|---|---|
| HF 加载 / 保存 / 生成 | 已完成 | `AutoConfig` / `AutoTokenizer` / `AutoModelForCausalLM`、`save_pretrained` / `from_pretrained`、`generate(use_cache=True)`。 |
| 官方权重转换 | 已完成 | 官方 `.pth` → HF `safetensors`;shape 推断覆盖已发布尺寸。 |
| 精度对齐 | smoke 基线通过 | 0.1B V100 对齐官方 `rwkv`,过 top-k / cosine / greedy-window 门禁。 |
| PEFT | smoke + 适配器生命周期 | LoRA fwd/bwd、adapter save/load/merge。 |
| Trainer / TRL | 小模型 smoke | HF Trainer + TRL SFT/DPO/GRPO smoke,校验有限 loss + trainable delta。 |
| DeepSpeed ZeRO | 2×V100 小 smoke | ZeRO-2、ZeRO-3 HF Trainer smoke。 |
| HF recurrent cache helper | 当前适配器已覆盖 | `RWKV7StateCache`:select/reorder/drop/compact、offload/restore、chunked prefill、telemetry。 |
| 量化加载 | 可用 | bnb 8/4-bit 加载/生成、显存下降;速度仍是生产缺口。 |
| Native / 无 FLA 后端 | 实验性 | 用于 upstream/AMD/CPU fallback;尚未替代 wrapper。 |
| 生产性能 | 部分 | V100 fast-token/native-graph 提升 decode;Albatross 级与量化速度门禁未闭合。 |
| 跨卡验证 | 部分 | V100 基线 + 部分 Blackwell;**4090 / 大规模验证进行中**。 |

## 硬件 / 卡适配状态

V100 是开发与回归基线。目标不是「一张卡能跑」,而是常见专业/消费卡上有明确行为。

| 硬件目标 | 当前状态 | 贡献者可补 |
|---|---|---|
| 1× V100 32GB | 主基线 | 保持 correctness / generation / cache / 量化 / 小训练回归行绿灯。 |
| 2× V100 32GB | ZeRO smoke 已记录 | 补 ZeRO checkpoint-resume 与大模型 ZeRO 行。 |
| RTX 50 系 / Blackwell | 已有部分验证 | 重跑 acceptance 脚本 + 补 decode/prefill/quant 行。 |
| RTX 4090 / Ada | **进行中** | fp16/bf16 速度、显存、量化、PEFT smoke 行。 |
| A100 / Ampere | 待补 | 生产级 batch sweep 与 ZeRO 行。 |
| H100 / Hopper | 待补 | 高端吞吐、bf16、量化、大模型行。 |
| Pascal / 老 NVIDIA | 条件允许时补 | fallback 行为、fp16 约束、量化策略。 |
| AMD / ROCm | 开放 | 先做 native / 无 FLA 纯 PyTorch 兼容,再考虑 kernel。 |
| CPU fallback | 部分 / 实验 | 保持无 CUDA import + tiny native 测试绿灯。 |

新增卡结果时至少记录:GPU 名称与数量、驱动 / CUDA 或 ROCm / PyTorch / Transformers / PEFT / TRL / DeepSpeed 版本、模型尺寸与 dtype、所用命令、`bench/results.jsonl` 行(支持 `--results` 时)、`BENCHMARK.md` 或 PR body 的一句说明。

## 当前缺口(摘要)

完整缺口清单见 [`HF_CRITERIA.md`](HF_CRITERIA.md) §3。当前重点:

- **大模型训练矩阵 + 4090 / 大规模验证:进行中。**
- 量化速度未达标(W8/W4 仍慢于 fp16,需 fused/native 量化 kernel)。
- Albatross / RWKV-LM 生产级性能未闭合(见 [`FUSED_BACKEND.md`](FUSED_BACKEND.md))。
- 更多卡覆盖(A100 / H100 / 5090 / Pascal / AMD)。

## 下一步去哪

- 实操路线图:[`HF_TODO.md`](HF_TODO.md)
- 性能数字:[`BENCHMARK.md`](BENCHMARK.md)
- 验收门禁 + 已完成 + 缺口:[`HF_CRITERIA.md`](HF_CRITERIA.md)
- 性能 kernel 路线:[`FUSED_BACKEND.md`](FUSED_BACKEND.md)
