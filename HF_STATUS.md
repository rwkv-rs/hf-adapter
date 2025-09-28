# RWKV-7 HF Adapter 状态

本页是 **Hugging Face / Transformers 适配**这条线的贡献者状态入口。仓库范围严格限定在 HF 加载/生成/训练、PEFT/TRL 兼容、HF state-cache helper、量化推理、可复现 benchmark。

vLLM、SGLang、DFlash 与独立服务引擎是后续项目,不得阻塞 HF 适配交付。

> 本页只放「状态快照 + 硬件矩阵」。**已完成进展详见 [`docs/reference/HF_CRITERIA.md`](docs/reference/HF_CRITERIA.md) §2、当前缺口详见 §3、验收门禁详见 §1;性能数字详见 [`BENCHMARK.md`](BENCHMARK.md);性能 kernel 路线详见 [`docs/performance/FUSED_BACKEND.md`](docs/performance/FUSED_BACKEND.md)。**

## 当前状态摘要

| 领域 | 状态 | 说明 |
|---|---|---|
| HF 加载 / 保存 / 生成 | 已完成 | `AutoConfig` / `AutoTokenizer` / `AutoModelForCausalLM`、`save_pretrained` / `from_pretrained`、`generate(use_cache=True)`。 |
| 官方权重转换 | 已完成 | 官方 `.pth` → HF `safetensors`;shape 推断覆盖已发布尺寸。 |
| 精度对齐 | smoke 基线通过 | 0.1B V100 对齐官方 `rwkv`,过 top-k / cosine / greedy-window 门禁;13.3B V100 对齐通过(cos 0.9999976,greedy 16/16)。 |
| PEFT | smoke + 适配器生命周期 | LoRA fwd/bwd、adapter save/load/merge。 |
| Trainer / TRL | 大模型 V100 + A100 smoke 已补 | V100 0.4B/1.5B/2.9B 训练生态已补;A100 40GB 0.4B/1.5B/2.9B/7.2B Trainer/SFT/DPO + HF checkpoint resume 通过;13.3B 推理对齐+decode 速度已验(单卡 V100-32GB fp16,native_jit 18.4 tok/s,1.58× fla),训练需 >32GB。 |
| DeepSpeed ZeRO | ZeRO2/3 base + resume smoke | ZeRO-2/3 HF Trainer smoke 通过;ZeRO2 checkpoint resume 已在 A100 40GB 验证到 7.2B;ZeRO3 checkpoint resume 已在 2×V100 0.1B native/HF 路径通过,仍需扩展到更大模型/A100。 |
| HF recurrent cache helper | 当前适配器已覆盖 | `RWKV7StateCache`:select/reorder/drop/compact、offload/restore、chunked prefill、telemetry。 |
| 量化加载 | 大模型 V100 功能通过 | bnb 8/4-bit 加载/生成、显存下降;0.4B/1.5B/2.9B/7.2B V100 pass/pass;**卡验证须含 native mm8/mm4 速度(不只是 bnb,见 HF_TODO §4 / AGENTS)**;速度仍是生产缺口。 |
| Native / 无 FLA 后端 | HF 全生态兼容(opt-in) | 纯 PyTorch,过 HF Cache 契约 / generate 全模式 / PEFT / Trainer / SFT / DPO / GRPO;fla 完全不可达也能 load+generate(#59/#60)。仍 opt-in(`RWKV7_NATIVE_MODEL=1`),未替换默认 wrapper。 |
| Apple Silicon / MPS | M5 16GB smoke pass | `flash-linear-attention` 已移到可选 extra;Apple Silicon smoke 脚本与文档已补。MacBook Air / Apple M5 / 16GB / macOS 26.5 / PyTorch 2.12.1 tiny native MPS `generate()`、0.1B HF load/forward/generate、0.4B HF fp32/fp16 load/forward/短 generate 通过;tiny MPS train/PEFT LoRA smoke 通过,tiny HF Trainer/PEFT Trainer 通过,0.1B 和 0.4B 真模型 PEFT LoRA backward + HF Trainer + TRL SFT/DPO/GRPO 通过;0.4B fp32/fp16 prompt 16/64/128 sweep、0.4B fp16 prompt 256/512 sweep、0.4B Trainer/TRL 2-step 已补;1.5B fp16 load/forward/短 generate + prompt16/64/128/256/512 sweep + prompt512/new8 通过,1.5B fp32 PEFT LoRA manual backward + HF Trainer + TRL SFT/DPO/GRPO 1/2/3/5/10-step 均有有限参数更新。更长训练、完整 MLX/Metal 后端、Apple W8/W4 生产速度仍待补;native MM8/MM4 MPS 功能 smoke 已补(tiny + 0.1B/0.4B min-params sweep, packed footprint 下降);初始 MLX recurrent reference 已补(tiny MLX/Torch recurrent parity、state-cache select/chunked-prefill/session、tokenizer prompt/API、dynamic-batch state select、0.1B/0.4B/1.5B HF full MLX recurrent prefill/generate、scripts/mlx_generate.py 文本生成 CLI、MLXGenerationSession 分段 decode/session smoke、selected safetensor export)。 |
| 生产性能 | 部分 | V100 fast-token/native-graph 提升 decode;Albatross 级与量化速度门禁未闭合。 |
| 跨卡验证 | 部分 | V100 基线已加强;A100 40GB 0.1B 基线 + 0.4B/1.5B/2.9B/7.2B 大模型 smoke/batch/quant/training/resume/ZeRO 已补;Pascal GTX 1080 Ti 0.1B fp16 smoke/bnb W8-W4/native mm8-mm4 quant speed/bsz1-4 + 0.4B fp16 bench 已补;A100 80GB、Turing/H100/AMD 等仍需贡献。 |

## 硬件 / 卡适配状态

V100 是开发与回归基线。目标不是「一张卡能跑」,而是常见专业/消费卡上有明确行为。

| 硬件目标 | 当前状态 | 贡献者可补 |
|---|---|---|
| 1× V100 32GB | 主基线加强 | 见 [`docs/validation/V100_HF_VALIDATION.md`](docs/validation/V100_HF_VALIDATION.md):0.4B/1.5B/2.9B 训练生态、7.2B PEFT/quant、量化功能矩阵。 |
| 2× V100 32GB | ZeRO2/3 base + resume | ZeRO2 resume 已验证到 2.9B;ZeRO3 resume 已在 0.1B native/HF 路径通过(`bench/results_v100_zero3_resume_2gpu_20260703.jsonl`)。 |
| RTX 50 系 / Blackwell | 已有部分验证 | 重跑 acceptance 脚本 + 补 decode/prefill/quant 行。 |
| RTX 4090 / Ada | **基础验证完成** | fp16/bf16 速度、显存、量化、PEFT smoke 已 PASS(见 BENCHMARK 4090 段);fused/native 量化速度持续优化。 |
| A100 / Ampere | A100 40GB 大模型验证已补 | 见 [`docs/validation/A100_HF_VALIDATION.md`](docs/validation/A100_HF_VALIDATION.md):0.1B 基线 + 0.4B/1.5B/2.9B/7.2B smoke、fp16/bf16 batch sweep、8/4-bit quant 功能/显存与 interim speed、Trainer/SFT/DPO、HF checkpoint resume、2×A100 ZeRO-2/3 base、ZeRO2 resume。A100 80GB 未测。 |
| H100 / Hopper | 待补 | 高端吞吐、bf16、量化、大模型行。 |
| Pascal / 老 NVIDIA | GTX 1080 Ti smoke 已补 | 0.1B fp16 默认 native/no-FLA fallback、bnb 8/4-bit 量化加载与 decode speed、native mm8/mm4 decode speed、bench_speed、bsz 1/2/4 batch sweep 通过;0.4B fp16 bench_speed 通过;训练未跑。bnb 慢于 fp16,但 native mm8/mm4 在 `lm_head` 量化下接近 fp16 decode。 |
| AMD / ROCm | 开放 | 先做 native / 无 FLA 纯 PyTorch 兼容,再考虑 kernel。 |
| Apple Silicon / MPS | 初始可跑 | 见 [`docs/hardware/APPLE_SILICON.md`](docs/hardware/APPLE_SILICON.md):native/no-FLA 安装、MPS probe、0.1B/0.4B smoke、tiny train/PEFT/Trainer、0.1B/0.4B PEFT LoRA/Trainer/SFT/DPO/GRPO 命令、0.4B generation sweep 到 512、1.5B fp16 inference/sweep 到 prompt512/new8 和 fp32 manual/Trainer/TRL PEFT LoRA 1/2/3/5/10-step 行、初始 MLX recurrent reference + session decode smoke、RafaelUI MLX/Metal 后续参考。 |
| CPU fallback | 部分 / 实验 | 保持无 CUDA import + tiny native 测试绿灯。 |

新增卡结果时至少记录:GPU 名称与数量、驱动 / CUDA 或 ROCm / PyTorch / Transformers / PEFT / TRL / DeepSpeed 版本、模型尺寸与 dtype、所用命令、`bench/results.jsonl` 行(支持 `--results` 时)、`BENCHMARK.md` 或 PR body 的一句说明。

## 当前缺口(摘要)

完整缺口清单见 [`docs/reference/HF_CRITERIA.md`](docs/reference/HF_CRITERIA.md) §3。当前重点:

- **ZeRO3 checkpoint resume** V100 0.1B native/HF smoke 已闭合;下一步扩到 0.4B+ / A100 大模型矩阵。
- **A100 80GB 验证** 当前集群不可用;A100 40GB 大模型 smoke/training/ZeRO 证据已补。
- 量化速度未达标(bnb W8/W4 仍慢于 fp16;Pascal native mm8/mm4 的 0.1B `lm_head` 行接近 fp16,但更大模型/更多投影仍需 fused/native 量化矩阵)。
- Albatross / RWKV-LM 生产级性能未闭合(见 [`docs/performance/FUSED_BACKEND.md`](docs/performance/FUSED_BACKEND.md))。
- 更多卡覆盖(Turing / H100 / 5090 / AMD / Apple Silicon)与更长训练吞吐。

## 下一步去哪

- 实操路线图:[`HF_TODO.md`](HF_TODO.md)
- 性能数字:[`BENCHMARK.md`](BENCHMARK.md)
- A100 训练/量化/ZeRO 验证矩阵:[`docs/validation/A100_HF_VALIDATION.md`](docs/validation/A100_HF_VALIDATION.md)
- V100 训练/量化/ZeRO 验证矩阵:[`docs/validation/V100_HF_VALIDATION.md`](docs/validation/V100_HF_VALIDATION.md)
- 验收门禁 + 已完成 + 缺口:[`docs/reference/HF_CRITERIA.md`](docs/reference/HF_CRITERIA.md)
- 性能 kernel 路线:[`docs/performance/FUSED_BACKEND.md`](docs/performance/FUSED_BACKEND.md)
