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
| Trainer / TRL | 大模型 V100 + A100 + A6000 smoke 已补 | V100 0.4B/1.5B/2.9B 训练生态已补;A100 40GB 0.4B/1.5B/2.9B/7.2B Trainer/SFT/DPO + HF checkpoint resume 通过;RTX A6000 48GB 0.4B/1.5B/2.9B Trainer/SFT/DPO + HF checkpoint resume 通过;13.3B 推理对齐+decode 速度已验(单卡 V100-32GB fp16,native_jit 18.4 tok/s,1.58× fla),训练需 >32GB。 |
| DeepSpeed ZeRO | ZeRO2/3 base + resume smoke | ZeRO-2/3 HF Trainer smoke 通过;ZeRO2 checkpoint resume 已在 A100 40GB 验证到 7.2B;ZeRO2/3 base + resume 已在 2×RTX A6000 验证到 2.9B;ZeRO3 checkpoint resume 已在 2×V100 0.1B native/HF 路径通过。 |
| HF recurrent cache helper | 当前适配器已覆盖 | `RWKV7StateCache`:select/reorder/drop/compact、offload/restore、chunked prefill、telemetry。 |
| 量化加载 | 大模型 V100 / A6000 功能通过;5090 native speed lane full matrix 已补 | bnb 8/4-bit 加载/生成、显存下降;0.4B/1.5B/2.9B/7.2B V100 与 RTX A6000 pass/pass;RTX A6000 另有 native mm8/mm4 decode telemetry 行。RTX 5090 native MM8/MM4 `speed` policy 已补 1.5B/2.9B/7.2B fresh-process 216-row 矩阵:footprint 全部下降,same-next 144/144,多行速度超过 fp16,但 7.2B 大压力形状仍低于 fp16;`memory` policy 仍是最大 footprint 下降但非速度达标 lane。 |
| Native / 无 FLA 后端 | HF 全生态兼容(opt-in) | 纯 PyTorch,过 HF Cache 契约 / generate 全模式 / PEFT / Trainer / SFT / DPO / GRPO;fla 完全不可达也能 load+generate(#59/#60)。仍 opt-in(`RWKV7_NATIVE_MODEL=1`),未替换默认 wrapper。 |
| Apple Silicon / MPS | M5 16GB smoke + CoreML stateful correctness + first Qwen3.5 gap row | `flash-linear-attention` 已移到可选 extra;Apple Silicon smoke 脚本与文档已补。MacBook Air / Apple M5 / 16GB / macOS 26.5 / PyTorch 2.12.1 tiny native MPS `generate()`、0.1B HF load/forward/generate、0.4B HF fp32/fp16 load/forward/短 generate 通过;tiny MPS train/PEFT LoRA smoke 通过,tiny HF Trainer/PEFT Trainer 通过,0.1B 和 0.4B 真模型 PEFT LoRA backward + HF Trainer + TRL SFT/DPO/GRPO 通过;0.4B fp32/fp16 prompt 16/64/128 sweep、0.4B fp16 prompt 256/512 sweep、0.4B Trainer/TRL 2-step 已补;1.5B fp16 load/forward/短 generate + prompt16/64/128/256/512 sweep + prompt512/new8 通过,1.5B fp32 PEFT LoRA manual backward + HF Trainer + TRL SFT/DPO/GRPO 1/2/3/5/10-step 均有有限参数更新。更长训练、完整 MLX/Metal 后端、Apple W8/W4 生产速度仍待补;native MM8/MM4 MPS 功能 smoke 已补(tiny + 0.1B/0.4B min-params sweep, packed footprint 下降);初始 MLX recurrent reference 已补(tiny MLX/Torch recurrent parity、state-cache select/chunked-prefill/session、多会话交错 session、tokenizer prompt/API、dynamic-batch state select、0.1B/0.4B/1.5B HF full MLX recurrent prefill/generate、scripts/mlx_generate.py 文本生成 CLI、MLXGenerationSession 分段 decode/session smoke、MLXGenerationSessionBatch 多会话交错 decode smoke、MLX prompt/decode sweep + repeat pressure、selected safetensor export)。新增 CoreMLTools 9 `stateful-multifunction` 真实导出/运行:0.1B/0.4B fp32 compute 的 MLState transfer、chunk split、HF greedy parity 全通过;INT8 包体≈0.45x/0.36x 且短 greedy 对齐,但 decode≈0.95x/0.98x fp32;0.1B INT4/LUT4 包体下降但短 greedy 未对齐。首批同机 `qwen3.5:0.8b-mlx` vs RWKV 0.4B 行显示 fp16 decode≈0.90x-0.94x、prefill≈0.07x-0.11x Qwen;W4 peak≈0.568x fp16 但 decode≈0.60x Qwen。fp16 CoreML compute、长上下文、Qwen memory/quality 和确认 ANE placement 仍待补。 |
| 生产性能 | 部分 | V100 fast-token/native-graph 已完成 exact-sm70 production-close:0.1B/0.4B/1.5B × bsz1/2/4/8 dense decode=`0.908x-1.248x` Albatross，prompt512 prefill=`0.930x-1.047x`;native W8/W4 speed lane payload=`0.803x-0.956x`、decode=`1.006x-1.128x` fp16、paired prefill=`0.996x-1.007x` fp16，正确性/cache handoff 全通过。4090 fixed-shape prefill graph 的 0.4B prompt512 bsz1/4 已达 `64.51k/107.87k tok/s`，B4 对同机当次 Albatross=`1.007x`，对历史最强记录=`0.916x`;W8/W4 speed lane 同时实现 footprint 下降且 prefill/decode 不慢于 w16。5090 native speed lane 已有 1.5B/2.9B/7.2B full matrix。剩余为更多卡/模型、历史高水位追赶与 full-memory quant prefill fused kernel。 |
| 跨卡验证 | 部分 | V100 基线已加强;A100 40GB 0.1B 基线 + 0.4B/1.5B/2.9B/7.2B 大模型 smoke/batch/quant/training/resume/ZeRO 已补;RTX A6000 48GB 0.1B core + 0.4B/1.5B/2.9B/7.2B smoke/batch/quant/native-mm/training/2-card ZeRO 已补;Pascal GTX 1080 Ti 0.1B fp16 smoke/bnb W8-W4/native mm8-mm4 quant speed/bsz1-4 + 0.4B fp16 bench 已补;A100 80GB、Turing/H100/AMD 等仍需贡献。 |

## 硬件 / 卡适配状态

V100 是开发与回归基线。目标不是「一张卡能跑」,而是常见专业/消费卡上有明确行为。

| 硬件目标 | 当前状态 | 贡献者可补 |
|---|---|---|
| 1× V100 32GB | **production-close 矩阵通过** | 0.1B/0.4B/1.5B × bsz1/2/4/8 dense decode/prefill Albatross P1 全通过，native W8/W4 decode 严格快于 fp16、prefill 在 1% 等价带内且 payload 下降；见 [`bench/v100_production_close_20260711/README.md`](bench/v100_production_close_20260711/README.md)。训练生态与大模型功能矩阵见 [`docs/validation/V100_HF_VALIDATION.md`](docs/validation/V100_HF_VALIDATION.md)。 |
| 2× V100 32GB | ZeRO2/3 base + resume | ZeRO2 resume 已验证到 2.9B;ZeRO3 resume 已在 0.1B native/HF 路径通过(`bench/results_v100_zero3_resume_2gpu_20260703.jsonl`)。 |
| RTX 50 系 / Blackwell | 5090 full quant matrix 已补 | RTX 5090 fresh-process native MM8/MM4 speed-policy 矩阵完成(216/216 pass;1.5B/2.9B/7.2B × prompt128/512/2048 × decode128/512 × bsz1/2/4/8)。13.3B LFS 权重已拉取,但当前 converter 在 48GB RAM 主机上未产出 HF 目录,需 low-memory converter 或更大 RAM。 |
| RTX 4090 / Ada | **dense decode 全 batch达标；B1/B4 prefill 同机当次达标；量化 speed lane 达标** | 0.4B dense fp16 native-graph decode bsz1/2/4/8=`795.7/1469.5/2585.7/3185.3 tok/s`，为同 checkpoint Albatross 的 `1.007x/1.016x/1.008x/1.418x`，四个 graph 并存的 32-step greedy/HF fallback 通过。no-cat sequence mix + fused ReLU² 后，fixed-shape prefill prompt512 bsz1/4=`64.51k/107.87k tok/s`；B4 对同机当次 Albatross=`1.007x`，对历史最强 `117.79k`=`0.916x`。W8 speed lane payload `0.926x`、prefill `1.011x` fp16、decode bsz1/2/4/8=`1.001x-1.020x`;W4 speed lane payload `0.891x`、prefill `1.010x` bf16、decode 已测 bsz1/4=`1.043x/1.058x`。W4 memory lane仍保留 payload `0.399x`。剩余重点是历史 prefill 高水位与 full-memory quant prefill。见 BENCHMARK 4090 段。 |
| A100 / Ampere | A100 40GB 大模型验证已补 | 见 [`docs/validation/A100_HF_VALIDATION.md`](docs/validation/A100_HF_VALIDATION.md):0.1B 基线 + 0.4B/1.5B/2.9B/7.2B smoke、fp16/bf16 batch sweep、8/4-bit quant 功能/显存与 interim speed、Trainer/SFT/DPO、HF checkpoint resume、2×A100 ZeRO-2/3 base、ZeRO2 resume。A100 80GB 未测。 |
| A800 / Ampere | issue #98 验证已补 | 见 [`docs/validation/A800_HF_VALIDATION.md`](docs/validation/A800_HF_VALIDATION.md):0.1B generate/API/PEFT/alignment/Trainer/SFT/DPO/GRPO、0.4B/1.5B/2.9B/7.2B/13.3B native mm8/mm4、7.2B fp16 smoke、13.3B bnb W8/W4 量化 smoke、单卡和 2×A800 ZeRO-2/3 base + resume。mm8/mm4 省显存但 1.5B+ 慢于 fp16,不提升量化默认。 |
| RTX A6000 / Ampere sm_86 | **Issue #115 验证完成** | 1×RTX A6000 48GB:0.1B core smoke、0.4B/1.5B/2.9B/7.2B fp16/bf16 smoke + batch sweep、bnb W8/W4 功能/显存、native mm8/mm4 decode telemetry、Trainer/SFT/DPO/resume 通过;2×RTX A6000:ZeRO-2/3 base + resume 到 2.9B 通过。量化速度未达标,见 BENCHMARK A6000 段。 |
| H100 / Hopper | 待补 | 高端吞吐、bf16、量化、大模型行。 |
| Pascal / 老 NVIDIA | GTX 1080 Ti smoke 已补 | 0.1B fp16 默认 native/no-FLA fallback、bnb 8/4-bit 量化加载与 decode speed、native mm8/mm4 decode speed、bench_speed、bsz 1/2/4 batch sweep 通过;0.4B fp16 bench_speed 通过;训练未跑。bnb 慢于 fp16,但 native mm8/mm4 在 `lm_head` 量化下接近 fp16 decode。 |
| AMD / ROCm | 开放 | 先做 native / 无 FLA 纯 PyTorch 兼容,再考虑 kernel。 |
| Apple Silicon / MPS | MPS/MLX 初始可跑;CoreML stateful 0.1B correctness pass | 见 [`docs/hardware/APPLE_SILICON.md`](docs/hardware/APPLE_SILICON.md):native/no-FLA、PEFT/Trainer/TRL、MLX recurrent/session/quant/Metal、CoreML stateful prefill/decode 与 Qwen3.5 验收入口。生产 fused Metal、CoreML 量化和确认 ANE placement 仍开放。 |
| CPU fallback | 部分 / 实验 | 保持无 CUDA import + tiny native 测试绿灯。 |

新增卡结果时至少记录:GPU 名称与数量、驱动 / CUDA 或 ROCm / PyTorch / Transformers / PEFT / TRL / DeepSpeed 版本、模型尺寸与 dtype、所用命令、`bench/results.jsonl` 行(支持 `--results` 时)、`BENCHMARK.md` 或 PR body 的一句说明。

## 当前缺口(摘要)

完整缺口清单见 [`docs/reference/HF_CRITERIA.md`](docs/reference/HF_CRITERIA.md) §3。当前重点:

- **ZeRO3 checkpoint resume** V100 0.1B native/HF smoke 和 A800 0.4B smoke 已闭合;下一步扩到更多大模型矩阵。
- **A100 80GB 验证** 当前集群不可用;A100 40GB 大模型 smoke/training/ZeRO 证据已补。
- 量化速度分 lane: bnb W8/W4 仍慢于 fp16;native mm8/mm4 `speed` policy 的 RTX 5090 fresh-process full matrix 已有 216/216 pass、footprint 全部下降、same-next 144/144、多行速度超过 fp16 的证据;更大 footprint 下降的 `memory` policy 和 7.2B 大压力形状仍需 fused/native 量化矩阵。
- Albatross / RWKV-LM 的 V100 dense decode/prefill P1 与 native W8/W4 speed lane 已闭合;仍需 P2/P3、更大模型、full-memory quant 与跨卡复现(见 [`docs/performance/FUSED_BACKEND.md`](docs/performance/FUSED_BACKEND.md))。
- 更多卡覆盖(Turing / H100 / 5090 / AMD / Apple Silicon)与更长训练吞吐。

## 下一步去哪

- 实操路线图:[`HF_TODO.md`](HF_TODO.md)
- 性能数字:[`BENCHMARK.md`](BENCHMARK.md)
- A100 训练/量化/ZeRO 验证矩阵:[`docs/validation/A100_HF_VALIDATION.md`](docs/validation/A100_HF_VALIDATION.md)
- V100 训练/量化/ZeRO 验证矩阵:[`docs/validation/V100_HF_VALIDATION.md`](docs/validation/V100_HF_VALIDATION.md)
- 验收门禁 + 已完成 + 缺口:[`docs/reference/HF_CRITERIA.md`](docs/reference/HF_CRITERIA.md)
- 性能 kernel 路线:[`docs/performance/FUSED_BACKEND.md`](docs/performance/FUSED_BACKEND.md)
