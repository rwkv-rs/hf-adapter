# RWKV-7 HF Adapter TODO / 贡献者路线图

贡献者实操 TODO。范围严格 **HF 适配**:Transformers 加载/生成、Trainer、PEFT、TRL、DeepSpeed、HF state-cache helper、量化 HF 推理、硬件验证、生产就绪证据。

不要把 native vLLM/SGLang 工作放进本 TODO,那是独立项目。

> 缺口总览见 [`docs/reference/HF_CRITERIA.md`](docs/reference/HF_CRITERIA.md) §3、性能数字见 [`BENCHMARK.md`](BENCHMARK.md)、性能 kernel 路线见 [`docs/performance/FUSED_BACKEND.md`](docs/performance/FUSED_BACKEND.md)、整体状态见 [`HF_STATUS.md`](HF_STATUS.md)。本文件是这些文档的**实操展开**(做什么 + 怎么做 + 完成定义),不重复其内容。

## 贡献规则

1. 改动限定在 HF 适配,除非 PR 明确是 benchmark 或纯文档更新。
2. 每个行为变更都补/更新测试。
3. GPU 工作在命令支持 `--results` 时,把可复现证据写入 `bench/results.jsonl`。
4. 硬件 PR 必须写明:卡名、驱动、CUDA/ROCm、PyTorch、dtype、模型尺寸、确切命令。
5. 为新卡优化时不得回退 V100 基线。
6. 测试因缺 GPU/库而 optional 时,skip 要显式,并保持 CPU / 无 CUDA import 路径绿灯。

## P0:闭合 HF 验收证据

### 1. 大模型训练矩阵 【V100 + A100 40GB 主体已补,继续扩卡/长训】

小模型 PEFT/Trainer/TRL smoke 已有。2026-07-02 已补 V100 大模型矩阵,详见 [`docs/validation/V100_HF_VALIDATION.md`](docs/validation/V100_HF_VALIDATION.md);同日已补 A100 40GB 0.4B/1.5B/2.9B/7.2B 矩阵,详见 [`docs/validation/A100_HF_VALIDATION.md`](docs/validation/A100_HF_VALIDATION.md)。2026-07-03 已补 2×V100 0.1B ZeRO3 checkpoint resume smoke。下一步不是重复跑 0.1B,而是把 ZeRO3 resume 扩到 0.4B+/A100、A100 80GB、长 step/吞吐和更多卡补成强证据。

| 模型尺寸 | PEFT | SFT | DPO | GRPO | ZeRO-2 | ZeRO-3 | 备注 |
|---|---|---|---|---|---|---|---|
| 0.4B | pass | pass | pass | pass | pass + resume | base pass | V100 + A100 40GB 主体完成;下一步扩展 ZeRO3 resume 到 0.4B+。 |
| 1.5B | pass | pass | pass | pass | pass + resume | base pass | V100 + A100 40GB 主体完成;继续补吞吐/更长 step。 |
| 2.9B | pass | pass native / A100 pass | pass native / A100 pass | pass native | pass + resume | base pass | V100 native/no-FLA 兼容路径通过;A100 40GB Trainer/SFT/DPO/ZeRO 已补。 |
| 7.2B | PEFT pass | A100 pass | A100 pass | 待大卡/长训 | A100 pass + resume | A100 base pass | V100 单卡受限;A100 40GB smoke 已补,仍需大模型 ZeRO3 resume/长训/80GB。 |

完成定义:

- 有限 loss;
- trainable 参数变化;
- 无静默 NaN/Inf;
- 记录命令与模型路径;
- 支持时追加 `bench/results.jsonl` 行;
- 在 `BENCHMARK.md` 或 PR body 加摘要。

### 2. ZeRO checkpoint resume 【ZeRO2 已补到 A100 7.2B,ZeRO3 V100 smoke 已闭合】

`tests/test_deepspeed_resume_smoke.py` 已新增,并在 2×V100 上验证 ZeRO2 resume 到 2.9B、在 2×A100 40GB 上验证 ZeRO2 resume 到 7.2B。2026-07-03 又在 2×V100 上验证 0.1B native/HF ZeRO3 checkpoint resume (`bench/results_v100_zero3_resume_2gpu_20260703.jsonl`)。后续目标是把同一流程扩到 0.4B+ 和 A100 大模型矩阵:

1. ZeRO-3 下初始化 HF Trainer + PEFT LoRA;
2. 训练一步;
3. 保存 checkpoint;
4. 释放旧模型 / trainer / engine;
5. 重新初始化 model / trainer;
6. 从 checkpoint resume;
7. 再训一步;
8. 断言有限 loss、预期 global step、trainable 参数 delta。

已有文件:`tests/test_deepspeed_resume_smoke.py`
已有结果类型:`deepspeed_resume_smoke`
已修复 native/HF 路径的关键点:ZeRO3 参数分片依赖 module pre-forward hook gather,所以 native batched loop 必须通过 attention/FFN `Module.__call__` 访问 raw TMix/CMix 参数,不能只把 module 对象传给 functional helper。A100 40GB 大模型仍需复测 checkpoint epilogue dtype-mismatch。

### 3. 一键 HF 验收脚本 【已完成,继续使用】

已合入脚本,让新贡献者不用读每个测试文件就能复现当前验收状态。

已完成脚本:

- `scripts/run_hf_acceptance.sh`
- `scripts/run_hardware_smoke.sh`
- `scripts/run_hf_training_matrix.sh`
- `scripts/run_zero_training_smoke.sh`

继续使用方式:

- card issue:优先跑 `scripts/run_hardware_smoke.sh`;
- 训练矩阵:优先跑 `scripts/run_hf_training_matrix.sh`;
- ZeRO base smoke:跑 `scripts/run_zero_training_smoke.sh`;
- 新脚本/新参数必须继续支持 `MODEL`、`RESULTS`、`CUDA_VISIBLE_DEVICES`、dtype override 和环境元数据打印。

### 4. 卡适配矩阵 【4090 进行中】

搭一个可复现的卡矩阵。目标是常见专业 / 消费硬件上的生产级信心,而不只是一台服务器。

每卡最小 smoke 优先使用一键脚本:

```bash
MODEL=/path/to/model DEVICE=cuda DTYPE=fp16 bash scripts/run_hardware_smoke.sh
```

需要拆分定位时再跑原始命令:

```bash
python tests/smoke_hf_generate.py --model /path/to/model
python tests/test_hf_api_contract.py --model /path/to/model
python tests/test_quantized_inference.py --model /path/to/model --device cuda
python bench/bench_speed.py --hf-dir /path/to/model --backend hf --dtype fp16 --device cuda --results bench/results.jsonl
python bench/bench_batch_sweep.py --hf-dir /path/to/model --dtype fp16 --device cuda --results bench/results.jsonl
```

> **量化验证必须同时报 native `mm8`/`mm4` 的 decode tok/s + footprint,不只是 bnb W8/W4。**
> `test_quantized_inference.py` 只测 bnb(通用慢路径);仓库自有量化是 native `mm8`/`mm4`(融合 Triton dequant-GEMV,PR #85/#88),要单独出速度行对比 fp16 / bnb:
>
> ```bash
> # 加载 → 量化 → bench_speed 计时 decode,追加到 bench/results.jsonl
> python -c "from rwkv7_hf.native_quant_mm8 import quantize_model_mm8"   # mm4: native_quant_mm4
> # quantize_model_mm8(model, min_params=8_000_000)  # 大层(含 lm_head)换 int8;然后 bench_speed
> ```
>
> 老卡(sm_61 Pascal 等)上 `mm8`/`mm4` 若因 Triton(`.evict_last` 需 sm_70+)跑不通,记「该卡 mm8/mm4 不可用,bnb 是唯一量化 fallback」也算有效结论,不用硬调。

可训练卡还应跑:

```bash
python tests/test_peft_lora.py --model /path/to/model --device cuda --attn-mode fused_recurrent
python tests/test_hf_training_smoke.py --model /path/to/model --device cuda --attn-mode fused_recurrent --backend both --results bench/results.jsonl
python tests/test_hf_rl_training_smoke.py --model /path/to/model --device cuda --attn-mode fused_recurrent --backend dpo --results bench/results.jsonl
```

多卡 / 多机还应跑:

```bash
torchrun --standalone --nproc_per_node=2 tests/test_deepspeed_training_smoke.py \
  --model /path/to/model \
  --zero-stage both \
  --train-dtype fp32 \
  --max-steps 1 \
  --batch-size 1 \
  --gradient-accumulation-steps 1 \
  --max-length 32 \
  --results bench/results.jsonl
```

卡目标:

| 优先级 | 卡族 | 目标 |
|---|---|---|
| P0 | V100 1×/2× | 保持基线绿灯;ZeRO2 resume 和 0.4B/1.5B/2.9B 矩阵已补;ZeRO3 resume 已有 0.1B 2×V100 smoke,继续扩到 0.4B+。 |
| P0 | A100 | A100 40GB 已补 0.1B 基线 + 0.4B/1.5B/2.9B/7.2B smoke、fp16/bf16 batch sweep、8/4-bit 量化速度/显存、bf16 Trainer/SFT/DPO、HF checkpoint resume、2×A100 ZeRO-2/3 base、ZeRO2 resume;继续补 A100 80GB、大模型 ZeRO3 resume、长 step 与吞吐矩阵。 |
| P0 | RTX 4090 | **进行中** —— 补常见消费级 Ada 证据。 |
| P1 | H100 | 补 Hopper 高端吞吐与 bf16 / 量化行。 |
| P1 | RTX 5090 / 50 系 | 补 Blackwell 消费级验证与回归行。 |
| P1 | Pascal / Turing | Pascal GTX 1080 Ti 0.1B fp16 smoke/bnb W8-W4 + native mm8-mm4 quant speed/bsz1-4 + 0.4B fp16 bench 已补;Turing 仍待验证 fallback、量化与老卡约束。 |
| P2 | AMD ROCm | 先做 native / 无 FLA 兼容并记录缺口。 |
| P2 | CPU | 保持 tiny native / 无 FLA import 与 API 测试可用。 |

## P1:生产化 HF 体验

### 5. Accelerate / `device_map` / offload

- `device_map="auto"` smoke;
- 大模型手动多卡分层 placement smoke;
- CPU offload smoke;
- 明确文档:分片时 fast-token shortcut 何时被禁用;
- 单卡 / 多卡 / offload 加载示例。

### 6. PEFT / QLoRA 矩阵

- 记录推荐 LoRA target module;
- 验证 adapter merge 后 `generate()`;
- 卡支持时补 QLoRA 8/4-bit 训练 smoke;
- 记录 QLoRA 加载的显存差。

### 7. TRL 训练加固

- 更长的 SFT/DPO/GRPO smoke;
- 每种 trainer 的 checkpoint save/load;
- 更清晰地处理 fp16/bf16/fp32 训练 dtype 行为;
- 小型公开 toy 数据集示例。

### 8. Hub 与示例

- 最小推理示例;
- 最小 LoRA 示例;
- 最小 SFT 示例;
- 最小 DPO/GRPO 示例;
- model-card 说明:RWKV recurrent state cache 与 Transformer KV cache 的区别;
- `trust_remote_code=True` 加载说明与依赖。

### 9. CI 与打包

- 无 CUDA import 测试;
- CPU tiny-model API 测试;
- 转换 / 配置测试;
- 可选 GPU smoke benchmark workflow;
- training / 量化 / dev docs 的 dependency extras。

### 9A. Apple Silicon / MPS / MLX

已完成首批 M5/16GB 证据:0.1B load/forward/generate、0.4B fp32/fp16 load/forward/短 generate、0.4B fp32/fp16 prompt 16/64/128 generation sweep、0.4B fp16 prompt 256/512 sweep、tiny native train、tiny PEFT/Trainer、0.1B 和 0.4B 真模型 PEFT LoRA backward + HF Trainer + TRL SFT/DPO/GRPO、0.4B Trainer/TRL 2-step、1.5B fp16 load/forward/短 generate + prompt16/64/128/256/512 sweep + prompt512/new8、1.5B fp32 PEFT LoRA manual backward + HF Trainer + TRL SFT/DPO/GRPO 1/2/3/5/10-step、Apple native MM8/MM4 tiny + 0.1B/0.4B/1.5B MPS min-params smoke、初始 MLX recurrent reference smoke(tiny MLX save/load/matmul、0.1B HF projection matmul、selected HF safetensor → MLX bundle、tiny MLX/Torch recurrent parity、state-cache select/chunked-prefill/session、多会话交错 session、tokenizer prompt/API、dynamic-batch state select、0.1B/0.4B/1.5B full MLX recurrent prefill/generate、scripts/mlx_generate.py 文本生成 CLI、MLXGenerationSession 分段 decode/session smoke、MLXGenerationSessionBatch 多会话交错 decode + repeat pressure smoke、MLX prompt/decode sweep + repeat pressure(0.1B prompt128/256 decode4/8 repeat=2;0.4B/1.5B prompt128/256 decode4/8 repeat=1))。继续补:

- 扩展 0.4B Apple 到更长 decode 和 3+ step 稳定性行;
- 把 1.5B 从 prompt512/new8 和 10-step 继续扩到更长 decode、>10-step 训练,并持续记录 memory-pressure;
- 补 M-series Pro/Max/Ultra 的长上下文、显存峰值、tok/s 行;
- 把 MLX recurrent reference/session helper 继续扩到 0.4B/1.5B 更高并发/更长 repeat 的 memory-pressure 遥测;
- 评估 RafaelUI Metal WKV7 / MLX 路线是否做 sibling backend,并把 correctness-first MLX inner loop 替换为 fused WKV/packed quant kernel;
- Apple native MM8/MM4 功能/min-params smoke 已有;生产级 W8/W4 仍不走 bitsandbytes,需要 MLX/Metal packed quant + fused/dequant speed path。

## P2:闭合性能与量化缺口

> 路线与数字权威见 [`docs/performance/FUSED_BACKEND.md`](docs/performance/FUSED_BACKEND.md) 与 [`BENCHMARK.md`](BENCHMARK.md);本节只列实操动作。

### 10. Albatross / RWKV-LM 速度缺口

继续走 fast-token / native-graph 路线,而非堆 wrapper 层。当前路线:`native_graph → fused fp16 kernel → fused W8/W4 kernel`(详见 FUSED_BACKEND)。需补:

- 同卡同 checkpoint 的 prefill / decode / batch-size sweep;
- latency 与峰值显存行;
- cache 命中率行;
- `bench/analyze_results.py` / `bench/check_results.py` 里的明确 ratio gate。

### 11. 量化速度

现状:W8/W4 加载与显存下降可用,速度未达生产级(详见 BENCHMARK 量化段 + FUSED_BACKEND quant target)。需补:

- native packed W8/W4 权重布局;
- fused dequant + projection 路径;
- V100 / A100 / 4090 / H100 / 50 系卡专项调优;
- 接近 llama.cpp Q*_K_M 实用量级的质量 telemetry;
- 速度目标:W8/W4 在常见卡上不慢于 fp16。

### 12. 训练吞吐

- 尽可能对标 HF Trainer / PEFT 吞吐与 RWKV-LM 训练;
- batch-size 与序列长度 sweep;
- activation / checkpointing 显存行;
- ZeRO-2/3 吞吐与显存行。

## P3:upstream 与长期兼容

### 13. Native Transformers 方向

长期 upstream 形态(详见 [`docs/reference/HF_CRITERIA.md`](docs/reference/HF_CRITERIA.md) §3 缺口 5):

```text
src/transformers/models/rwkv7/
  configuration_rwkv7.py
  modeling_rwkv7.py
  tokenization_rwkv7.py
  convert_rwkv7_original_to_hf.py
```

需补:

- 不强依赖 FLA 的 pure PyTorch / reference 路径;
- 可选 CUDA / Triton kernel;
- CPU 与 AMD 兼容方案;
- Transformers model common tests;
- generation 测试;
- tokenizer / model-card 文档。

### 14. HF 兼容 speculative decoding

现状:`rwkv7_speculative_generate`(`modeling_rwkv7.py`)已实现 batch=1、greedy、block verify + 不匹配时从接受前缀重建 draft cache;`tests/test_speculative_decode.py` 校验 same-model draft `acceptance=1.0` 且与 `generate()` 逐 token 一致,`--draft-model` 可换小 draft(0.1B→0.4B,acceptance ~0.78)。verify 当前仅 greedy(`do_sample=False`)。

> **准则(守红线)**:训练化的 draft **复用现有 verify,不改它**;增强只通过「加载不同的 draft checkpoint」(即现有 `--draft-model` / `draft_model=` 开关)生效。默认行为、函数签名、verify 路径不变 —— 现有 0.1B→0.4B 路径/测试/benchmark 行永远是安全回退点,训练过的 draft 随时能关掉且零损失。

需补:

- **draft 训练化(提 acceptance)**:`scripts/` 下加**独立**训练脚本,用小 RWKV 对 target logits 做 **LoRA 对齐**(推荐,保通用泛化)或从 target cache 蒸馏;recipe 参考 DeepSeek DeepSpec/SpecForge 的 draft 训练 + acceptance 评估([github.com/deepseek-ai/DeepSpec](https://github.com/deepseek-ai/DeepSpec))。脚本不进核心 forward 路径,不引入硬依赖。注:DSpark/DFlash/Eagle3 的 draft 架构是 transformer-KV,**不能直接用于 RWKV recurrent state**,draft 必须是小 RWKV。
- 训练后的 draft 经现有 `--draft-model` 加载;结果按 `draft=trained` vs `draft=off-the-shelf` 写入 `bench/results.jsonl`,不覆盖旧行。
- 更多 draft / target 尺寸组合、更长 prompt 与更大 batch。
- **verify 采样正确性(单独子任务,慎动 verify)**:若加采样,须按 speculative-sampling 公式做 acceptance correction,并**先补分布正确性测试再改 verify** —— 这是唯一会触碰 verify 的工作,单独立项。
- 对 target greedy 的正确性校验;文档:speculative decoding 何时有益 / 有害。

## 贡献者 PR checklist

开 PR 前包含:

- [ ] 改了什么、为什么。
- [ ] 确切命令。
- [ ] GPU 工作的软硬件版本。
- [ ] 结果行或 benchmark 摘要(适用时)。
- [ ] 行为 / 支持矩阵 / TODO 状态变化时同步更新文档。
- [ ] 确认改动是 HF 适配范围。

纯文档 PR 至少跑:

```bash
git diff --check
```
