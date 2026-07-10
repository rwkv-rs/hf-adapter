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

### 4. 卡适配矩阵 【V100 production-close 已完成，继续扩卡】

V100 exact-sm70 已完成 0.1B/0.4B/1.5B × bsz1/2/4/8 dense
decode/prefill Albatross P1、native W8/W4 speed/footprint、正确性和 cache
handoff 验收。结果与 fail-closed gate 见
[`bench/v100_production_close_20260711/README.md`](bench/v100_production_close_20260711/README.md)。
下一步不再重复 V100 小矩阵，优先扩到 Turing/Hopper/ROCm、更大模型与
full-memory quant lane。

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


### 4.1 Apple / Qwen3.5 同机对齐矩阵 【0.8B 首批实测已落,性能 gap】

目标是把“Apple / 移动端超过 Qwen3.5”从口号变成可复现 evidence。入口文档见
[`docs/hardware/QWEN35_APPLE_BASELINE.md`](docs/hardware/QWEN35_APPLE_BASELINE.md),
共用 runner 为 `bench/run_qwen35_apple_baseline.py`;一键 wrapper 为 `scripts/run_qwen35_apple_acceptance.sh`。

完成定义:

- 同一台 Apple 设备、同一 prompt 文本,同时记录 Qwen3.5 MLX/Ollama 与 RWKV-7 MLX/CoreML 行;
- 覆盖 `qwen3.5:0.8b-mlx`、`2b-mlx`、`4b-mlx`、`9b-mlx` 至少前三档;
- RWKV 覆盖 0.4B / 1.5B / 2.9B,并分别记录 fp16、W4/Metal、后续 CoreML/LUT/INT4;
- JSONL 字段包含 TTFT、prefill tok/s、decode tok/s、显存/MLX peak/cache、量化 backend、chunked prefill diff、seen-token 检查;
- 在 PR body 和 `BENCHMARK.md` 摘要里只根据实测行 claim,不得把 harness 存在当作性能达成;
- 用 `bench/compare_qwen35_apple_baseline.py` 生成 `qwen35_apple_baseline_comparison` gate 行,通过后再写“超过”;
- `scripts/export_rwkv7_coreml.py` 已提供 import-safe manifest、full-logits 兼容导出和真实 `stateful-multifunction` prefill/decode 导出;`bench/run_coreml_apple_baseline.py` 已能做 MLState transfer、chunk split、HF greedy parity。M5 0.1B/0.4B fp32-compute 正确性行已通过,但 1.5B、生产长度上下文、LUT4/INT4 质量、确认 ANE placement 和 Qwen3.5 2B+ 完整同机矩阵仍未完成;在此之前,移动端“超过”仍不能正式 claim。
- M5/16GB 首批 `qwen3.5:0.8b-mlx` vs RWKV-7 0.4B 同文本实测已落(`bench/results_qwen35_apple_m5_20260710_{fp16,w4}.jsonl`)。修复了 Ollama thinking 空响应、prompt-cache 伪 prefill、TTFT load 混算和跨 prompt MLX 模型未释放导致的 2x 内存污染。128/512 chars + decode32 保守 repeat 下,fp16 decode≈0.82x/0.92x Qwen,prefill≈0.090x/0.049x;W4 decode≈0.62x/0.60x,prefill≈0.064x/0.030x,但 RWKV peak≈0.568x fp16。已记录 Qwen `/api/ps` loaded memory≈1.09-1.11GB,但它不是 peak;1k/4k/8k、2B+ 和正式 quality rubric 仍待补,当前不能 claim “超过”。

最小 dry-run:

```bash
PYTHONPATH=. python bench/run_qwen35_apple_baseline.py \
  --dry-run \
  --prompt-target-chars 1024,4096 \
  --decode-lengths 128,512 \
  --qwen-models qwen3.5:0.8b-mlx,qwen3.5:2b-mlx \
  --rwkv-mlx-models /path/to/rwkv7-g1d-0.4b-hf,/path/to/rwkv7-g1g-1.5b-hf
```

CoreML export 原型 dry-run:

```bash
PYTHONPATH=. python scripts/export_rwkv7_coreml.py \
  /path/to/rwkv7-g1g-1.5b-hf \
  exports/rwkv7-g1g-1.5b-coreml \
  --dry-run \
  --state-mode wkv-coreml \
  --quantization lut4 \
  --results bench/results_qwen35_apple_baseline.jsonl
```

一键验收 dry-run / live 入口:

```bash
DRY_RUN=1 \
RWKV_MLX_MODELS=/path/to/rwkv7-g1d-0.4b-hf,/path/to/rwkv7-g1g-1.5b-hf \
COREML_EXPORT_MODELS=/path/to/rwkv7-g1g-1.5b-hf \
scripts/run_qwen35_apple_acceptance.sh

PULL_QWEN=1 \
RWKV_MLX_MODELS=/path/to/rwkv7-g1d-0.4b-hf,/path/to/rwkv7-g1g-1.5b-hf \
COREML_EXPORT_MODELS=/path/to/rwkv7-g1g-1.5b-hf \
RESULTS=bench/results_qwen35_apple_baseline.jsonl \
scripts/run_qwen35_apple_acceptance.sh
```

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

已完成首批 M5/16GB 证据:0.1B load/forward/generate、0.4B fp32/fp16 load/forward/短 generate、0.4B fp32/fp16 prompt 16/64/128 generation sweep、0.4B fp16 prompt 256/512 sweep、tiny native train、tiny PEFT/Trainer、0.1B 和 0.4B 真模型 PEFT LoRA backward + HF Trainer + TRL SFT/DPO/GRPO、0.4B Trainer/TRL 2-step、1.5B fp16 load/forward/短 generate + prompt16/64/128/256/512 sweep + prompt512/new8、1.5B fp32 PEFT LoRA manual backward + HF Trainer/SFT 20-step + TRL DPO/GRPO 12-step、Apple native MM8/MM4 tiny + 0.1B/0.4B/1.5B MPS min-params smoke、初始 MLX recurrent reference smoke(tiny MLX save/load/matmul、0.1B HF projection matmul、selected HF safetensor → MLX bundle、tiny MLX/Torch recurrent parity、state-cache select/chunked-prefill/session、多会话交错 session、tokenizer prompt/API、dynamic-batch state select、0.1B/0.4B/1.5B full MLX recurrent prefill/generate、scripts/mlx_generate.py 文本生成 CLI、MLXGenerationSession 分段 decode/session smoke、MLXGenerationSessionBatch 多会话交错 decode + 0.1B/0.4B/1.5B 3-session repeat pressure smoke + 0.4B/1.5B 4-session rounds4,4 repeat4 pressure smoke + 0.4B 6-session repeat5 / 1.5B 5-session repeat2 更高并发 smoke + 0.4B 8-session rounds8,8 repeat2 / 1.5B 5-session rounds8,8 repeat2 长 session pressure、MLX prompt/decode sweep + repeat pressure(0.1B prompt128/256 decode4/8 repeat=2;0.4B/1.5B prompt128/256 decode4/8 repeat=1;0.4B/1.5B prompt256/512 decode16/32 repeat=1;0.4B/1.5B prompt1024/decode64 repeat=1;0.4B/1.5B prompt2048/decode128 repeat=1)、MLX packed W8/W4 affine quant path smoke(0.1B lm_head;0.4B/1.5B 各 49 FFN/lm_head modules;1.5B prompt32/decode4;W8/W4 footprint≈0.50/0.25)、初版 MLX/Metal WKV custom-kernel seam smoke(0.1B/0.4B/1.5B, `--wkv-backend metal`)、初版 MLX/Metal fused W8/W4 dequant-projection seam smoke + pressure matrix(0.1B 短 smoke;0.4B/1.5B prompt128/256 decode4/8 + prompt512/1024 decode16 + prompt2048/decode128;`--quant-backend metal/auto`;W4/W8 footprint≈0.25/0.50;0.4B W4 auto prompt2048/decode128 prefill/decode≈60.61/59.73 tok/s,decode≈1.25x fp16;1.5B W4 auto prompt2048/decode128 prefill/decode≈27.64/20.42 tok/s,decode≈0.75x fp16))。继续补:

- 扩展 0.4B Apple 到 3+ step / 更长训练稳定性行;
- 把 1.5B 从 prompt2048/decode128、Trainer/SFT 20-step 和 DPO/GRPO 12-step 继续扩到更长生产式训练/解码,并持续记录 memory-pressure;
- 用 `scripts/run_qwen35_apple_acceptance.sh` 补真实同机 Qwen3.5 0.8B/2B/4B/9B vs RWKV MLX/CoreML rows,并生成 speed/memory comparison gate 和 `bench/score_qwen35_quality.py` quality gate 后再 claim “超过”;
- 补 M-series Pro/Max/Ultra 的长上下文、显存峰值、tok/s 行;
- CoreML stateful decode/prefill multifunction 与 0.1B/0.4B fp32 correctness 已闭合;0.4B 已扩到 67-token prompt/32-token decode 且 HF greedy 32/32、chunk/state diff=0。INT8 包体≈0.45x/0.36x 且对齐,但 warmed decode 仍只是近似持平;0.1B INT4/LUT4 包体≈0.38x/0.13x 但短 greedy 未对齐。继续扩到 1.5B、生产长度 prompt/decode、校准/混合 W4,修复 fp16-compute mismatch,并补确认 ANE placement 的 benchmark 行;
- 把 MLX recurrent reference/session helper 继续扩到更长上下文、更长 repeat、更多不同 prompt 分布的 memory-pressure 遥测，并继续验证新加的 `SESSION_BACKEND=batched|auto` 等长 round MLX session batching 路径;
- 初版可选 MLX/Metal WKV seam 已存在;继续把它扩成 production fused WKV/projection/packed quant kernel,并评估 RafaelUI Metal WKV7 / MLX 路线是否做 sibling backend;
- Apple native MM8/MM4 功能/min-params smoke、初始 MLX packed W8/W4 affine dequant-matmul path、初版 MLX/Metal W8/W4 fused dequant-projection seam 已扩到 0.4B/1.5B prompt2048/decode128，并补了同形状 fp16 Metal ratio gate:prompt512/1024 decode16 下 0.4B W8/W4 decode≈0.79x/0.81x fp16、peak≈0.71x/0.57x;1.5B W8/W4 decode≈0.75x/0.84x fp16、peak≈0.70x/0.55x。新增 prompt2048/decode128 ratio 行:0.4B W8/W4 decode≈0.88x/1.04x fp16、peak≈0.71x/0.56x;1.5B W8/W4 decode≈0.68x/0.73x fp16、peak≈0.70x/0.54x。优化后的 W4 `--quant-backend auto` 行记录 `metal=202885`:0.4B W4 auto prefill/decode≈60.61/59.73 tok/s、decode≈1.25x fp16、peak≈0.56x;1.5B W4 auto prefill/decode≈27.64/20.42 tok/s、decode≈0.75x fp16、peak≈0.54x。新增 prompt4096/decode256/chunk1024 gate:0.4B fp16 prefill/decode≈94.08/75.38 tok/s、W4 auto≈62.01/55.29 tok/s、peak≈515MB(≈0.56x);1.5B fp16≈35.34/33.21 tok/s、W8/Metal≈22.52/20.54 tok/s、peak≈2147MB(≈0.70x)、W4 auto≈27.40/25.46 tok/s、peak≈1677MB(≈0.54x),chunked/full max_abs=0.0。新增 1.5B prompt8192/decode512/chunk2048 gate:fp16≈27.97/26.02 tok/s、W4 auto≈22.77/21.20 tok/s、peak≈1677MB(≈0.54x)、`metal=811525`、chunked/full max_abs=0.0;direct grouped R/K/V W4 进一步扩到 prompt8192/decode1024,≈21.09/20.48 tok/s、peak≈1075MB、quantized-linear `metal=2507781`、grouped `metal=417792`、fallback=0、chunked/full max_abs=0.0,说明显存继续下降但长上下文 W4/W8 速度仍未稳定超过 fp16。quant+Metal session-batch 压力行已扩到:0.4B W8/W4 4-session repeat=2 min decode≈40.18/41.17 tok/s、6-session repeat=3 min decode≈34.33/27.14 tok/s;1.5B W8/W4 4-session repeat=1 min decode≈19.58/20.38 tok/s、5-session repeat=2 min decode≈15.60/18.87 tok/s。新增等长 round `SESSION_BACKEND=batched` W4 行:0.4B 6-session repeat=2 aggregate round min≈105.44 tok/s、0.4B 8-session rounds8,8 repeat=2 aggregate round min≈103.91 tok/s、1.5B 5-session repeat=1 aggregate round min≈32.38 tok/s、1.5B 5-session rounds8,8 repeat=2 aggregate round min≈29.63 tok/s,均保持 one-shot token/text 对齐。1.5B W8 auto rounds8,8 repeat=2 在 guard 下 aggregate round min≈18.38 tok/s。严格 W8/Metal batched 长 decode 发现 batch-exactness gap,所以 `SESSION_BACKEND=auto` 现在用 `auto_mm8_metal_batch_exactness_guard` 自动回退 sequential,并补了 0.4B/1.5B W8 safe-auto 行。新增 `mlx_session_batch_backend_compare` gate:0.4B/1.5B W4 sequential-vs-batched 均 token/text/one-shot 对齐,batched aggregate round min≈145.89/38.31 tok/s;1.5B W8 本矩阵对齐且 batched aggregate round min≈34.67 tok/s;0.4B W8 复现 mismatch,短 prompt 第 6 个 token 起偏;新增 `--trace-mismatch-logits` 后确认该点是低 margin/tie case(顺序路径 token 11/261 logits≈8.476562/8.476562 且 argmax 选 11,batched Metal 将 token 11 降到≈8.46875 后选 261,max_abs≈0.03125),并新增显式 `SESSION_BACKEND=batched_stable` 低 margin argmax 策略,0.4B W8/Metal 3-session 和 6-session strict compare 均恢复 token/one-shot 对齐(6-session batched aggregate round min≈162.12/163.72 tok/s,peak≈790MB,`metal=20378`);新增覆盖:1.5B W8/Metal rounds8,8 strict compare 通过,0.4B W8/Metal 8-session rounds8,8 repeat=2 one-shot 对齐且 aggregate round min≈184.62 tok/s,1.5B W8/Metal 5-session rounds8,8 repeat=4 one-shot 对齐且 aggregate round min≈26.11 tok/s、peak≈2311MB、`metal=50728`;同形状 1.5B W4 auto repeat=4 one-shot 对齐且 aggregate round min≈30.94 tok/s、peak≈1841MB、`metal=50728`,说明持续压力下正确性保持但吞吐下降。默认 W8/Metal auto 仍 guarded,但 `RWKV7_MLX_SESSION_AUTO_W8_STABLE=1` 可显式让 `SESSION_BACKEND=auto` 走 stable policy,0.4B W8/Metal auto row 通过且 aggregate round min≈90.73 tok/s(`metal=5126`)。新增 `--quant-backend auto` 安全路由与 backend-count telemetry:0.4B W4 auto 常规 prefill/decode row 走 Metal(长 decode `metal=202885`,strict compare `metal=4913`)且 strict compare 通过,batched aggregate round min≈78.68/69.17 tok/s;0.4B W8 auto 默认走 affine,现在 `SESSION_BACKEND=auto` 可直接 batched,one-shot 对齐且 aggregate round min≈49.76 tok/s(`affine=5126`)。新增默认关闭的模型级 grouped R/K/V quant projection seam:`RWKV7_MLX_GROUP_RKV_QUANT_PROJECTION=1` 会在 R/K/V 均量化且走 Metal 时预打包三组权重,用 grouped Metal kernel 处理三个不同 R/K/V 输入;初始 0.4B/1.5B W4 与 W8 A/B 行已证明端到端命中且 fallback=0:0.4B W4 prompt128/decode8 baseline→grouped≈39.33/38.68→44.33/41.38 tok/s,1.5B W4≈19.03/18.37→20.18/19.03 tok/s,0.4B W8≈40.32/38.93→43.01/42.80 tok/s,1.5B W8 prompt64/decode4≈17.62/17.22→19.02/17.53 tok/s;同时 0.4B W4 grouped 4-session batched rounds4,4 one-shot 对齐通过。随后默认模式切到 `direct`,直接使用三组已有 packed weights 而不复制 grouped cache;0.4B W4 direct prompt128/decode8≈43.30/42.54 tok/s、peak≈365MB,1.5B W4 direct≈20.95/19.52 tok/s、peak≈1075MB,基本消除了旧 packed cache 的峰值显存惩罚;新增 prompt512/decode16 direct 行:0.4B≈45.83/45.17 tok/s、1.5B≈20.69/19.28 tok/s,均 fallback=0;session 压力扩到 0.4B 6-session rounds8,8 repeat=2 与 1.5B 5-session rounds4,4,one-shot 对齐通过。它目前仍是 A/B 集成点,不是生产默认。结论:显存下降成立，batching seam 和 grouped projection seam 已开始验证，但生产级 W8/W4 仍需要更长 repeat/session、更深 fused WKV+projection kernel、更多 M 系列机器和端到端稳定超过 fp16 的证据。

## P2:闭合性能与量化缺口

> 路线与数字权威见 [`docs/performance/FUSED_BACKEND.md`](docs/performance/FUSED_BACKEND.md) 与 [`BENCHMARK.md`](BENCHMARK.md);本节只列实操动作。

### 10. Albatross / RWKV-LM 速度缺口

继续走 fast-token / native-graph 路线,而非堆 wrapper 层。当前路线:`native_graph → fused fp16 kernel → fused W8/W4 kernel`(详见 FUSED_BACKEND)。需补:

- V100 0.1B/0.4B/1.5B 的同卡同 checkpoint prefill/decode/bsz1/2/4/8 已闭环;继续补 2.9B/7.2B 与 4090/A100/H100/AMD 同口径矩阵;
- final MATH500 acceptance runner 已补(`bench/run_math500_final_acceptance.py` + `scripts/run_math500_final_acceptance.sh`): 自动 best-bsz sweep → full avg@64 → Albatross summary gate → uncheatable compression alignment;接下来要在 5090/H100 等正式机器上产出 artifact;
- V100 faster3a v3a 的严格同模型/同卡 baseline 已补;继续补 Albatross 更新路径和其他 GPU，避免跨卡或历史数字比较;
- V100 decode latency、峰值显存、public API/runner diff 与 cache 命中率行已补;继续扩到长稳态和跨卡;
- `bench/analyze_results.py` 已支持按模型区分 overhead/Albatross ratio；继续把全矩阵 P2/P3 约束接入 release gate。

### 11. 量化速度

现状:W8/W4 加载与显存下降可用。RTX 4090 的 0.4B TorchAO W4 memory lane
已在 bsz1/2/4/8 达到同口径 bf16 的 `1.24x-1.62x`、Albatross fp16 的
`1.17x-1.52x`，payload 降至 `0.399x` 且 cosine `>=0.999239`。新增 fixed-shape
prefill graph 后，W8 `speed` lane 达到 payload `0.926x`、prefill `1.011x`
fp16、decode bsz1/2/4/8 `1.001x-1.020x`;W4 `speed` lane 达到 payload
`0.891x`、prefill `1.010x` bf16、decode 已测 bsz1/4 `1.043x/1.058x`。
因此 4090 已有“所有推理阶段不慢于 w16”的适度省显存 lane；`memory` lane
仍需 fused quant prefill 才能同时保持约 `0.4x` payload 与 prefill 不回退。RTX 5090 native MM8/MM4 `speed`
policy 已在 1.5B/2.9B/7.2B 完成 fresh-process full matrix
(216/216 pass,quantized same-next 144/144,footprint 全部下降,多行速度超过
fp16)。但 7.2B 大 bsz/长 prompt 压力行仍低于 fp16;`memory` policy/通用 bnb
仍慢,不能作为速度达标口径(详见 BENCHMARK 量化段 + FUSED_BACKEND quant
target)。13.3B LFS 权重已拉取,当前 converter 在 48GB RAM 5090 主机上未
产出 HF 目录。需补:

- `speed` policy 在更多 GPU/更大模型上稳定 >= fp16 的证据;
- 13.3B low-memory/streaming converter,再跑 fp16/mm8/mm4 fresh-process 边界行;
- `memory` policy 的 fused dequant + projection 路径;
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
