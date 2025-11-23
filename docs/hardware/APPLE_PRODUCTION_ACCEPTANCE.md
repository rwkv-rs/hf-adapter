# Apple Silicon 生产级硬门清单

> **结论：尚未达到生产级。** 当前通过 **48 / 148** 个必选硬门。任何 `FAIL`、`MISSING` 或 `UNKNOWN` 都禁止声明 Apple 生产级完成。

Manifest 版本：`2026-07-12.7`。当前状态由已提交 JSONL 和明确登记的文件证据实时计算，不从源码功能名或说明文字推断 PASS。

本清单覆盖 RWKV-7 Hugging Face 适配中的 Apple 路线：HF/MPS 兼容与训练、MLX/Metal 生产推理、CoreML/ANE 部署。CUDA、vLLM、SGLang、DeepSpeed ZeRO 属于其他验收路线，不用 Apple 兼容结果替代。

## 完成规则

1. 只有真实机器、真实 checkpoint、可复现命令和 JSONL 数据才算证据；源码里存在某个函数不算通过。
2. 冒烟、单模型、单 bsz、单芯片结果只关闭对应原子门，不能替代完整矩阵。
3. 性能通过必须在同一配置同时通过正确性/质量门，并记录冷/热、真实峰值内存、重复次数和精确硬件。
4. W8/W4 必须实际降低峰值内存，并在所有声明支持的 Apple 卡型/模型/bsz 上不慢于 W16；否则只是功能完成。
5. CoreML 只有实际运行证据；设置 `compute_units` 不能替代 ANE 落核证明。
6. 运行严格审计：`STRICT=1 scripts/run_apple_production_acceptance.sh` 或 `python bench/check_apple_production_acceptance.py --strict`。任一硬门未通过时命令必须非零退出。

## Apple M5 Production Close 与质量 Proxy（限定范围）

最新主线已经关闭 Apple M5 16GB、batch1、chars512/decode64 的一组原子门；同 checkpoint fp16 量化质量 proxy 也在此登记，但只有证据 JSONL 被 Git 跟踪后才会 PASS。这些原子门只证明表内精确配置，**不替代**完整 bsz/上下文、M1-M4、外部 Q*_K_M 对照、CoreML/ANE 或稳定性门。

| 状态 | Gate ID | 已证明范围 | 当前证据 |
|---|---|---|---|
| ✅ PASS | `release.mlx_runtime_boundaries` | 最新 main 的 model/state/session/policy/quant/speculative 模块、兼容入口、架构文档和边界测试文件齐备；该结构门不替代任何真实模型、性能或压力证据。 | all required evidence files exist |
| ✅ PASS | `mlx.speculative_m5_b1_exact` | M5 16GB、1.5B W4 target + 0.1B W4 draft、chars512/decode64、proposal32、repeat3；最终 token 与 target greedy 完全一致，seen_tokens 正确，并记录接受率、verify/replay/fallback。 | 3/3 child proofs pass |
| ✅ PASS | `perf.m5_close_qwen_0p4_w4_b1` | 同一 M5 16GB、batch1、chars512/decode64、warmup1+repeat3：RWKV decode/prefill≥Qwen、TTFT≤1.1×、真实峰值内存≤Qwen；仅关闭该形状，不替代完整矩阵。 | 2/2 child proofs pass |
| ✅ PASS | `perf.m5_close_qwen_1p5_spec_w4_b1` | 同一 M5 16GB、batch1、chars512/decode64、warmup1+repeat3：1.5B W4+0.1B draft 的decode/prefill≥Qwen2B、TTFT≤1.1×、真实峰值内存≤Qwen，并通过 target-greedy oracle；仅关闭该形状。 | 3/3 child proofs pass |
| ✅ PASS | `quant.groupwise_w8_1p5_m5_close` | M5 batch1 chars512/decode64、warmup1+repeat2 的已提交保留行：groupwise W8 guarded compiled min decode≥46 tok/s、peak≤2.25GB，64-token token/logits/state 门通过。repeat2 是显式限定，仍不足以关闭全量 W8 生产性能门。 | 3/3 child proofs pass |
| ✅ PASS | `quant.groupwise_w4_0p4_m5_close` | M5 batch1 chars512/decode64、warmup1+repeat3 的 native groupwise W4 路径实际 dispatch packed quantized matmul；compiled validation token 一致、logits/state≤0.125，peak≤0.51GB。 | 2/2 child proofs pass |
| ✅ PASS | `quant.groupwise_w4_1p5_m5_close` | M5 batch1 chars512/decode64、warmup1+repeat3：groupwise W4 guarded compiled min decode≥55 tok/s、peak≤1.78GB，64-token token/logits/state 门通过；不替代全 bsz/长上下文/质量门。 | 3/3 child proofs pass |
| ✅ PASS | `quant.quality_fp16_proxy_m5` | M5 上以同一 RWKV checkpoint 的 fp16 为 teacher，覆盖 0.4B/1.5B、276 scored tokens、4 个 greedy prompts：W8 NLL delta≤0.02、perplexity ratio≤1.02、teacher top1≥0.95；mixed W8/W4 q4_k_m proxy 分别≤0.08/1.09/≥0.80，且权重存储下降。该门不声称 llama.cpp/GGUF Q*_K_M 等价，也不关闭 greedy/state parity。 | 7/7 child proofs pass |

## 分类状态

| 分类 | PASS | FAIL | MISSING | UNKNOWN | 总数 |
|---|---:|---:|---:|---:|---:|
| 发布、安装与证据 (`release`) | 9 | 0 | 2 | 0 | 11 |
| HF/Transformers MPS 推理 (`hf_mps_inference`) | 1 | 0 | 13 | 0 | 14 |
| HF PEFT/Trainer/TRL 训练 (`hf_mps_training`) | 7 | 0 | 7 | 0 | 14 |
| MLX/Metal 数值与模型正确性 (`mlx_correctness`) | 7 | 0 | 13 | 0 | 20 |
| MLX 生产服务能力 (`mlx_serving`) | 9 | 0 | 3 | 0 | 12 |
| 性能与 Qwen3.5 验收 (`performance`) | 5 | 0 | 13 | 0 | 18 |
| W8/W4 量化 (`quantization`) | 7 | 0 | 12 | 0 | 19 |
| CoreML/ANE 部署 (`coreml_ane`) | 0 | 0 | 17 | 0 | 17 |
| 稳定性、运维与 CI (`reliability`) | 1 | 0 | 12 | 0 | 13 |
| Apple 芯片与内存档覆盖 (`hardware`) | 2 | 0 | 8 | 0 | 10 |

## 全部硬门

### 发布、安装与证据

| 状态 | Gate ID | 验收点 | 硬判据 | 当前证据 |
|---|---|---|---|---|
| ✅ PASS | `release.clean_install` | 干净环境安装 | 在受支持 macOS/arm64 的全新虚拟环境中安装基础包及 mlx extra；不需要 CUDA、Triton 或 FLA，安装、导入和最小推理均成功。 | 3/3 child proofs pass |
| ✅ PASS | `release.import_without_cuda` | 无 CUDA/FLA 安全导入 | 未安装 CUDA、Triton、flash-linear-attention、bitsandbytes 时，rwkv7_hf 与 HF auto classes 可安全导入，Apple 后端延迟加载。 | 3/3 child proofs pass |
| ✅ PASS | `release.mlx_optional_extra` | MLX 可选依赖契约 | pyproject 的 mlx extra 能安装受支持版本；非 macOS 平台不会误装或在 import 时失败。 | 3/3 child proofs pass |
| ⬜ MISSING | `release.converter_cli` | 可复现模型转换 CLI | HF→MLX/CoreML 转换命令具备参数校验、模型/config/tokenizer 哈希、量化元数据、失败非零退出和可复现产物清单。 | no machine-verifiable proof registered |
| ✅ PASS | `release.one_command_gate` | 一键严格验收入口 | 提供单命令生产验收审计；默认不清空旧证据，STRICT=1 时只要一个硬门缺失/失败就非零退出。 | all required evidence files exist |
| ✅ PASS | `release.machine_readable_evidence` | 机器可读证据 | 每次功能/性能/质量运行输出带 axis、status、环境、模型、精度、后端、参数和测量值的 append-only JSONL。 | 3/3 child proofs pass |
| ✅ PASS | `release.safe_defaults` | 保守生产默认值 | 未对当前芯片、模型、精度和 bsz 验证的 fused/compiled/quant 路径不得自动启用；必须回退到已验证路径。 | 2/2 child proofs pass |
| ✅ PASS | `release.backend_telemetry` | 路由与回退遥测 | 每次请求可观测实际 prefill/decode/WKV/quant backend、策略原因、编译命中/回退原因和数值门结果。 | 3/3 child proofs pass |
| ⬜ MISSING | `release.artifact_roundtrip` | 保存与重载 | HF、MLX 及量化产物 save/load 后 config、tokenizer、权重、量化布局和固定输入输出保持一致。 | no machine-verifiable proof registered |
| ✅ PASS | `release.production_runbook` | 生产运维文档 | 文档覆盖安装、转换、服务、训练、量化、性能复现、故障排查、限制和硬门状态，且列出所有验收项。 | all required evidence files exist |
| ✅ PASS | `release.mlx_runtime_boundaries` | MLX runtime 模块边界材料 | 最新 main 的 model/state/session/policy/quant/speculative 模块、兼容入口、架构文档和边界测试文件齐备；该结构门不替代任何真实模型、性能或压力证据。 | all required evidence files exist |

### HF/Transformers MPS 推理

| 状态 | Gate ID | 验收点 | 硬判据 | 当前证据 |
|---|---|---|---|---|
| ✅ PASS | `hf.mps.load_generate.models` | MPS 模型加载与生成矩阵 | 0.1B、0.4B、1.5B 在 MPS 上通过真实 checkpoint load、forward 与 generate。 | proof matched 4 evidence rows |
| ⬜ MISSING | `hf.mps.dtype_matrix` | MPS dtype 矩阵 | 0.1B/0.4B/1.5B 的 fp32 与 fp16 均完成 forward、cache continuation、generate 和峰值内存验收。 | no machine-verifiable proof registered |
| ⬜ MISSING | `hf.mps.official_logits_parity` | 官方数学 logits 对齐 | 固定语料、短长序列和全部模型上与官方 RWKV-7 参考实现逐步 logits 达到声明容差，argmax/greedy 完全一致。 | no machine-verifiable proof registered |
| ⬜ MISSING | `hf.mps.state_parity` | 递归状态对齐 | 每层 time-mix/shift/WKV state 在 full、token-by-token、分块执行间达到声明容差，seen_tokens 精确。 | no machine-verifiable proof registered |
| ⬜ MISSING | `hf.mps.chunked_prefill` | HF 分块 prefill | prompt 1/127/128/129/512/1k/4k/8k tokens 的 full 与多种 chunk 切分 logits/state/greedy 对齐。 | no machine-verifiable proof registered |
| ⬜ MISSING | `hf.mps.cache_continuation` | HF state cache 连续生成 | prefill 一次后多轮 generate/forward 复用 RWKV7StateCache，与 one-shot 输出一致且缓存不被意外别名修改。 | no machine-verifiable proof registered |
| ⬜ MISSING | `hf.mps.batch_matrix` | HF 批大小矩阵 | bsz 1/2/4/8 的 forward、prefill、decode、cache 和 generate 均正确，并记录吞吐与内存。 | no machine-verifiable proof registered |
| ⬜ MISSING | `hf.mps.ragged_dynamic_batch` | HF 动态/不等长批 | 不同 prompt 长度、不同 seen_tokens 的请求可 select/reorder/compact；加入和离开后各会话与独立执行一致。 | no machine-verifiable proof registered |
| ⬜ MISSING | `hf.mps.generate_modes` | HF GenerationMixin 常用模式 | greedy、temperature/top-k/top-p sampling、stopping criteria、streamer、return_dict_in_generate 均正常。 | no machine-verifiable proof registered |
| ⬜ MISSING | `hf.mps.beam_cache_reorder` | Beam 与 cache reorder | beam search/beam sample 调用 cache reorder 后 state、batch 索引和输出正确，无跨 beam 污染。 | no machine-verifiable proof registered |
| ⬜ MISSING | `hf.mps.tokenizer_chat` | Tokenizer 与聊天模板 | 官方 tokenizer、特殊 token、Unicode/中英文、chat template、encode/decode roundtrip 与 model.generate 集成通过。 | no machine-verifiable proof registered |
| ⬜ MISSING | `hf.mps.save_reload` | HF save_pretrained roundtrip | save_pretrained/from_pretrained 后固定语料 logits/state/greedy、generation config 和自定义代码加载一致。 | no machine-verifiable proof registered |
| ⬜ MISSING | `hf.mps.long_context` | HF 长上下文稳定性 | 1k/4k/8k token prefill+至少 256 token decode 无 NaN/OOM/状态漂移，内存符合 RWKV 状态常数上下文增长预期。 | no machine-verifiable proof registered |
| ⬜ MISSING | `hf.mps.no_silent_cpu_fallback` | 禁止静默 CPU 回退 | MPS 路径通过 profiler/算子审计证明关键算子未静默落 CPU；必要回退显式记录且不作为性能通过。 | no machine-verifiable proof registered |

### HF PEFT/Trainer/TRL 训练

| 状态 | Gate ID | 验收点 | 硬判据 | 当前证据 |
|---|---|---|---|---|
| ✅ PASS | `train.mps.backward_smoke` | MPS 原生反向传播 | 真实模型 loss 有限、梯度非零且一步更新后参数变化；覆盖 0.1B/0.4B/1.5B。 | proof matched 3 evidence rows |
| ✅ PASS | `train.mps.peft_lora` | PEFT LoRA 模型矩阵 | 0.1B/0.4B/1.5B PEFT LoRA 注入、backward、optimizer step 和可训练参数统计通过。 | proof matched 3 evidence rows |
| ✅ PASS | `train.mps.trainer` | HF Trainer 模型矩阵 | 0.1B/0.4B/1.5B 的 Trainer+LoRA 有限 loss、参数更新和至少 20 步 1.5B 记录。 | proof matched 10 evidence rows |
| ✅ PASS | `train.mps.trl_sft` | TRL SFTTrainer 模型矩阵 | 0.1B/0.4B/1.5B SFTTrainer+LoRA 训练成功，loss 有限且参数更新。 | proof matched 10 evidence rows |
| ✅ PASS | `train.mps.trl_dpo` | TRL DPOTrainer 模型矩阵 | 0.1B/0.4B/1.5B DPOTrainer+LoRA 训练成功，loss 有限且参数更新。 | proof matched 9 evidence rows |
| ✅ PASS | `train.mps.trl_grpo` | TRL GRPOTrainer 模型矩阵 | 0.1B/0.4B/1.5B GRPOTrainer+LoRA 训练成功、奖励/生成链路有效且参数更新。 | proof matched 9 evidence rows |
| ⬜ MISSING | `train.mps.peft_methods` | 常见 PEFT 方法兼容 | LoRA、AdaLoRA、IA3、prefix/prompt tuning 中适用于 CausalLM 的常见配置能初始化、训练、保存和加载；不支持项有明确技术说明。 | no machine-verifiable proof registered |
| ✅ PASS | `train.mps.gradient_accumulation` | 梯度累积 | gradient_accumulation_steps 1/2/4 的有效 batch、loss 缩放和更新次数正确。 | 2/2 child proofs pass |
| ⬜ MISSING | `train.mps.checkpoint_resume` | 训练断点续训 | Trainer/TRL 保存 optimizer、scheduler、RNG、adapter 与 global step；resume 后与连续训练在声明容差内一致。 | no machine-verifiable proof registered |
| ⬜ MISSING | `train.mps.adapter_roundtrip` | Adapter 保存合并与重载 | PEFT adapter save/load、merge/unmerge 后固定评测 logits/greedy 与训练前后预期一致。 | no machine-verifiable proof registered |
| ⬜ MISSING | `train.mps.mixed_precision` | 混合精度训练 | MPS fp16/autocast 可训练，无不支持算子静默错误；loss/grad 有限并与 fp32 基线质量对齐。 | no machine-verifiable proof registered |
| ⬜ MISSING | `train.mps.long_run_1000` | 1000 步稳定训练 | 至少 1.5B LoRA 的 Trainer、SFT、DPO、GRPO 各完成生产形状长跑（至少 1000 optimizer steps），无 NaN、OOM 或持续内存增长。 | no machine-verifiable proof registered |
| ⬜ MISSING | `train.mps.context_batch_matrix` | 训练序列与批矩阵 | seq 128/512/2k 与 micro-bsz 1/2/4 覆盖 Trainer/PEFT/TRL，记录 steps/s、峰值内存和梯度稳定性。 | no machine-verifiable proof registered |
| ⬜ MISSING | `train.mps.deterministic_resume` | 训练可复现性 | 固定 seed 的两次短训与中断恢复在声明容差内复现 loss、梯度、权重与样本顺序。 | no machine-verifiable proof registered |

### MLX/Metal 数值与模型正确性

| 状态 | Gate ID | 验收点 | 硬判据 | 当前证据 |
|---|---|---|---|---|
| ✅ PASS | `mlx.converted_models_load` | MLX 转换模型加载生成 | 0.1B/0.4B/1.5B 转换产物均能真实加载、prefill 和 decode。 | proof matched 3 evidence rows |
| ⬜ MISSING | `mlx.hf_parity_matrix` | MLX 与 HF 全模型对齐 | 0.1B/0.4B/1.5B 在 fp32/fp16、短长 prompt、bsz 1/2/4 上逐层/最终 logits、state 和 greedy 与 HF 参考对齐。 | no machine-verifiable proof registered |
| ⬜ MISSING | `mlx.recurrent_state_parity` | MLX recurrent state 对齐 | token、full、chunk 多种切分的五类/全部层状态和 seen_tokens 达到容差，无序列边界污染。 | no machine-verifiable proof registered |
| ⬜ MISSING | `mlx.metal_wkv_reference` | Metal WKV 与参考实现 | 支持 shape/dtype/bsz/context 矩阵下 Metal WKV 与独立 MLX/PyTorch reference 对齐，unsupported shape 安全回退。 | no machine-verifiable proof registered |
| ✅ PASS | `mlx.dplr_model_matrix` | DPLR/WY 真实模型矩阵 | M5 fp16 0.1B/0.4B/1.5B、真实 prompt 的 tiled Metal DPLR prefill 通过 logits/state 容差和 greedy 完全一致。 | proof matched 9 evidence rows |
| ⬜ MISSING | `mlx.dplr_partial_chunks` | DPLR 尾块与边界 | 长度 1/7/63/64/65/127/128/129/511/512/513 的 identity padding、trim 和最终 state/output 与 recurrent 对齐。 | no machine-verifiable proof registered |
| ⬜ MISSING | `mlx.dplr_chunk_matrix` | DPLR chunk/window 矩阵 | chunk 16/32/64/128、window 256/512/1024 与跨窗口 prefix combine 均通过，策略不依赖单一硬编码形状。 | no machine-verifiable proof registered |
| ⬜ MISSING | `mlx.long_context_matrix` | MLX 1k/4k/8k 上下文矩阵 | 0.1B/0.4B/1.5B 的 fp16 及 W4 在 1k/4k/8k tokens 上 full/chunk/DPLR continuation 一致，至少 decode 256 tokens。 | no machine-verifiable proof registered |
| ✅ PASS | `mlx.long_context_w4_1p5b` | 1.5B W4 8k 实测 | 1.5B W4/Metal 完成 8192-token prefill、1024-token decode，chunked/full max_abs=0 且 token/state 计数正确。 | proof matched 1 evidence rows |
| ✅ PASS | `mlx.session_continuation_models` | MLX 会话续写矩阵 | 0.1B/0.4B/1.5B 分多轮 decode 与 one-shot token/text 完全一致，seen_tokens 正确。 | proof matched 3 evidence rows |
| ⬜ MISSING | `mlx.cache_select_reorder` | MLX cache select/reorder/clone | 状态 cache select、reorder、clone、concat、split 后每个序列 continuation 与独立运行一致，源 cache 不被修改。 | no machine-verifiable proof registered |
| ✅ PASS | `mlx.compiled_decode_fp16_0p4` | 0.4B fp16 guarded compiled decode | 0.4B fp16 batch1 64-token eager/compiled token、logits、全部 state 严格一致，auto 实际选择 compiled。 | proof matched 1 evidence rows |
| ✅ PASS | `mlx.compiled_decode_w4_0p4` | 0.4B W4 guarded compiled decode | 0.4B W4 batch1 64-token eager/compiled token、logits、全部 state 严格一致，auto 实际选择 compiled。 | proof matched 1 evidence rows |
| ⬜ MISSING | `mlx.decode_policy_model_matrix` | 自动 decode 策略全模型 | 按 chip+model shape+dtype+quant+bsz 自动选择 reference/fast norm 与 eager/compiled；0.1B/0.4B/1.5B 全部通过数值门且不得比安全基线慢。 | registered proof is incomplete; run the auditor for missing paths and child details |
| ⬜ MISSING | `mlx.decode_policy_batch_matrix` | Compiled decode 批矩阵 | bsz 1/2/4/8 分别编译、验证和缓存图；动态批切换不复用错误 shape 图，失败自动回退。 | registered proof is incomplete; run the auditor for missing paths and child details |
| ⬜ MISSING | `mlx.fast_norm_matrix` | Fast LayerNorm 全矩阵 | 0.1B/0.4B/1.5B、fp16/W8/W4、bsz 1/2/4/8 与 reference norm 的 greedy、logits、state 和质量门通过。 | no machine-verifiable proof registered |
| ⬜ MISSING | `mlx.fallback_contract` | 不支持形状安全回退 | Metal/DPLR/compiled/quant 的不支持 dtype、shape、芯片和 batch 明确回退；输出正确、原因可观测、无崩溃。 | no machine-verifiable proof registered |
| ⬜ MISSING | `mlx.determinism` | MLX 确定性 | 固定 seed/greedy 下跨进程、冷/热图、chunk 切分和会话调度产生相同 token；数值差在门限内。 | no machine-verifiable proof registered |
| ✅ PASS | `mlx.speculative_m5_b1_exact` | M5 batch1 初步投机解码精确性 | M5 16GB、1.5B W4 target + 0.1B W4 draft、chars512/decode64、proposal32、repeat3；最终 token 与 target greedy 完全一致，seen_tokens 正确，并记录接受率、verify/replay/fallback。 | 3/3 child proofs pass |
| ⬜ MISSING | `mlx.speculative_rejection_fallback` | 投机解码拒绝重放与低接受率回退 | 真实 target/draft 在包含接受、拒绝、部分 block、EOS 和低接受率的语料矩阵上与 target greedy 完全一致；拒绝后 state replay 正确，低于阈值时安全回退且无状态污染。 | no machine-verifiable proof registered |

### MLX 生产服务能力

| 状态 | Gate ID | 验收点 | 硬判据 | 当前证据 |
|---|---|---|---|---|
| ✅ PASS | `serve.interleaved_sessions_smoke` | 多会话交错执行实测 | 0.1B/0.4B/1.5B 至少 2 个会话交错执行，one-shot token/text/seen_tokens 全匹配。 | proof matched 91 evidence rows |
| ⬜ MISSING | `serve.true_dynamic_batch` | 真实动态批处理 | 非顺序循环伪批：bsz 1/2/4/8 使用实际 batched kernel，吞吐随批量合理增长且每会话严格对齐。 | no machine-verifiable proof registered |
| ✅ PASS | `serve.ragged_mixed_prompts` | 不等长 prompt 批处理 | 混合 prompt/已生成长度在同轮 compact/expand，mask/state 索引正确，无跨请求污染。 | 2/2 child proofs pass |
| ✅ PASS | `serve.arrival_departure` | 请求动态加入退出 | 持续 decode 中请求可加入、完成、取消并释放槽位；剩余请求输出和延迟不受错误影响。 | 2/2 child proofs pass |
| ✅ PASS | `serve.state_prefix_cache` | State/prefix cache | 规范化 token prefix 生成可复用 RWKV state；命中后 continuation 与完整 prefill 一致，支持 chunk boundary。 | 2/2 child proofs pass |
| ✅ PASS | `serve.cache_key_isolation` | Cache key 与模型隔离 | cache key 包含 model/revision/tokenizer/dtype/quant/backend/prefix；不同租户或配置不得误命中。 | 2/2 child proofs pass |
| ✅ PASS | `serve.cache_eviction` | Cache 容量与淘汰 | LRU/TTL/显存水位淘汰可配置；引用中的 state 不被提前释放，淘汰后内存回收。 | 2/2 child proofs pass |
| ✅ PASS | `serve.cache_hit_rate` | 合理 cache 命中率 | 公开可复现的 Zipf/共享前缀负载下报告请求/字节加权命中率；达到预设 workload SLO，不能用合成全相同 prompt 夸大。 | 2/2 child proofs pass |
| ✅ PASS | `serve.cancellation_backpressure` | 取消与背压 | 队列上限、超时、取消、过载拒绝和清理路径正确；不会无限排队或遗留 state/graph。 | 2/2 child proofs pass |
| ⬜ MISSING | `serve.streaming_api` | 流式生成接口 | 逐 token 流式输出、停止词/EOS、客户端取消和错误传播正确，首 token/结束状态可观测。 | no machine-verifiable proof registered |
| ✅ PASS | `serve.concurrency_32` | 高并发稳定性 | 在能容纳的量化模型上 32 并发、混合长度、至少 10k 请求；无错 token、死锁、崩溃或无界内存。 | proof matched 1 evidence rows |
| ⬜ MISSING | `serve.latency_percentiles` | 服务延迟分位数 | 统一负载报告 TTFT/TPOT/E2E 的 p50/p95/p99、吞吐、公平性、队列时间及冷/热状态。 | no machine-verifiable proof registered |

### 性能与 Qwen3.5 验收

| 状态 | Gate ID | 验收点 | 硬判据 | 当前证据 |
|---|---|---|---|---|
| ✅ PASS | `perf.dplr_prefill_m5_models` | M5 DPLR prefill 性能台阶 | M5 prompt111 fp16 tiled DPLR：0.1B median≥4.5k、0.4B≥2.0k、1.5B≥0.9k tok/s，且每项三次以上正确性通过。 | 3/3 child proofs pass |
| ✅ PASS | `perf.qwen_0p4_short_speed` | 0.4B 对 Qwen0.8B 短请求速度 | M5 chars128/512、decode32、repeat3：RWKV decode/prefill≥Qwen，TTFT≤1.1×Qwen；独立于内存门。 | proof matched 2 evidence rows |
| ✅ PASS | `perf.qwen_0p4_memory` | 0.4B 对 Qwen0.8B 峰值内存 | 相同实际运行范围内捕获双方 process/device true peak；RWKV≤Qwen，不能用包大小替代 Qwen 峰值。 | 2/2 child proofs pass |
| ✅ PASS | `perf.m5_close_qwen_0p4_w4_b1` | M5 0.4B W4 对 Qwen0.8B W4 限定门 | 同一 M5 16GB、batch1、chars512/decode64、warmup1+repeat3：RWKV decode/prefill≥Qwen、TTFT≤1.1×、真实峰值内存≤Qwen；仅关闭该形状，不替代完整矩阵。 | 2/2 child proofs pass |
| ⬜ MISSING | `perf.qwen_0p4_full_matrix` | 0.4B 对 Qwen0.8B 完整矩阵 | prompt 128/512/1k/4k/8k tokens、decode 32/128/512、bsz 1/2/4/8 的 prefill/decode/TTFT/峰值内存全部不差于 Qwen。 | no machine-verifiable proof registered |
| ⬜ MISSING | `perf.qwen_1p5_vs_2b` | 1.5B 对 Qwen3.5 2B | 同 prompt、tokenizer 统计透明、相同精度级别和运行条件下，全部长度/bsz 的速度、TTFT、峰值内存与质量通过。 | no machine-verifiable proof registered |
| ✅ PASS | `perf.m5_close_qwen_1p5_spec_w4_b1` | M5 1.5B W4 投机对 Qwen2B W4 限定门 | 同一 M5 16GB、batch1、chars512/decode64、warmup1+repeat3：1.5B W4+0.1B draft 的decode/prefill≥Qwen2B、TTFT≤1.1×、真实峰值内存≤Qwen，并通过 target-greedy oracle；仅关闭该形状。 | 3/3 child proofs pass |
| ⬜ MISSING | `perf.qwen_2p9_vs_4b` | 2.9B 对 Qwen3.5 4B | 同 prompt、相同运行条件下，全部长度/bsz 的速度、TTFT、峰值内存与质量通过。 | no machine-verifiable proof registered |
| ⬜ MISSING | `perf.qwen_7p2_vs_9b` | 7.2B 对 Qwen3.5 9B | 同 prompt、相同运行条件下，全部长度/bsz 的速度、TTFT、峰值内存与质量通过；不适配内存不足机器时明确能力边界。 | no machine-verifiable proof registered |
| ⬜ MISSING | `perf.batch_scaling` | 吞吐批量扩展 | 0.1B/0.4B/1.5B、fp16/W8/W4 在 bsz 1/2/4/8 的 prefill/decode 总吞吐、每请求延迟和内存满足生产 SLO。 | no machine-verifiable proof registered |
| ⬜ MISSING | `perf.cold_ttft` | 冷启动 TTFT | 包含模型加载、首次 Metal 编译和 tokenizer 的冷 TTFT 单独报告并满足对标门；不把预热结果冒充冷启动。 | no machine-verifiable proof registered |
| ⬜ MISSING | `perf.warm_ttft` | 热请求 TTFT | 已加载/已编译、cache miss 与 hit 分开报告，p50/p95 不差于对应 Qwen 基线。 | no machine-verifiable proof registered |
| ⬜ MISSING | `perf.peak_memory` | 真实峰值内存 | 统一进程隔离采集 MLX active/peak/cache、MPS driver、RSS/压缩/交换；报告权重、state、临时 buffer 和编译 cache。 | no machine-verifiable proof registered |
| ⬜ MISSING | `perf.state_memory_scaling` | State cache 内存缩放 | 实测 state cache 随 layers×heads×head_dim²×active_sequences 线性，和 prompt 长度无关；误差在 10% 内。 | no machine-verifiable proof registered |
| ⬜ MISSING | `perf.compile_amortization` | 编译收益与摊销 | 每个策略记录 compile/warmup 时间、图缓存命中和 break-even token/request 数；短请求策略不得因编译变慢。 | no machine-verifiable proof registered |
| ⬜ MISSING | `perf.variance` | 可重复性能 | 隔离负载、固定功耗/温度说明、至少 5 次重复；报告 min/median/p95，关键门在独立复跑仍通过且变异系数≤10%。 | no machine-verifiable proof registered |
| ⬜ MISSING | `perf.thermal_sustained` | 持续热稳态性能 | 至少 30 分钟连续负载记录温度/频率/功耗和吞吐；热稳态 p95 不低于冷峰值的既定 SLO。 | no machine-verifiable proof registered |
| ⬜ MISSING | `perf.model_load_startup` | 模型加载与首次请求 | fp16/W8/W4 各模型的磁盘读取、转换/反序列化、首次请求和缓存后加载时间达到生产预算。 | no machine-verifiable proof registered |

### W8/W4 量化

| 状态 | Gate ID | 验收点 | 硬判据 | 当前证据 |
|---|---|---|---|---|
| ✅ PASS | `quant.w8_memory_1p5b` | 1.5B W8 内存下降 | 同 M5 4k 形状下 1.5B W8 peak≤2.2GB，fp16 peak≥3.0GB，证明实际运行峰值下降。 | 2/2 child proofs pass |
| ✅ PASS | `quant.groupwise_w8_1p5_m5_close` | M5 1.5B groupwise W8 decode 保留门 | M5 batch1 chars512/decode64、warmup1+repeat2 的已提交保留行：groupwise W8 guarded compiled min decode≥46 tok/s、peak≤2.25GB，64-token token/logits/state 门通过。repeat2 是显式限定，仍不足以关闭全量 W8 生产性能门。 | 3/3 child proofs pass |
| ✅ PASS | `quant.w4_memory_0p4b` | 0.4B W4 内存下降 | 同 M5 4k 形状下优化 W4 peak≤0.60GB，fp16 peak≥0.90GB。 | 2/2 child proofs pass |
| ✅ PASS | `quant.groupwise_w4_0p4_m5_close` | M5 0.4B groupwise W4 compiled 限定门 | M5 batch1 chars512/decode64、warmup1+repeat3 的 native groupwise W4 路径实际 dispatch packed quantized matmul；compiled validation token 一致、logits/state≤0.125，peak≤0.51GB。 | 2/2 child proofs pass |
| ✅ PASS | `quant.w4_memory_1p5b` | 1.5B W4 内存下降 | 同 M5 4k 形状下优化 W4 peak≤1.20GB，fp16 peak≥3.0GB。 | 2/2 child proofs pass |
| ✅ PASS | `quant.groupwise_w4_1p5_m5_close` | M5 1.5B groupwise W4 decode 限定门 | M5 batch1 chars512/decode64、warmup1+repeat3：groupwise W4 guarded compiled min decode≥55 tok/s、peak≤1.78GB，64-token token/logits/state 门通过；不替代全 bsz/长上下文/质量门。 | 3/3 child proofs pass |
| ✅ PASS | `quant.quality_fp16_proxy_m5` | M5 同 checkpoint FP16 量化质量 proxy | M5 上以同一 RWKV checkpoint 的 fp16 为 teacher，覆盖 0.4B/1.5B、276 scored tokens、4 个 greedy prompts：W8 NLL delta≤0.02、perplexity ratio≤1.02、teacher top1≥0.95；mixed W8/W4 q4_k_m proxy 分别≤0.08/1.09/≥0.80，且权重存储下降。该门不声称 llama.cpp/GGUF Q*_K_M 等价，也不关闭 greedy/state parity。 | 7/7 child proofs pass |
| ⬜ MISSING | `quant.memory_all_models` | W8/W4 全模型内存矩阵 | 0.1B/0.4B/1.5B/可运行更大模型的真实 process/device peak 分别低于 W16，W8/W4 降幅接近权重量化理论且无隐藏全量反量化副本。 | no machine-verifiable proof registered |
| ⬜ MISSING | `quant.w8_decode_faster` | W8 decode 不慢于 W16 | 所有支持 Apple 芯片、模型和 bsz 上，稳态 W8 decode tok/s≥W16；正确性/质量门同时通过。 | no machine-verifiable proof registered |
| ⬜ MISSING | `quant.w4_decode_faster` | W4 decode 不慢于 W16 | 所有支持 Apple 芯片、模型和 bsz 上，稳态 W4 decode tok/s≥W16；正确性/质量门同时通过。 | no machine-verifiable proof registered |
| ⬜ MISSING | `quant.prefill_faster` | 量化 prefill 不慢于 W16 | 1k/4k/8k 与 bsz 1/2/4/8 的 W8/W4 prefill tok/s≥W16，包含 dequant/pack 开销。 | no machine-verifiable proof registered |
| ⬜ MISSING | `quant.packed_kernels` | 真实 packed W8/W4 kernel | 关键 Linear/RKV/FFN/lm_head 直接消费 packed 权重；profiler/计数证明走 Metal kernel，无全权重 materialize。 | no machine-verifiable proof registered |
| ⬜ MISSING | `quant.greedy_state_parity` | 量化 token/state 门 | 固定短长语料和随机输入，W8/W4 相对 W16 的 greedy token、state 稳定性和 logits 误差达到模型级门。 | no machine-verifiable proof registered |
| ⬜ MISSING | `quant.quality_qkm` | Q*_K_M 水准质量 | 标准 perplexity/任务集/长上下文上，W8/W4 质量尽可能达到对应 llama.cpp Q*_K_M 水准；报告均值、最坏项和统计置信区间。 | no machine-verifiable proof registered |
| ⬜ MISSING | `quant.batch_dynamic` | 量化动态批正确性 | W8/W4 bsz 1/2/4/8、ragged session、select/reorder、stable argmax 均与独立执行一致。 | no machine-verifiable proof registered |
| ⬜ MISSING | `quant.long_context` | 量化长上下文 | W8/W4 1k/4k/8k prefill+512 decode 无溢出/漂移，内存和速度门继续成立。 | no machine-verifiable proof registered |
| ⬜ MISSING | `quant.serialization` | 量化产物序列化 | packed 权重、scale/zero/group metadata 可保存、哈希验证、跨进程重载，不在启动时重新全量量化。 | no machine-verifiable proof registered |
| ⬜ MISSING | `quant.auto_policy` | 量化后端自动策略 | 按芯片/shape/bsz 选择 affine/Metal/grouped kernel；只有同时通过速度与质量门才默认启用，否则显式回退。 | no machine-verifiable proof registered |
| ⬜ MISSING | `quant.error_handling` | 量化能力与错误契约 | 不支持的 group size、shape、dtype 或旧芯片给出明确错误/安全回退；不产生静默错误输出。 | no machine-verifiable proof registered |

### CoreML/ANE 部署

| 状态 | Gate ID | 验收点 | 硬判据 | 当前证据 |
|---|---|---|---|---|
| ⬜ MISSING | `coreml.tools_export` | CoreML 工具链真实导出 | 非 dry-run 使用受支持 coremltools 导出并编译 .mlpackage/.mlmodelc；manifest 含版本、目标、compute units 和 hashes。 | no machine-verifiable proof registered |
| ⬜ MISSING | `coreml.model_matrix` | CoreML 模型尺寸矩阵 | 0.1B/0.4B/1.5B 均能导出、编译、加载和执行；超出设备能力的型号有明确限制而非假通过。 | no machine-verifiable proof registered |
| ⬜ MISSING | `coreml.stateful_multifunction` | Stateful multifunction 图 | prefill/decode 函数共享并更新 CoreML state；跨多轮调用 seen_tokens/state 语义正确，无 host 全量 state copy。 | no machine-verifiable proof registered |
| ⬜ MISSING | `coreml.chunked_prefill` | CoreML chunked prefill | chunk 1/16/64/128 与 full prefill 的 logits/state/greedy 对齐，尾块和跨 chunk state 正确。 | no machine-verifiable proof registered |
| ⬜ MISSING | `coreml.hf_parity` | CoreML 与 HF 对齐 | 真实 checkpoint 的 fp16 CoreML 在短长语料上相对 HF reference 通过 logits/state/greedy 门。 | no machine-verifiable proof registered |
| ⬜ MISSING | `coreml.compute_unit_matrix` | Compute unit 矩阵 | CPUOnly、CPUAndGPU、CPUAndNeuralEngine/All 的可用配置均运行并报告实际设备、正确性、速度和峰值内存。 | no machine-verifiable proof registered |
| ⬜ MISSING | `coreml.ane_placement` | ANE 实际落核证据 | 使用 Instruments/Core ML compute plan 或等价证据证明关键层实际运行在 ANE；仅设置 compute_units 不算通过。 | no machine-verifiable proof registered |
| ⬜ MISSING | `coreml.macos_runtime` | macOS 生产运行时 | 独立 macOS app/CLI 加载 compiled model，完成多轮 prefill/decode、错误处理、资源释放和性能采集。 | no machine-verifiable proof registered |
| ⬜ MISSING | `coreml.ios_runtime` | iOS/iPadOS 运行时 | 真机 iPhone/iPad app 完成加载、状态续写、后台/前台切换、内存警告处理和性能/能耗采集。 | no machine-verifiable proof registered |
| ⬜ MISSING | `coreml.swift_api` | Swift 示例与 API | 提供可构建 Swift 包/示例，覆盖 tokenizer、state、streaming、取消、错误和量化模型选择。 | no machine-verifiable proof registered |
| ⬜ MISSING | `coreml.int8_size` | CoreML INT8 体积与内存 | INT8 包大小和运行峰值相对 fp16 明显下降，无隐藏 fp16 权重副本。 | no machine-verifiable proof registered |
| ⬜ MISSING | `coreml.int8_speed` | CoreML INT8 速度 | 所有目标设备上 INT8 prefill/decode 均不慢于 fp16，且落在预期计算单元。 | no machine-verifiable proof registered |
| ⬜ MISSING | `coreml.int8_quality` | CoreML INT8 精度 | INT8 logits/greedy/状态与任务质量达到量化门。 | no machine-verifiable proof registered |
| ⬜ MISSING | `coreml.int4_size` | CoreML INT4/LUT4 体积与内存 | INT4/LUT4 包大小和运行峰值相对 fp16/INT8 按预期下降。 | no machine-verifiable proof registered |
| ⬜ MISSING | `coreml.int4_speed` | CoreML INT4/LUT4 速度 | 目标设备上 INT4/LUT4 prefill/decode 均不慢于 fp16，包含解包开销。 | no machine-verifiable proof registered |
| ⬜ MISSING | `coreml.int4_quality` | CoreML INT4/LUT4 精度 | INT4/LUT4 greedy、logits/state 和标准质量达到既定门，不接受当前已知 token 漂移。 | no machine-verifiable proof registered |
| ⬜ MISSING | `coreml.long_context_memory` | CoreML 长上下文与内存 | 1k/4k/8k prefill、512 decode、state 生命周期与峰值内存通过，反复请求无泄漏。 | no machine-verifiable proof registered |

### 稳定性、运维与 CI

| 状态 | Gate ID | 验收点 | 硬判据 | 当前证据 |
|---|---|---|---|---|
| ⬜ MISSING | `reliability.soak_24h` | 24 小时 soak | MLX fp16/W8/W4 生产混合负载连续 24 小时；零崩溃/死锁/错 token，错误率满足 SLO。 | no machine-verifiable proof registered |
| ✅ PASS | `reliability.requests_10k` | 至少 10k 请求回归 | 固定可复现混合长度/并发 workload 完成≥10k 请求，结果、延迟分位和资源统计完整。 | proof matched 1 evidence rows |
| ⬜ MISSING | `reliability.memory_leak` | 内存泄漏门 | 稳态窗口内 active/RSS/driver/cache 在请求释放后回落；24h 趋势增长≤2% 或有可解释有界 cache。 | no machine-verifiable proof registered |
| ⬜ MISSING | `reliability.thermal_energy` | 温度与能耗 | 记录 wall power/energy per token、热降频和电池场景；不同 backend/quant 的性能功耗比可复现。 | no machine-verifiable proof registered |
| ⬜ MISSING | `reliability.oom_recovery` | OOM 与压力恢复 | 内存预算、预检查、cache eviction、清晰 OOM 错误和后续请求恢复通过；不导致系统卡死/重启。 | no machine-verifiable proof registered |
| ⬜ MISSING | `reliability.crash_recovery` | 进程异常恢复 | worker 异常退出、编译失败、损坏 cache 后能重启并重建状态；未完成请求明确失败，不返回错误 continuation。 | no machine-verifiable proof registered |
| ⬜ MISSING | `reliability.input_validation` | 输入与边界校验 | 空 prompt、超长 prompt、非法 token id、坏 config/权重、NaN 参数和不支持选项给出确定错误。 | no machine-verifiable proof registered |
| ⬜ MISSING | `reliability.model_integrity` | 模型完整性与版本锁定 | 模型 revision、config/tokenizer/weight hashes、转换器版本记录并在加载时校验，避免混用产物。 | no machine-verifiable proof registered |
| ⬜ MISSING | `reliability.thread_process_safety` | 线程/进程安全 | 并发调用、fork/spawn 限制、Metal command queue 与 cache 锁行为有测试，无 data race 或状态串线。 | no machine-verifiable proof registered |
| ⬜ MISSING | `reliability.ci_unit` | 跨平台 CPU 单测 CI | Linux/macOS CPU CI 覆盖数学、cache、转换、manifest、错误路径，主分支持续通过。 | no machine-verifiable proof registered |
| ⬜ MISSING | `reliability.ci_apple` | Apple 真机 CI | 至少一台 Apple Silicon runner 对 MPS+MLX 做真实模型 smoke、数值门和短性能回归，不以 mock 替代。 | no machine-verifiable proof registered |
| ⬜ MISSING | `reliability.performance_regression` | 性能回归 CI | 按芯片保存基线；prefill/decode/TTFT/memory 超过允许回退即阻止发布，结果保留可追溯。 | no machine-verifiable proof registered |
| ⬜ MISSING | `reliability.version_matrix` | 依赖与系统版本矩阵 | 项目声明的最小/最新 macOS、Python、PyTorch、Transformers、PEFT、TRL、MLX、coremltools 组合均通过或明确锁定。 | no machine-verifiable proof registered |

### Apple 芯片与内存档覆盖

| 状态 | Gate ID | 验收点 | 硬判据 | 当前证据 |
|---|---|---|---|---|
| ⬜ MISSING | `hardware.m1_family` | Apple M1 系列 | 至少 M1 base 与一款高 GPU/内存档实测全部适用正确性门、性能、量化、稳定性；记录具体 SKU。 | no machine-verifiable proof registered |
| ⬜ MISSING | `hardware.m2_family` | Apple M2 系列 | 至少 M2 base 与一款高 GPU/内存档实测全部适用门；专门策略不得照搬 M5。 | no machine-verifiable proof registered |
| ⬜ MISSING | `hardware.m3_family` | Apple M3 系列 | 至少 M3 base 与一款高 GPU/内存档实测全部适用门。 | no machine-verifiable proof registered |
| ⬜ MISSING | `hardware.m4_family` | Apple M4 系列 | 至少 M4 base 与一款高 GPU/内存档实测全部适用门。 | no machine-verifiable proof registered |
| ✅ PASS | `hardware.m5_base` | Apple M5 base 证据 | M5 base 机器环境、MPS 与 MLX 真机结果可追溯。 | 2/2 child proofs pass |
| ⬜ MISSING | `hardware.m5_high_tier` | Apple M5 高配档 | M5 系列可商购高 GPU/内存档至少一款完成适用全门，记录具体 SKU/核心数/内存带宽。 | no machine-verifiable proof registered |
| ⬜ MISSING | `hardware.memory_8gb` | 8GB 统一内存档 | 8GB 真机验证能运行的模型/量化、并发、OOM 防护和系统响应；明确不能运行的边界。 | no machine-verifiable proof registered |
| ✅ PASS | `hardware.memory_16gb` | 16GB 统一内存档 | 16GB M5 真机环境与峰值证据存在。 | proof matched 4 evidence rows |
| ⬜ MISSING | `hardware.memory_32gb_plus` | 32GB+ 统一内存档 | 32GB 或更高真机覆盖较大模型、并发、长上下文和 W8/W4 性能门。 | no machine-verifiable proof registered |
| ⬜ MISSING | `hardware.mobile_ane` | iPhone/iPad ANE 真机 | 至少一代仍受支持 iPhone 与 iPad 真机完成 CoreML/ANE 正确性、内存、速度、能耗和热稳态门。 | no machine-verifiable proof registered |

## 当前实施顺序

1. **先关闭正确性广义门**：HF/MPS 与 MLX 的 dtype、bsz、chunk、长上下文、fallback、determinism 全矩阵。
2. **量化生产闭环**：保留已通过的 M5 groupwise W8/W4 原子门，补 W16 对照、bsz 1/2/4/8、1k/4k/8k、Q*_K_M 质量和序列化。
3. **Qwen3.5 全矩阵**：从已通过的两个 batch1/512/64 配对扩展到所有长度、bsz、0.8B/2B/4B/9B，并加入冷/热与热稳态。
4. **投机解码负路径**：补拒绝重放、部分 block、EOS、低接受率 fallback、不同 draft 与长跑，不能用 100% 接受单行替代。
5. **CoreML/ANE**：真实 export/runtime、HF/state/chunk parity、INT8/INT4、Instruments/compute-plan 落核证据。
6. **训练、可靠性与设备矩阵**：1000 步 Trainer/TRL、24h/10k、泄漏/OOM/热稳态，以及 M1-M4、不同内存档和 iPhone/iPad。

## 证据文件

- 机器可读清单：`bench/apple_production_gates.json`
- 严格审计器：`bench/check_apple_production_acceptance.py`
- 一键入口：`scripts/run_apple_production_acceptance.sh`
- 审计输出：`bench/results_apple_production_acceptance.jsonl`（默认 append-only）
- 最新限定 M5 说明：`docs/hardware/APPLE_PRODUCTION_CLOSE.md`
- 最新限定 M5 汇总：`bench/apple_production_close_qwen35_gate_m5_20260711.jsonl`

本文件由 `bench/render_apple_production_acceptance.py` 从 manifest 和已提交证据生成；新增或修改硬门后必须重新生成。
