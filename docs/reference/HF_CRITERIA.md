# RWKV-7 HF 适配优化准则

本仓库当前只交付 **Hugging Face / Transformers 适配**。vLLM、SGLang、独立服务引擎属于后续项目；HF 侧只负责标准 `from_pretrained` / `forward` / `generate` / Trainer / PEFT / TRL / HF state-cache helper 能力。

## 1. 最终门禁

| 领域 | 目标 | 主要证据 |
|---|---|---|
| 精度 | HF logits / greedy 结果对齐官方 RWKV-LM / `rwkv` 路径 | `tests/test_official_alignment.py`, `tests/test_reload_roundtrip.py` |
| 推理性能 | HF prefill / decode / batch generate 接近 RWKV-LM 与 Albatross，同卡同模型同 dtype 对比 | `bench/bench_speed.py`, `bench/bench_batch_sweep.py`, `bench/bench_generate_fast_path.py`, `bench/bench_ttft_tpot.py` |
| 显存 | fp16 峰值接近官方；W8/W4 显存按档位下降 | `bench/bench_speed.py`, `bench/bench_quantization.py`, `tests/test_quantized_inference.py` |
| HF 训练生态 | Trainer、PEFT、TRL SFT/DPO/GRPO 跑通，LoRA 梯度非零 | `tests/test_peft_lora.py`, `tests/test_hf_training_smoke.py`, `tests/test_hf_rl_training_smoke.py` |
| state cache | RWKV recurrent cache 支持 select/reorder/drop/compact、offload/restore、chunked prefill、cache telemetry | `tests/test_batch_cache.py`, `tests/test_dynamic_batch_cache.py`, `tests/test_chunked_prefill.py`, `tests/test_native_graph_cache.py` |
| 多卡方向 | HF `device_map`/PP smoke 正确；训练 ZeRO-2/3 配置可用 | `tests/test_device_map_generate.py`, `tests/test_deepspeed_configs.py` |
| 量化 | 8bit/4bit 可加载、能生成、显存下降；速度目标为不慢于 W16 | `tests/test_quantized_inference.py`, `bench/bench_quantization.py` |
| 投机解码 | HF RWKV draft -> target greedy 等价并记录 acceptance/resync/speedup | `tests/test_speculative_decode.py`, `bench/bench_speculative_decode.py` |
| 大模型/硬件 | 0.1B 到 13.3B 转换和 smoke；V100 与新卡持续补验证 | `bench/bench_larger_model_smoke.py`, `bench/results.jsonl` |

## 2. 当前已验证进展

- **精度**：V100 fp16 official alignment、reload roundtrip、greedy window 已通过，`fuse_norm=false` 路径稳定。
- **fp16 decode**：V100 fused native_graph 的 0.1B bsz1 已达到 `637.9 tok/s`；0.1B/0.4B/1.5B、bsz1/2/4/8 共 12 个同卡 Albatross 对比行全部通过 P1，bsz8 全部通过 P3，0.4B/1.5B bsz8 已超过 Albatross。
- **HF forward/generate 快路径**：`RWKV7_FAST_FORWARD=1` 默认把 eval/no-grad cached one-token `forward()` / `generate()` 路由到 fast-token backend。
- **state cache**：已有 `RWKV7StateCache`、dynamic batch select/reorder/drop/compact、chunked prefill、offload/restore、cache metrics、native-graph runner LRU / hit-rate telemetry。
- **训练兼容**：PEFT LoRA、HF Trainer、TRL SFT、TRL DPO、TRL GRPO smoke 已有脚本覆盖；Trainer/TRL smoke 现支持 batch size / gradient accumulation 参数，并校验 LoRA/trainable 参数确实更新；V100 fp32 smoke 已覆盖 Trainer/SFT batch=2 grad_accum=2 以及 DPO/GRPO batch=2；训练 smoke 可写入 `training_smoke` JSONL telemetry，analyzer/check gate 会汇总并校验 trainable delta。
- **多卡方向**：2 x V100 手动 `device_map` PP generate smoke 已通过；ZeRO-2/3 runtime 与 resume 已覆盖到 2.9B 的 V100 native/HF 路径。
- **大模型**：0.4B / 1.5B / 2.9B / 7.2B / 13.3B 已完成 HF 转换和 V100 load/forward/generate smoke rows。
- **投机解码**：0.1B draft -> 0.4B target V100 smoke 已保持 target greedy 一致，并用 cached-prefix resync 达到短样例约 `2.1x` target-greedy speedup。
- **experimental native PyTorch**：已保留为非默认长期底座，覆盖 FLA-free bsz=1、batched forward / incremental cache 对齐测试，并把 cached decode 接到 optional native_jit；V100 fp32 smoke 里 native_model cached decode 从约 `61.7 tok/s` 到 `115.5 tok/s`；只作为 upstream / AMD / CPU fallback 方向，不替代当前 wrapper。
- **量化可用性**：bitsandbytes 8bit/4bit 可加载生成，默认 `memory` policy 跳过 `lm_head` 与 RWKV 小 rank LoRA projection，V100 显存从 fp16 `364.4MB` 降到 8bit `283.4MB`、4bit `242.9MB`。新增 `decode_hot` policy 可继续把 attention r/k/v/o projection 保持 dense，在 V100 4bit smoke 中把 cached decode 从约 `32 tok/s` 提到约 `37 tok/s`，footprint 约 `283MB`，作为量化速度/显存折中探针。仓库原生 native MM8/MM4 另分 `memory`/`speed` 两条 lane：`memory` 追最大 footprint 下降但可能慢，`speed` 只量化 `lm_head` 并以“footprint 下降 + decode 不慢 + logits/greedy 对齐”为验收口径。

## 3. 当前最大缺口

1. **量化速度分 lane 闭合**：generic bitsandbytes W8/W4 decode 仍明显慢于 fp16 native_graph。V100 0.1B canonical `memory` policy 记录约为 fp16 `217 tok/s`、8bit `16 tok/s`、4bit `33 tok/s`；`decode_hot` hybrid policy 只能把 4bit 推到约 `37 tok/s`。native MM8/MM4 `speed` policy 已在 RTX 5090 1.5B/2.9B/7.2B 给出 footprint 下降、logits/greedy 对齐且 decode≈`0.97x-1.01x` fp16 的证据；要把大幅 footprint 下降的 `memory` lane 也做到“不慢于 W16”，仍需要 native/fused quantized projection 或专门的 int8/int4 fast-token path。
2. **Albatross 尚未全矩阵 P3**：V100 同卡、同 checkpoint 的 faster3a 基线已经闭环，当前 decode 比率为 `0.629x-1.185x`；下一步是把 bsz1/2/4 的低位行推进到统一 P2/P3，并在更多架构复现。
3. **训练吞吐未对标**：Trainer/TRL 兼容已经有 smoke，但 full/LoRA 多 batch、gradient accumulation、ZeRO runtime、RWKV-LM 训练吞吐基线仍需补。
4. **更多硬件未覆盖**：V100 和一张 Blackwell 5070 有验证记录；Pascal/Ampere/Ada/H100/AMD 仍需补。
5. **原生 Transformers 形态未完成**：当前仍依赖 FLA remote-code wrapper；最终 upstream 方向需要 pure PyTorch/reference + optional kernels。

## 4. 每轮优化流程

1. 从 `bench/analyze_results.py --json` 和 `bench/check_results.py` 找当前未达门禁。
2. 优先处理“有测试覆盖 + 能推动最终目标”的缺口：量化 fast decode、TTFT/TPOT、训练吞吐、13.3B speed/precision、更多 GPU。
3. 每次只合入一个清晰优化点：补测试、实现、跑本地静态检查、跑 V100 相关 smoke/bench。
4. 更新 `bench/results.jsonl`、`../../BENCHMARK.md` / `../archive/NEXT_STEPS.md` / 本文件中的状态，保证数字可复现。
5. PR 标题和正文只描述技术变更，不加入无关标记。
