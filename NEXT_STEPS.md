# RWKV-7 HF 适配下一步

## 已完成：阶段 1 / wrapper 可用版

- 官方 `.pth` -> HF `model.safetensors`
- HF `config.json` + `generation_config.json`
- remote-code wrapper：`AutoConfig` / `AutoModelForCausalLM`
- RWKV trie slow tokenizer：`AutoTokenizer`
- 转换脚本已从权重 shape 推断 layer count / hidden size / head_dim / value_dim / rank dims，并新增离线 shape 测试，避免后续 0.4B+ 尺寸继续沿用 0.1B hardcode。
- `generate(use_cache=True)` 跑通
- PEFT LoRA forward/loss/backward 跑通
- HF Trainer / TRL SFTTrainer 1-step LoRA smoke 跑通
- 官方 `rwkv` pip logits 对齐：top5 一致，fp16 cosine≈0.999996

模型目录：

```text
/home/data/wangyue/models/rwkv7/rwkv7-g1d-0.1b-hf
```

适配项目：

```text
/home/data/wangyue/projects/rwkv7-hf-adapter
```

## 阶段 2：把 wrapper 做到更完整

已新增并在 V100 跑通：

- `tests/test_official_alignment.py`：官方 `rwkv` vs HF logits + greedy 64 token 对齐。
- `tests/test_reload_roundtrip.py`：`save_pretrained` / reload roundtrip。
- `tests/test_fast_cache.py`：轻量 `RWKV7StateCache` 与 FLA 默认 cache 的 prefill/decode 等价测试。
- `tests/test_fast_decode_api.py`：`rwkv7_forward_token` batched one-token decode API 和 `rwkv7_forward_one` bsz=1 兼容入口与 HF recurrent forward 的等价测试。
- `tests/test_batch_cache.py`：bsz=1/2/4 repeated prompt cache/layout smoke，覆盖批量 recurrent state。
- `tests/test_dynamic_batch_cache.py`：heterogeneous prompts + cache reorder 后继续 decode，对比逐条 independent states，覆盖 dynamic batching state reorder/drop 风险。
- `tests/test_hf_training_smoke.py`：HF Trainer / TRL SFTTrainer 1-step LoRA smoke。
- `bench/bench_decode_breakdown.py`：decode 瓶颈拆分。
- `bench/bench_batch_sweep.py`：bsz=1/2/4/8 serving-style prefill/decode sweep，记录 total/per-seq throughput。
- `bench/bench_dynamic_batch.py`：模拟 active batch reorder/drop，记录 dynamic batching 相关 total decoded tok/s。
- `bench/bench_decode_micro.py`：稳定记录 HF forward decode、fast token API、`lm_head`、argmax、embedding、empty loop 等 micro timing。
- `bench/bench_decode_components.py`：细分 fast-token layer path 的 projection/recurrent/norm/FFN/top layer 耗时，用于决定下一步 fusion 目标。
- `bench/bench_projection_lora.py`：专项测 attention projection/LoRA 子模块和简单 PyTorch bmm 候选，确认下一步需要 custom fusion 而不是简单拼 bmm。
- `bench/bench_native_decode.py`：正式记录 `rwkv7_hf.native_jit` 的 native JIT / CUDA graph decode 结果，用作下一轮 fast-token integration 的性能上限参考。
- `bench/analyze_results.py`：从 `bench/results.jsonl` 输出 target/gap report，直接列出 decode/memory ratio、缺失 benchmark axis 和下一步优化焦点。
- `bench/check_results.py`：把 JSONL 结果变成可执行 gate；默认 regression gate 当前通过，`--target` gate 在 decode 达到 0.9x official 前预期失败。
- `bench/bench_speed.py` 已改成 serving-style prefill：`use_cache=True + logits_to_keep=1`，并可用 `--hf-decode-api rwkv7_forward_token` 测快 decode API。
- `bench/profile_decode.py`：单 token decode profiler。
- `scripts/convert_rwkv7_to_hf.py` 新增 `--no-fuse-norm`，作为当前 V100 推理推荐配置。
- remote config 改为唯一 `rwkv7_hf_adapter` model_type，避免 Transformers 环境中已注册的 FLA `rwkv7` 本地类绕过本仓库 wrapper。

当前 V100 结论：

- correctness：`fuse_norm=false` 下 top5/argmax/cosine/greedy64 均通过，fp16 max_abs 约 0.072。
- memory：HF 406.4 MB vs official 406.2 MB，0.1B serving path 已基本持平。
- speed：`fuse_norm=false` + `RWKV7StateCache` 下标准 remote-code HF decode 约 41.2 tok/s；FLA `rwkv7_forward_token` V100 bsz=1 约 59.2 tok/s；`RWKV7_FAST_TOKEN_BACKEND=native_jit` 的 HF fast-token bsz=1 已到 92.1 tok/s，和 official 92.1 tok/s 持平，target gate 已通过；`RWKV7_FAST_TOKEN_BACKEND=native_graph` 已把 CUDA graph 接入 HF `rwkv7_forward_token` 的固定 bsz=1/2/4/8，speed_mem bsz=1 达到 255.5 tok/s，batch sweep 达到 253.9/434.3/852.6/1539.1 aggregate tok/s；dynamic batch native_graph reorder/drop 达到 524.7 total tok/s；component bench 显示 `attn_linears_lora` 最大，约 9.87ms/token。
- quant：已新增 bitsandbytes 8bit/4bit smoke 和 benchmark。V100 0.1B 上 model footprint 从 fp16 `364.4MB` 降到 8bit `278.4MB`、4bit `235.3MB`；但 generic bnb decode 只有 `9.5` / `27.1 tok/s`，低于 fp16 `40.4 tok/s`，所以仍需自定义/融合量化 serving path 才能满足“不比 16bit 慢”的目标。
- native decode prototype：`rwkv7_hf.native_jit` 在 V100 0.1B 上已验证，logit cosine≈1.00000024、graph-vs-JIT greedy 16/16 一致；native JIT 约 103.5 tok/s，native CUDA graph 约 254.3 tok/s（2.76x official）。这个 reduced-launch 路径已接进 HF 固定 batch 和 dynamic active-batch serving API，graph runner 已改成按 active batch size 的 per-model LRU cache，并提供 `rwkv7_clear_native_graph_cache()` 释放缓存；下一步是更大模型、更多 GPU 和量化 fast path 验证。
- profiler：`fuse_norm=true` 的 FLA `LayerNormFunction` CPU 开销很大，native norm 把 norm CPU total 从约 54.8ms/6tok 降到约 6.6ms/6tok。
- breakdown：argmax 开销约等于 0，`chunk` 和 `fused_recurrent` 单 token decode 基本一样，剩余瓶颈在 HF/FLA model+state/cache+小 kernel launch 路径。

下一步继续补：

1. 支持全部已发布尺寸的配置推断和转换：0.4B / 1.5B / 2.9B / 7.2B / 13.3B。
   - 已把 converter 的 head_dim/value_dim/rank dims 改成权重 shape 推断，并用 `tests/test_convert_config.py` 覆盖非 64 head_dim、value_dim 列表和错误 shape。
   - 下一步需要拿真实 0.4B+ `.pth` 跑转换、load、alignment 和 speed smoke。
2. 增加批量转换脚本和 SHA256 manifest。
3. 补 HF behavior：
   - 已补 `resize_token_embeddings` 固定词表保护
   - 已新增 `tests/test_hf_api_contract.py` 覆盖 `prepare_inputs_for_generation`、beam cache reorder、`gradient_checkpointing_enable`
   - 继续补更完整 Transformers 原生 test suite
4. 训练路径：
   - 已跑通 PEFT LoRA backward、HF Trainer 1-step、TRL `SFTTrainer` 1-step
   - 继续扩大到真实 SFT 小数据、多 batch、gradient accumulation
   - 当前 smoke 明确 `TORCHDYNAMO_DISABLE=1`，并关闭 `use_l2warp` 避免 Trainer loss 原地缩放与 L2Wrap backward 冲突
5. 性能路径：
   - 继续 profile 单 token decode
   - `RWKV7StateCache` 已减少 generic CacheLayer 开销
   - 已新增 `rwkv7_forward_token` batched one-token fast decode entrypoint，并保留 `rwkv7_forward_one` bsz=1 兼容入口
   - 已新增 batch cache/sweep、dynamic-batch reorder/drop harness、decode microbench、decode component bench、projection/LoRA bench、gap analyzer 和 result gate；V100 bundle 已跑通，native_jit fast-token 已支持 bsz=1/2/4/8 和 dynamic batching，native_graph 已支持固定 bsz=1/2/4/8 和 dynamic active-batch serving，下一轮重点是更大模型、更多 GPU 和量化 fast path
   - native JIT block-step 已接入 `rwkv7_forward_token`，支持 bsz=1/2/4/8 和 dynamic reorder/drop；native graph replay 已接入 `rwkv7_forward_token` 固定 bsz=1/2/4/8 和 dynamic active-batch 场景；graph cache 管理已补 per-model LRU 和清理接口；下一步做更大模型验证和量化 serving fast path
   - 已新增 bitsandbytes 8bit/4bit 加载、生成、benchmark；下一步需要把量化权重接到 fast-token/native path 或定制 fused int8/int4 projection，解决 generic bnb decode 慢的问题

## 阶段 3：Transformers 原生 PR 方向

需要从 FLA wrapper 迁移为 Transformers 原生 backend：

```text
src/transformers/models/rwkv7/
  configuration_rwkv7.py
  modeling_rwkv7.py
  tokenization_rwkv7.py
  convert_rwkv7_original_to_hf.py
  __init__.py
```

关键要求：

- 不依赖 `fla` 包作为必需依赖。
- 先提供 pure PyTorch/reference recurrent 实现保证 CPU/GPU 正确性。
- 再接可选 CUDA/Triton kernels 做性能路径。
- HF tests：modeling common、generation、tokenizer、PEFT/Trainer smoke。
- 文档和 model card：state cache 不同于 KV cache。

## 当前 blocker

- FLA 源对 V100 的 chunk/backward 编译不稳定，已临时用 `fused_recurrent + TORCHDYNAMO_DISABLE=1` 跑 PEFT。
- FLA `chunk` prefill 在 V100 首次编译耗时很长，性能还没达到官方 `rwkv` 路径。
- 真正悬赏级别需要去 FLA 依赖、补 Transformers test suite、以及大模型/多卡/量化验证。
