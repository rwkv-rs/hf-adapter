# RWKV-7 HF 适配下一步

## 已完成：阶段 1 / wrapper 可用版

- 官方 `.pth` -> HF `model.safetensors`
- HF `config.json` + `generation_config.json`
- remote-code wrapper：`AutoConfig` / `AutoModelForCausalLM`
- RWKV trie slow tokenizer：`AutoTokenizer`
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
- `tests/test_hf_training_smoke.py`：HF Trainer / TRL SFTTrainer 1-step LoRA smoke。
- `bench/bench_decode_breakdown.py`：decode 瓶颈拆分。
- `bench/bench_speed.py` 已改成 serving-style prefill：`use_cache=True + logits_to_keep=1`。
- `bench/profile_decode.py`：单 token decode profiler。
- `scripts/convert_rwkv7_to_hf.py` 新增 `--no-fuse-norm`，作为当前 V100 推理推荐配置。
- remote config 改为唯一 `rwkv7_hf_adapter` model_type，避免 Transformers 环境中已注册的 FLA `rwkv7` 本地类绕过本仓库 wrapper。

当前 V100 结论：

- correctness：`fuse_norm=false` 下 top5/argmax/cosine/greedy64 均通过，fp16 max_abs 约 0.072。
- memory：HF 406.4 MB vs official 406.2 MB，0.1B serving path 已基本持平。
- speed：`fuse_norm=false` + `RWKV7StateCache` 下真实 remote-code HF decode 约 41.2 tok/s，official 约 92.5 tok/s；decode 仍是主优化点。
- profiler：`fuse_norm=true` 的 FLA `LayerNormFunction` CPU 开销很大，native norm 把 norm CPU total 从约 54.8ms/6tok 降到约 6.6ms/6tok。
- breakdown：argmax 开销约等于 0，`chunk` 和 `fused_recurrent` 单 token decode 基本一样，剩余瓶颈在 HF/FLA model+state/cache+小 kernel launch 路径。

下一步继续补：

1. 支持全部已发布尺寸的配置推断和转换：0.4B / 1.5B / 2.9B / 7.2B / 13.3B。
2. 增加批量转换脚本和 SHA256 manifest。
3. 补 HF behavior：
   - `resize_token_embeddings` 禁用或安全处理
   - `gradient_checkpointing_enable`
   - `prepare_inputs_for_generation`/cache shape 文档化
4. 训练路径：
   - 已跑通 PEFT LoRA backward、HF Trainer 1-step、TRL `SFTTrainer` 1-step
   - 继续扩大到真实 SFT 小数据、多 batch、gradient accumulation
   - 当前 smoke 明确 `TORCHDYNAMO_DISABLE=1`，并关闭 `use_l2warp` 避免 Trainer loss 原地缩放与 L2Wrap backward 冲突
5. 性能路径：
   - 继续 profile 单 token decode
   - `RWKV7StateCache` 已减少 generic CacheLayer 开销；下一步继续减少 `CausalLMOutputWithPast` / per-layer update / tiny kernel launch 开销
   - 做专用 fast decode entrypoint

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
