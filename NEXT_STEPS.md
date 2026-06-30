# RWKV-7 HF 适配下一步

## 已完成：阶段 1 / wrapper 可用版

- 官方 `.pth` -> HF `model.safetensors`
- HF `config.json` + `generation_config.json`
- remote-code wrapper：`AutoConfig` / `AutoModelForCausalLM`
- RWKV trie slow tokenizer：`AutoTokenizer`
- `generate(use_cache=True)` 跑通
- PEFT LoRA forward/loss/backward 跑通
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

1. 支持全部已发布尺寸的配置推断和转换：0.4B / 1.5B / 2.9B / 7.2B / 13.3B。
2. 增加批量转换脚本和 SHA256 manifest。
3. 加官方 RWKV 对齐测试：
   - prompt logits top-k
   - max/mean abs diff
   - greedy generate token-by-token equality window
4. 补 HF behavior：
   - `save_pretrained` / reload roundtrip
   - `resize_token_embeddings` 禁用或安全处理
   - `gradient_checkpointing_enable`
   - `prepare_inputs_for_generation`/cache shape 文档化
5. 训练路径：
   - PEFT LoRA SFT 小数据跑通
   - TRL `SFTTrainer` smoke
   - 明确 `TORCHDYNAMO_DISABLE=1` 或修 FLA backward compile 问题

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
