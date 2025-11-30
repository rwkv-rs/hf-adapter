# 推理、转换和缓存教学

本教程覆盖第一次生成之外的 HF 适配能力：可复现转换、无 FLA 原生后端、
loss 和 mask、模型保存迁移、循环状态复用、动态批处理以及分块 prefill。

前置条件：先完成 [`USER_GUIDE_ZH.md`](USER_GUIDE_ZH.md)，并把下文的 `MODEL`
替换成已经通过目录检查的 HF 模型路径。

## 1. 批量转换 checkpoint

先只枚举文件、计算 SHA256 并生成 manifest，不加载模型权重：

```bash
python scripts/batch_convert_rwkv7_to_hf.py \
  --input-dir /path/to/official-pth-files \
  --output-root /path/to/hf-models \
  --vocab-file /path/to/rwkv_vocab_v20230424.txt \
  --precision fp16 --attn-mode fused_recurrent --no-fuse-norm \
  --max-shard-size 5GB --low-memory --dry-run
```

检查 `/path/to/hf-models/manifest.json`，确认后删除 `--dry-run` 再正式转换。
退出码必须为 0，每个请求项都应是 `converted` 或有意的 `skipped`，不能出现
`failed`。`--force` 会覆盖已有输出，只能在明确需要时使用。

单个模型仍使用第一次运行教程中的 `convert_rwkv7_to_hf.py`。`--low-memory`
只降低转换过程的主机内存，不会降低加载模型所需的 RAM/VRAM。

转换目录中包含适配器 remote code 的快照。仓库更新后，可以先预览，再只刷新
Python 代码，不重写大权重：

```bash
python scripts/sync_hf_adapter_code.py MODEL --dry-run
python scripts/sync_hf_adapter_code.py MODEL
```

两条命令都必须退出 0。生产目录刷新前先备份或纳入版本管理，刷新后重新运行
生成和 reload 验收。

## 2. 选择 FLA 或便携原生后端

在已验证的 Linux NVIDIA 环境使用优化 FLA：

```bash
python examples/generate.py --model MODEL --prompt "Hello" \
  --device cuda --dtype fp16 --backend fla --max-new-tokens 8
```

在 CPU、MPS 或没有 FLA 的 CUDA 环境使用原生后端：

```bash
python examples/generate.py --model MODEL --prompt "Hello" \
  --device cpu --dtype fp32 --backend native --max-new-tokens 8
```

退出码为 0 且打印新文本才算通过。原生后端是兼容和优化承载路线，不能据此
宣称它在所有显卡和 shape 上都快于 FLA。

直接使用 HF API 时，要在加载模型前设置：

```python
import os
os.environ["RWKV7_NATIVE_MODEL"] = "1"

from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("MODEL", trust_remote_code=True)
```

## 3. 计算 causal loss 并使用 attention mask

适配器接受标准 HF token batch、`attention_mask` 和 `labels`：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

path = "MODEL"
tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    path, trust_remote_code=True, dtype=torch.float32
).train()
batch = tok(["alpha beta", "gamma"], padding=True, return_tensors="pt")
out = model(
    input_ids=batch["input_ids"],
    attention_mask=batch["attention_mask"],
    labels=batch["input_ids"],
    use_cache=False,
)
assert torch.isfinite(out.loss)
out.loss.backward()
print("PASS", float(out.loss))
```

第一次反向传播使用小模型。更完整的训练验收见
[`TRAINING_WORKFLOWS.md`](TRAINING_WORKFLOWS.md)。

## 4. 保存、重载和离线运行

使用标准 HF 目录合同：

```python
model.save_pretrained("saved-model", safe_serialization=True)
tok.save_pretrained("saved-model")
```

验证保存前后 logits：

```bash
python tests/test_reload_roundtrip.py \
  --model MODEL --device cuda --dtype fp16
```

命令必须打印 `PASS`。模型准备在本地后，可以禁止联网：

```bash
python examples/generate.py --model saved-model --prompt "Hello" \
  --local-files-only --max-new-tokens 8
```

## 5. 复用循环状态

RWKV cache 是固定大小的循环状态，不是不断增长的 Transformer KV cache。
保留 forward 返回的对象并传给下一个 token：

```python
with torch.inference_mode():
    prefill = model(**batch, use_cache=True, logits_to_keep=1)
    state = prefill.past_key_values
    next_id = prefill.logits[:, -1:].argmax(dim=-1)
    step = model(next_id, past_key_values=state, use_cache=True, logits_to_keep=1)

print(step.past_key_values.rwkv7_cache_metrics())
```

cache 支持 `clone()`、`detach()`、`select_batch()`/`batch_select()`、
`reorder_cache()`、`reset()` 和 `.to(device=...)`。例如删除已结束请求，并在不
修改原对象的情况下 offload/restore：

```python
keep = torch.tensor([0, 2], dtype=torch.long, device=next_id.device)
active = state.select_batch(keep, inplace=False)
parked = active.to(device="cpu", inplace=False)
restored = parked.to(device=next_id.device, inplace=False)
```

在真实模型上验证 batch cache 和逐行一致性：

```bash
python tests/test_batch_cache.py --model MODEL --device cuda \
  --dtype fp16 --batch-sizes 1 2 4 --prompt-tokens 64 --decode-steps 8
```

最后打印 `PASS` 才算通过。

## 6. 动态批处理

动态批处理可以重排活动请求并移除已结束请求，同时保留各自的循环状态：

```bash
python tests/test_dynamic_batch_cache.py --model MODEL --device cuda \
  --dtype fp16 --batch-size 3 --prompt-tokens 64 --decode-steps 4 \
  --modes forward fast_token
```

两个 mode 都要打印自己的 `PASS`，最后还要有总 `PASS`。这只证明测试 shape
的 select/reorder/drop 语义；排队、准入、超时和网络 serving 仍由上层负责。

## 7. 对长 prompt 做分块 prefill

模型 helper 会在 prompt chunk 之间携带循环状态，并只保留需要的 logits：

```python
with torch.inference_mode():
    out = model.rwkv7_prefill_chunks(
        batch["input_ids"],
        attention_mask=batch.get("attention_mask"),
        chunk_size=256,
        logits_to_keep=1,
    )
```

新模型/新显卡部署前，先比较分块和普通 prefill：

```bash
python tests/test_chunked_prefill.py --model MODEL --device cuda \
  --dtype fp16 --batch-size 2 --chunk-sizes 1 2 4 8
```

每个 chunk 的差异输出后必须打印 `PASS`。生产 chunk size 要依据同卡内存和
吞吐测试选择；smoke 默认值不是性能推荐。

## 8. 交给 AI 执行

需要 AI 协助时，请打开 [`AI_ASSISTED_SETUP.md`](AI_ASSISTED_SETUP.md)，选择
“转换与推理”或“缓存与分块 prefill”。AI 会返回完整命令、退出码和验收结果。

## 完整 HF API 合同门槛

发布新转换或刚刷新过 remote code 的模型目录前，运行：

```bash
python tests/test_hf_api_contract.py --model MODEL \
  --device cuda --dtype fp16 --attn-mode fused_recurrent
```

命令会检查固定词表行为、generation 输入准备、循环 cache 重排/beam 生成和
gradient checkpointing 开关，最后必须打印 `PASS`。
