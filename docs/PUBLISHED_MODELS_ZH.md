# 已发布的 RWKV7-G1 Hugging Face 模型

英文版：[`PUBLISHED_MODELS.md`](PUBLISHED_MODELS.md)

全部可直接加载的 FP16 模型已经汇总到
[`RWKV7-G1 Transformers` Collection](https://huggingface.co/collections/wangyue114514/rwkv7-g1-transformers-6a85b04191034d4c2d1896f1)。
每个尺寸都是独立的 Transformers 模型仓库。不可变的 `v0.7.0` manifest 记录发布
时所用运行库；这些模型仓库与当前 `rwkv7-hf==0.8.0` 运行库兼容。

## 模型矩阵

| 模型 | 参数量 | FP16 权重文件数 | 权重大小 | 建议起步可用内存 |
|---|---:|---:|---:|---:|
| [`rwkv7-g1d-0.1b-hf`](https://huggingface.co/wangyue114514/rwkv7-g1d-0.1b-hf) | 191,034,624 | 1 | 0.38 GB | 1 GB |
| [`rwkv7-g1d-0.4b-hf`](https://huggingface.co/wangyue114514/rwkv7-g1d-0.4b-hf) | 450,767,872 | 1 | 0.90 GB | 2 GB |
| [`rwkv7-g1g-1.5b-hf`](https://huggingface.co/wangyue114514/rwkv7-g1g-1.5b-hf) | 1,527,404,544 | 6 | 3.05 GB | 6 GB |
| [`rwkv7-g1g-2.9b-hf`](https://huggingface.co/wangyue114514/rwkv7-g1g-2.9b-hf) | 2,947,735,040 | 13 | 5.90 GB | 10 GB |
| [`rwkv7-g1g-7.2b-hf`](https://huggingface.co/wangyue114514/rwkv7-g1g-7.2b-hf) | 7,199,141,888 | 4 | 14.40 GB | 20 GB |
| [`rwkv7-g1g-13.3b-hf`](https://huggingface.co/wangyue114514/rwkv7-g1g-13.3b-hf) | 13,269,245,952 | 7 | 26.54 GB | 32 GB |

内存列是首次加载的保守起点，不是所有后端下都成立的峰值显存承诺。实际占用会随
设备、dtype、batch、提示长度、训练、量化和切分方式变化。验证新环境时固定先用
0.1B。

## 直接安装和加载

```bash
python -m pip install "rwkv7-hf==0.8.0"
```

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "wangyue114514/rwkv7-g1d-0.1b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    dtype="auto",
).eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
inputs = tokenizer("User: 你好！ Assistant:", return_tensors="pt")
inputs = {name: value.to(device) for name, value in inputs.items()}
with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=16)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

模型仓库只包含权重、配置、Tokenizer 以及三个很小的 remote-code 入口；维护中的
模型实现和优化算子来自兼容的 PyPI 包，不在六个模型仓库里重复复制。

## 一条命令验收公开发布

在本仓库目录执行：

```bash
python scripts/verify_hf_release.py \
  --model wangyue114514/rwkv7-g1d-0.1b-hf
```

成功时最后显示 `RESULT: PASS`。脚本会检查 Hub revision、转换清单、远端 LFS
大小与 SHA256、配置、Tokenizer、加载键、参数量、有限 logits，并完成一次真实生成。

大模型可以只验证元数据，避免下载全部权重：

```bash
python scripts/verify_hf_release.py \
  --model wangyue114514/rwkv7-g1g-13.3b-hf \
  --metadata-only
```

## 发布模型与原始权重的关系

[`BlinkDL/rwkv7-g1`](https://huggingface.co/BlinkDL/rwkv7-g1) 继续作为权威的
多 checkpoint 原始权重仓库。本页模型是面向用户的 Transformers 成品形态：每个
可独立加载的模型一个仓库，再使用 Collection 汇总。每个模型仓库都包含
`conversion_manifest.json`，记录源 revision、输入输出哈希、参数量、运行时版本和
验收结果。
