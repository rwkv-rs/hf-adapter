# W8/W4 使用教学

RWKV-7 提供三类量化：标准 HF bitsandbytes、原生 MM8/MM4 和 Apple MLX packed
W8/W4。它们的硬件支持和性能不同。本教程严格区分三个结论：

1. **功能通过**：加载、有限 logits、cache decode 和生成通过；
2. **节省内存**：模型 footprint 小于同一 dense baseline；
3. **速度通过**：在精确显卡和 shape 上，配对端到端时间不慢于 baseline。

English version: [`QUANTIZATION_USAGE.md`](QUANTIZATION_USAGE.md)

## 1. 安装并建立 dense baseline

```bash
python -m pip install -e ".[quant]"
python tests/test_quantized_inference.py --model MODEL \
  --device cuda --dtype fp16 --quantization none --max-new-tokens 4
```

保存 JSON 行，其中包含 `model_footprint_mb`、`peak_vram_mb`、生成 token 和
时间遥测。没有匹配 dense 行的量化结果，不能证明更省内存或更快。

## 2. 标准 HF bitsandbytes W8/W4

W8：

```bash
python tests/test_quantized_inference.py --model MODEL \
  --device cuda --dtype fp16 --quantization 8bit --max-new-tokens 4
```

NF4 W4 + double quant：

```bash
python tests/test_quantized_inference.py --model MODEL \
  --device cuda --dtype fp16 --quantization 4bit \
  --bnb-4bit-quant-type nf4 --bnb-4bit-use-double-quant \
  --max-new-tokens 4
```

每条命令都必须输出 `status: pass`、非零量化模块数、有限 logits、生成 token，
最后打印 `PASS`。正式验收不要使用 `--optional`，因为 skip 不等于 pass。

直接 HF API：

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

qconfig = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
model = AutoModelForCausalLM.from_pretrained(
    "MODEL", trust_remote_code=True,
    quantization_config=qconfig, device_map="cuda",
)
```

本仓库中的 bitsandbytes 适合 CUDA 兼容和省显存场景。追求速度时，请使用
同显卡、同模型、同 batch 的 dense fp16 配对结果选择路线。

## 3. bitsandbytes + 原生无 FLA 模型

```bash
RWKV7_NATIVE_MODEL=1 python tests/test_native_bnb_quant_smoke.py \
  --model MODEL --device cuda --dtype fp16 --quantization both
```

8-bit 和 4-bit 都要输出 pass JSON，验证真实量化 Linear、forward/decode/
generate，最后打印 `NATIVE BNB QUANT PASS`。当原生 JIT 没有对应 packed operand
时，native bnb decode 会使用兼容 eager 路径。

## 4. 原生 MM8/MM4

原生量化不依赖 bitsandbytes。在配置中只打开一个模式：

```python
import os
os.environ["RWKV7_NATIVE_MODEL"] = "1"

from transformers import AutoConfig, AutoModelForCausalLM

path = "MODEL"
config = AutoConfig.from_pretrained(path, trust_remote_code=True)
config.use_native_mm8 = True
config.use_native_mm4 = False
config.native_mm8_policy = "speed"       # 或 "memory"
config.native_mm8_min_params = 8_000_000
model = AutoModelForCausalLM.from_pretrained(
    path, trust_remote_code=True, config=config
).eval()
print(model._rwkv7_native_mm_quantization)
print(model._rwkv7_native_mm_replaced_modules)
```

MM4 则设置 `use_native_mm8=False`、`use_native_mm4=True`，并配置
`native_mm4_policy`/`native_mm4_min_params`。MM8 和 MM4 互斥。

- `speed` 保留多数 dense block，只量化选定昂贵投影；省内存较少，但只有这类
  路线可以按精确显卡证据晋升为速度路线。
- `memory` 替换更多 Linear，通常更省内存，但并不保证普遍快于 fp16。

### V100 上的 MM4 decode 配置

前置条件：精确 `sm_70` V100、fp16、可用的 CUDA 12.x 编译工具链，以及已经
转换好的本地 HF 模型目录。先从 1.5B 和最小矩阵开始，不要直接用未验证模型
改服务默认值。

```python
import os
os.environ["RWKV7_NATIVE_MODEL"] = "1"

# Select one exact-card profile before loading the model.
profile = "2.9b"  # "1.5b", "2.9b", or "7.2b"
profiles = {
    "1.5b": ("memory", 128, "1"),
    "2.9b": ("speed", 256, "0"),
    "7.2b": ("memory", 128, "0"),
}
policy, group_size, fused_epilogue = profiles[profile]
os.environ["RWKV7_SM70_W4_FUSED_EPILOGUE"] = fused_epilogue

from transformers import AutoConfig, AutoModelForCausalLM

path = f"/path/to/rwkv7-g1g-{profile}-hf"
config = AutoConfig.from_pretrained(path, trust_remote_code=True)
config.use_native_mm8 = False
config.use_native_mm4 = True
config.native_mm4_policy = policy
config.native_mm4_min_params = 8_000_000
config.native_mm4_group_size = group_size
config.native_mm4_group_policy = "lm_head"
model = AutoModelForCausalLM.from_pretrained(
    path, trust_remote_code=True, config=config, device_map="cuda"
).eval()

assert model._rwkv7_native_mm_quantization == "mm4"
assert model._rwkv7_native_mm_replaced_modules > 0
```

三个配置不能混用：1.5B 使用 `memory + group128 + fused epilogue`，2.9B 使用
`speed + group256 + unfused`，7.2B 使用 `memory + group128 + unfused`。1.5B/7.2B
替换更多模块、更省模型内存，但 prefill 会明显变慢。fused epilogue 的全局默认值
仍为关闭，只有精确的 1.5B 配置显式打开。2.9B 只替换 `lm_head`，因此省内存
较少，但当前七个 paired-fp16 prefill 单元也达到 `1.0006x-1.0603x`。

精确 `sm_70` 上，默认 `RWKV7_SM70_W4_PREFILL_BACKEND=auto` 会在逻辑行数
`>=16` 时选择临时反量化 FP16 + cuBLAS；cached decode 和带 `out=` 的图捕获调用
仍走原来的 DP4A。可用 `RWKV7_SM70_W4_PREFILL_BACKEND=dp4a` 回退做 A/B，或用
`dequant_blas` 强制长行路径；非 `sm_70` 和少于 16 行始终 fail-closed 到 DP4A。
1.5B 另有经过验证的 head-only 速度配置：
`speed + group256 + lm_head + unfused`，在 P128/D128 的 B1/B2/B4/B8
同时通过显存、prefill、decode 和精度门。

直接复制下面的命令做严格验收：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \
python bench/run_v100_sm70_mm4_production_matrix.py \
  --model 2.9b=/path/to/rwkv7-g1g-2.9b-hf \
  --policy speed --group-size 256 --group-policy lm_head \
  --fused-epilogue false \
  --output-dir /tmp/v100-mm4-2p9b
```

验收 1.5B 时将模型、policy/group 改为 `1.5b`、`memory/128`，并增加
`--fused-epilogue true`；7.2B 使用 `memory/128` 和
`--fused-epilogue false`。

可观察的通过标准是命令退出 0，`summary.json` 中 `completed=7`，且 decode、
footprint、logits、完整 greedy 和 repeat determinism 五项都是 `7`。runner 也会
拒绝复用 policy/group/fused 配置不一致的旧结果。

失败时先查看每个 cell 的 `.log`，确认 `CUDA_HOME`、编译器和当前 GPU 确实是
`sm_70`；然后只删除失败 cell 的 JSONL 再重跑同一目录。不要把 fallback 的
Torch 执行、一次 microbench 胜利或 `status=pass` 当成速度验收。当前限制是
full-memory prefill 尚未晋升、2.9B speed 只量化 `lm_head`、其他显卡和模型必须
重新跑精确矩阵。需要 AI 执行时只使用统一入口
[`AI_ASSISTED_SETUP.md`](AI_ASSISTED_SETUP.md)，选择其中的“量化验收”任务。

### Tesla T4 上的精确卡路线

Tesla T4 默认使用精确设备名保护的 DP4A W8/W4 内核。不要在 RTX 2080 或其他
`sm_75` 显卡强制打开这条路线。端到端验收：

```bash
PYTHONPATH=. python bench/bench_native_quant_e2e_decode.py \
  --hf-dir /path/to/rwkv7-hf --model-size-label 0.4b \
  --device cuda --dtype fp16 --attn-mode fused_recurrent \
  --fast-token-backend native_graph \
  --quantizations none mm8 mm4 --policy speed \
  --batch-size 8 --prompt-tokens 64 --decode-tokens 32 \
  --results /tmp/t4-quant-speed.jsonl
```

`speed` 是 `lm_head` 路线：本次精确 T4 矩阵中 W8/W4 decode 26/26 不慢于
fp16，但省显存幅度较小。把 `--policy` 改成 `memory` 并降低
`--min-params` 会量化全模型，footprint 可降到 W8 `0.5291x–0.6331x`、W4
`0.3004x–0.4542x`，但目前不能保证所有 prefill 和 B4/B8 decode 不慢于
fp16。服务默认值必须按需求选择这两条互不替代的路线。

完整数字与失败边界：
[`../bench/t4_production_close_20260720/`](../bench/t4_production_close_20260720/README.md)。

先验证 config round-trip，再验证真实 MM8 持久化：

```bash
python tests/test_native_quant_config.py
python tests/test_native_mm8_persist.py --model MODEL
```

第一条打印 `NATIVE QUANT CONFIG PASS`；第二条打印 `PASS`，并检查重载后的
MM8 模块和 cosine。持久化 config 会在重载时重新打包符合条件的 Linear。

## 5. RTX 5090 g1h BN/TN Tensor Core W4

官方 g1h 1.5B、2.9B、7.2B 和 13.3B BF16 模型在 RTX 5090 上可以使用已晋升
的推理路径。先把 BF16 模型放到 CUDA，再调用 TorchAO W4 的 `speed` policy；
运行时按模型自动选择 Marlin FFN、`lm_head` 和保留的高敏感层。不要手工指定
BN/TN 或 layer exception。

```python
import torch
from transformers import AutoModelForCausalLM
from rwkv7_hf.native_quant_torchao import quantize_model_torchao_w4

path = "/path/to/rwkv7-g1h-7.2b-hf"
model = AutoModelForCausalLM.from_pretrained(
    path,
    trust_remote_code=True,
    dtype=torch.bfloat16,
).eval().to("cuda")

replaced = quantize_model_torchao_w4(
    model,
    min_params=1,
    policy="speed",
    group_size=128,
)
assert replaced == 65
assert model._rwkv7_native_mm_quantization == "marlin_w4_5090_hybrid"
assert model._rwkv7_native_mm_exact_5090_speed_modules == 64
assert model._rwkv7_native_mm_exact_5090_kernel == "bntn_marlin_bf16_w4"
assert model._rwkv7_native_mm_fused_relu2_ffn_modules == 32
```

默认 profile：

| 模型 | 自动配置 | 替换模块数 |
|---|---|---:|
| g1h 1.5B | dense head，最后一个 FFN 保持 BF16 | 46 |
| g1h 2.9B | dense head，全部 32 个 FFN 使用 Marlin | 64 |
| g1h 7.2B | TorchAO W4 head，全部 32 个 FFN 使用 Marlin | 65 |
| g1h 13.3B | TorchAO W4 head，最后一个 FFN 保持 BF16 | 121 |

这些差异来自配对质量门，不是临时手工配置。直接调用上面的 API 时会根据
hidden/intermediate/layer count 自动选择。

首次使用会通过 PyTorch extension cache 编译 vendored Marlin CUDA 源码，需要与
PyTorch 匹配的本地 CUDA toolkit。该路径仅用于推理。RTX 5090/SM120、BF16、
group128、精确 FFN 角色/shape 任一不匹配时，不会误入这条生产 kernel。

端到端复现：

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export TORCH_CUDA_ARCH_LIST=12.0
export PYTHONPATH=$PWD

python bench/bench_native_quant_e2e_decode.py \
  --hf-dir /path/to/rwkv7-g1h-7.2b-hf \
  --model-size-label 7.2b --dtype bf16 --device cuda \
  --attn-mode fused_recurrent --fast-cache true \
  --fast-token-backend native_graph \
  --single-quantization torchao_w4 --min-params 1 --policy speed \
  --batch-size 8 --prompt-tokens 128 --decode-tokens 128 \
  --warmup 1 --timing-repeats 5 --paired-baseline \
  --results /tmp/rwkv7_5090_bntn_w4.jsonl
```

四档 g1h 模型的 B1/B8 prefill、decode、footprint、prompt/final cosine 和
next-token 均通过精确验收。最紧的 prefill 是 7.2B B1 `1.0010x`，最紧的
decode 是 1.5B B1 `1.1854x`，最大 footprint 是 1.5B `0.6250x`，最低新增
prompt/final cosine 是 13.3B B8 `0.99955201/0.99955237`。原始证据：
[`../bench/5090_bntn_all_models_20260716/`](../bench/5090_bntn_all_models_20260716/README.md)。

需要为新的 shape 做实验时，先运行 `bench/bench_marlin_bn_tn.py`，再用
`bench/build_marlin_autotune_profile.py` 生成精确 GPU/runtime JSON。只有显式设置
`RWKV7_MARLIN_AUTOTUNE_PROFILE` 才会读取该文件；未知或版本不匹配的 profile
自动回退，不会改动生产默认值。

## 6. RTX 4080 B1/B8 配对验收

**前置条件和支持环境。** 使用 Linux、单张 RTX 4080 16GB，以及已验证的
PyTorch `2.6.0+cu124`、Triton `3.2.0`、TorchAO `0.16.0` 环境；另需
Transformers、FLA、causal-conv1d 和 bitsandbytes。4080 验收脚本默认严格检查
前三个版本，避免无意升级 kernel 编译器后继续沿用旧性能结论。仓库的 `cuda` 和
`quant` extra 不再单独升级 Triton，而是使用当前 PyTorch 配套的 Triton：

```bash
python -m pip install -e ".[train]"
python -m pip install bitsandbytes==0.49.2 torchao==0.16.0
```

仓库的通用 `torchao` extra 不再设置全局最低版本；4090、5090 和未来显卡各自
保留已经验证的软件栈，不会因 4080 的版本需求被强制升级。非验收环境实验请直接
使用通用 benchmark 入口；正式 4080 验收不提供跳过版本检查的开关。

**最小安全模型和输入。** 准备一组本地模型目录。支持的官方配对是 RWKV-7
0.4B/Qwen3.5-0.8B、RWKV-7 1.5B/Qwen3.5-2B 和 RWKV-7 2.9B/Qwen3.5-4B。
每次选择 B1 或 B8，脚本会运行 prompt 128/512/2048、decode 128/512；使用新的
输出目录，避免覆盖以前的结果。

**直接运行。** 在仓库根目录执行：

```bash
BATCH_SIZE=1 CUDA_VISIBLE_DEVICES=0 \
PYTHON_BIN=python \
bash bench/run_4080_qwen35_pair_acceptance.sh \
  rwkv-1.5b__qwen3.5-2b \
  /path/to/rwkv7-g1g-1.5b-hf \
  /path/to/Qwen3.5-2B \
  /tmp/rtx4080-acceptance
```

把 `BATCH_SIZE=1` 改为 `BATCH_SIZE=8` 即可运行 B8。另两组模型只需替换配对名
和两个模型目录：`rwkv-0.4b__qwen3.5-0.8b` 或
`rwkv-2.9b__qwen3.5-4b`。

**通过标准。** 命令退出码为 0，`matrix_failures.txt` 与
`pipeline_exit_code.txt` 都是 0，`summary.json.status` 为 `pass`，coverage 为
dense candidate/reference `6/6`、memory/paired quant `12/12`，且 `errors` 为空。
Qwen 行必须显示 full FLA；A8W8/W4 行还必须满足 decode、完整单元总耗时、
footprint、cosine 和 greedy 门槛。

**失败恢复和当前边界。** 保留输出目录中的 JSONL 和 `logs/`，先查看失败单元
日志；Qwen 失败时优先核对 FLA 与 causal-conv1d 绑定。修复环境后可把新结果写入
新的输出目录。BNB8/BNB4 是全模型省内存路线；配对速度结论只覆盖输出头 A8W8/
TorchAO-W4，不代表所有显卡和所有全模型量化都更快。

完整的 B1/B8 三组模型示例、环境版本和大模型显存边界见
[`../bench/4080_full_model_ladder_20260719/README.md`](../bench/4080_full_model_ladder_20260719/README.md)。需要 AI 代为检查环境、填写路径或解释失败时，统一使用
[`AI_ASSISTED_SETUP.md`](AI_ASSISTED_SETUP.md) 的“量化验收”入口。

## 7. 如何验收或否决量化路线

必须固定模型、显卡、dtype、batch size、prompt 长度和 decode 长度：

| 门槛 | 必须提供的证据 |
|---|---|
| 功能 | 量化模块真实存在；logits 有限；forward/cache decode/generate 退出 0 |
| 质量 | dense/quant logits 达到声明的 cosine/error 门槛，greedy next token 一致 |
| 内存 | quant `model_footprint_mb` 更低；峰值显存单独报告 |
| 速度 | 配对、预热后的 prefill/decode 端到端时间，不能拿 microbench 替代 |
| 可复现 | 完整 policy、threshold、替换模块数、依赖版本、GPU 和命令 |

先查 [`QUANTIZATION.md`](QUANTIZATION.md) 和
[`HARDWARE_MATRIX.md`](HARDWARE_MATRIX.md) 是否已有精确卡晋升证据。没有匹配
行时，只能写成“本地实验”。

## 8. Apple MLX W8/W4

MLX 使用独立 packed runtime，不使用 bitsandbytes。转换、生成、会话和 M5
证据边界见 [`APPLE_USAGE.md`](APPLE_USAGE.md#4-packed-mlx-w8w4)。

## 9. 交给 AI 执行

需要 AI 协助时，请打开 [`AI_ASSISTED_SETUP.md`](AI_ASSISTED_SETUP.md) 并选择
“量化验收”。AI 会返回完整命令、退出码和验收结果。
