# RWKV-7 vs Qwen3.5：完整参数、速度对照与复现

更新日期：**2026-08-12**。数字来自当前主分支已提升的同卡证据。
完整历史、量化路线和逐项遥测仍以 [`BENCHMARK.md`](../BENCHMARK.md) 为准。
[English version](QWEN35_SPEED_COMPARISON.md)

## 先看结论

> 当前正式的 NVIDIA dense FP16 optimized-Qwen 同卡对照共有 **29 个
> GPU/模型/Batch 组合**，另列 **3 个 Apple M5 target-only W4 组合**。
> 表内每一行的 RWKV-7 原始 Prefill 和原始 Decode 中位值均高于 Qwen3.5。
> 原始 Prefill/Decode 可达 **5.41x / 19.79x**；扣除较小参数量带来的天然速度
> 优势后，参数规模校正 Prefill/Decode 可达 **4.39x / 11.85x**。更直观的
> Prefill + Decode 端到端结果中，NVIDIA **116/116 个实测格**和 **29/29 个
> 组合中位值**在原始及参数规模校正后都超过 Qwen3.5；Apple M5 的 **3/3 个
> 组合**在校正后也全部超过。

**RTX 4080 已完成逐格全过：参数规模校正 Prefill 36/36、Decode 36/36
均超过 `1.00x`，全矩阵最小值为 `1.068520x / 1.140700x`。**

**RTX 3090 的最新 g1d/g1i 检查点矩阵也已完成：B1/B8、
P128/P512/P2048 共 24 格的参数规模校正 Prefill 均达到 `>=1.00x`，Qwen
参考全部使用 fail-closed full-FLA 路径。**

- `1.02x` 表示 RWKV 吞吐是 Qwen 的 1.02 倍，即约快 2%。
- Prefill 是处理输入提示词；Decode 是逐 token 生成，后者更接近日常聊天的
  持续生成速度。
- NVIDIA 主表的速度基线只采用 **dense FP16 原始 tok/s**，同时补充参数规模
  校正速度。除 V100 和最新 RTX 3090/5090 明示的形状外，`6格`表示
  `P128/512/2048 × D128/512` 六个形状的中位值；最新 RTX 3090/5090 的
  `3格`表示 `P128/512/2048 × D128`。
- 这里比较的是推理吞吐，不代表任何一方在指令遵循、推理、代码、多语言等
  任务质量上更好；模型质量需要单独的评测数据。

## 参数口径

模型名是发布档位，主表另列 benchmark 遥测得到的精确活跃参数数目：

| 模型对（RWKV / Qwen3.5） | RWKV 精确活跃参数 | Qwen 精确活跃参数 | RWKV/Qwen 参数比 |
|---|---:|---:|---:|
| 0.4B / 0.8B | `450,767,872` | `752,393,024` | `0.599112` |
| 1.5B / 2B | `1,527,404,544` | `1,881,825,088` | `0.811661` |
| 2.9B / 4B | `2,947,735,040` | `4,205,751,296` | `0.700882` |
| 7.2B / 9B | `7,199,141,888` | `8,953,803,264` | `0.804032` |

- **原始速度比** = RWKV tok/s ÷ Qwen tok/s，代表用户实际拿到的吞吐。
- **参数规模校正速度比** = 原始速度比 × RWKV/Qwen 活跃参数比。它把 Qwen
  线性缩放到 RWKV 的活跃参数规模，用于扣除“小模型本来就更快”的天然优势。
- **端到端速度比** =（Qwen Prefill 用时 + Qwen Decode 用时）÷（RWKV
  Prefill 用时 + RWKV Decode 用时）；**参数规模校正端到端速度比** = 原始
  端到端速度比 × RWKV/Qwen 活跃参数比。本页端到端用时指正式计时的推理阶段，
  不包含模型加载。
- 例如 RTX 4090 的 0.4B/0.8B B8：原始 Prefill `1.75x`，参数比
  `0.599112`，校正后为 `1.75 × 0.599112 ≈ 1.05x`。

## NVIDIA：全部正式同卡模型参数与速度

下面不再筛选代表项，而是逐行列出当前正式 optimized-Qwen 对照中所有
GPU、模型对和 Batch。`原始 P / D` 与 `参数规模校正 P / D` 都是
Prefill/Decode 中位值；`E2E 原始 / 校正`是逐格合并 Prefill 与 Decode 用时后
得到的端到端速度比中位值。

RTX 4080 现已通过更严格的逐格门槛：**参数校正 Prefill 36/36、Decode
36/36 全部超过**，全矩阵最小值为 `1.068520x / 1.140700x`。

| GPU | 模型对 | Batch | 范围 | RWKV 活跃参数 | Qwen 活跃参数 | 参数比 | 原始 P / D | 参数规模校正 P / D | E2E 原始 / 校正 | 证据 |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| V100 32GB | 1.5B / 2B | B1 | P512/D64 | 1.527405B | 1.881825B | `0.811661` | **2.82x / 5.91x** | **2.29x / 4.80x** | **5.59x / 4.54x** | [V100](../bench/v100_active_b1b8_20260715/README.md) |
| V100 32GB | 1.5B / 2B | B8 | P512/D64 | 1.527405B | 1.881825B | `0.811661` | **5.41x / 5.27x** | **4.39x / 4.28x** | **5.30x / 4.30x** | [V100](../bench/v100_active_b1b8_20260715/README.md) |
| RTX 3090 | 0.4B / 0.8B | B1 | 3格 | 0.450768B | 0.752393B | `0.599112` | **3.03x / 10.07x** | **1.82x / 6.03x** | **9.69x / 5.81x** | [3090 最新](../bench/3090_g1i_qwen35_prefill_pd_20260812/README.md) |
| RTX 3090 | 0.4B / 0.8B | B8 | 3格 | 0.450768B | 0.752393B | `0.599112` | **2.45x / 7.65x** | **1.47x / 4.58x** | **7.20x / 4.31x** | [3090 最新](../bench/3090_g1i_qwen35_prefill_pd_20260812/README.md) |
| RTX 3090 | 1.5B / 2B | B1 | 3格 | 1.527405B | 1.881825B | `0.811661` | **1.41x / 5.68x** | **1.15x / 4.61x** | **5.46x / 4.43x** | [3090 最新](../bench/3090_g1i_qwen35_prefill_pd_20260812/README.md) |
| RTX 3090 | 1.5B / 2B | B8 | 3格 | 1.527405B | 1.881825B | `0.811661` | **1.64x / 4.45x** | **1.33x / 3.62x** | **4.12x / 3.34x** | [3090 最新](../bench/3090_g1i_qwen35_prefill_pd_20260812/README.md) |
| RTX 3090 | 2.9B / 4B | B1 | 3格 | 2.947735B | 4.205751B | `0.700882` | **1.91x / 4.28x** | **1.34x / 3.00x** | **4.22x / 2.95x** | [3090 最新](../bench/3090_g1i_qwen35_prefill_pd_20260812/README.md) |
| RTX 3090 | 2.9B / 4B | B8 | 3格 | 2.947735B | 4.205751B | `0.700882` | **1.95x / 3.67x** | **1.36x / 2.57x** | **3.52x / 2.47x** | [3090 最新](../bench/3090_g1i_qwen35_prefill_pd_20260812/README.md) |
| RTX 3090 | 7.2B / 9B | B1 | 3格 | 7.199142B | 8.953803B | `0.804032` | **1.63x / 2.40x** | **1.31x / 1.93x** | **2.38x / 1.91x** | [3090 最新](../bench/3090_g1i_qwen35_prefill_pd_20260812/README.md) |
| RTX 3090 | 7.2B / 9B | B8 | 3格 | 7.199142B | 8.953803B | `0.804032` | **1.60x / 2.11x** | **1.28x / 1.70x** | **2.01x / 1.62x** | [3090 最新](../bench/3090_g1i_qwen35_prefill_pd_20260812/README.md) |
| RTX 4080 | 0.4B / 0.8B | B1 | 6格，全过 | 0.450768B | 0.752393B | `0.599112` | **1.83x / 4.91x** | **1.10x / 2.94x** | **4.82x / 2.88x** | [4080 全部 P/D](../bench/4080_adjusted_pd_20260811/README.md) |
| RTX 4080 | 0.4B / 0.8B | B8 | 6格，全过 | 0.450768B | 0.752393B | `0.599112` | **1.98x / 4.17x** | **1.19x / 2.50x** | **4.04x / 2.42x** | [4080 全部 P/D](../bench/4080_adjusted_pd_20260811/README.md) |
| RTX 4080 | 1.5B / 2B | B1 | 6格，全过 | 1.527405B | 1.881825B | `0.811661` | **1.55x / 1.90x** | **1.26x / 1.55x** | **1.90x / 1.54x** | [4080 全部 P/D](../bench/4080_adjusted_pd_20260811/README.md) |
| RTX 4080 | 1.5B / 2B | B8 | 6格，全过 | 1.527405B | 1.881825B | `0.811661` | **1.76x / 1.77x** | **1.43x / 1.44x** | **1.77x / 1.44x** | [4080 全部 P/D](../bench/4080_adjusted_pd_20260811/README.md) |
| RTX 4080 | 2.9B / 4B | B1 | 6格，全过 | 2.947735B | 4.205751B | `0.700882` | **1.75x / 1.63x** | **1.22x / 1.15x** | **1.63x / 1.15x** | [4080 全部 P/D](../bench/4080_adjusted_pd_20260811/README.md) |
| RTX 4080 | 2.9B / 4B | B8 | 6格，全过 | 2.947735B | 4.205751B | `0.700882` | **1.99x / 1.75x** | **1.40x / 1.23x** | **1.77x / 1.24x** | [4080 全部 P/D](../bench/4080_adjusted_pd_20260811/README.md) |
| RTX 4090 | 0.4B / 0.8B | B8 | 6格 | 0.450768B | 0.752393B | `0.599112` | **1.75x / 12.15x** | **1.05x / 7.28x** | **11.46x / 6.86x** | [4090 small](../bench/4090_small_bsz8_20260715/README.md) |
| RTX 4090 | 1.5B / 2B | B8 | 6格 | 1.527405B | 1.881825B | `0.811661` | **1.11x / 5.66x** | **0.90x / 4.59x** | **5.30x / 4.30x** | [4090 small](../bench/4090_small_bsz8_20260715/README.md) |
| RTX 4090 | 2.9B / 4B | B8 | 6格 | 2.947735B | 4.205751B | `0.700882` | **1.42x / 4.24x** | **1.00x / 2.97x** | **3.99x / 2.80x** | [4090 small](../bench/4090_small_bsz8_20260715/README.md) |
| RTX 4090 | 7.2B / 9B | B8 | 6格 | 7.199142B | 8.953803B | `0.804032` | **1.12x / 2.22x** | **0.90x / 1.79x** | **2.11x / 1.69x** | [4090 7.2B](../bench/4090_g1h_7p2_bsz8_20260715/README.md) |
| RTX 5070 Laptop | 1.5B / 2B | B8 | 6格 | 1.527405B | 1.881825B | `0.811661` | **1.33x / 2.62x** | **1.08x / 2.13x** | **2.48x / 2.02x** | [5070](../bench/5070_qwen35_full_fla_bsz8_20260714/README.md) |
| RTX 5090 | 0.4B / 0.8B | B1 | 3格 | 0.450768B | 0.752393B | `0.599112` | **3.86x / 19.79x** | **2.31x / 11.85x** | **18.63x / 11.16x** | [5090 最新](../bench/5090_g1i_qwen35_prefill_pd_sota_20260811/README.md) |
| RTX 5090 | 1.5B / 2B | B1 | 3格 | 1.527405B | 1.881825B | `0.811661` | **2.16x / 9.63x** | **1.75x / 7.82x** | **9.34x / 7.58x** | [5090 最新](../bench/5090_g1i_qwen35_prefill_pd_sota_20260811/README.md) |
| RTX 5090 | 2.9B / 4B | B1 | 3格 | 2.947735B | 4.205751B | `0.700882` | **1.87x / 7.49x** | **1.31x / 5.25x** | **7.13x / 5.00x** | [5090 最新](../bench/5090_g1i_qwen35_prefill_pd_sota_20260811/README.md) |
| RTX 5090 | 7.2B / 9B | B1 | 3格 | 7.199142B | 8.953803B | `0.804032` | **1.42x / 3.50x** | **1.14x / 2.81x** | **3.43x / 2.76x** | [5090 最新](../bench/5090_g1i_qwen35_prefill_pd_sota_20260811/README.md) |
| RTX 5090 | 0.4B / 0.8B | B8 | 3格 | 0.450768B | 0.752393B | `0.599112` | **2.24x / 7.99x** | **1.34x / 4.79x** | **7.61x / 4.56x** | [5090 最新](../bench/5090_g1i_qwen35_prefill_pd_sota_20260811/README.md) |
| RTX 5090 | 1.5B / 2B | B8 | 3格 | 1.527405B | 1.881825B | `0.811661` | **1.43x / 4.77x** | **1.16x / 3.87x** | **4.48x / 3.63x** | [5090 最新](../bench/5090_g1i_qwen35_prefill_pd_sota_20260811/README.md) |
| RTX 5090 | 2.9B / 4B | B8 | 3格 | 2.947735B | 4.205751B | `0.700882` | **1.69x / 3.92x** | **1.19x / 2.75x** | **3.68x / 2.58x** | [5090 最新](../bench/5090_g1i_qwen35_prefill_pd_sota_20260811/README.md) |
| RTX 5090 | 7.2B / 9B | B8 | 3格 | 7.199142B | 8.953803B | `0.804032` | **1.54x / 2.72x** | **1.24x / 2.19x** | **2.54x / 2.05x** | [5090 最新](../bench/5090_g1i_qwen35_prefill_pd_sota_20260811/README.md) |

这张长表按 GPU、模型和 Batch 保留全部正式对照，便于直接查看不同参数档位的
原始吞吐和参数规模校正速度。

### RTX 3090 最新检查点严格门槛

最新 RTX 3090 证据使用 RWKV-7 g1d 0.4B 和 2026-08-05 g1i
1.5B/2.9B/7.2B，对照官方 Qwen3.5 0.8B/2B/4B/9B，并逐格检查
B1/B8、P128/P512/P2048、D128。全部 `24/24` 个 Qwen 参考格均验证
FLA、Triton causal convolution、实时 fused bindings 和 full-fused contract。

严格 Prefill 门槛 `24/24` 全部通过：原始 Prefill 最低/中位为
`1.393044x/1.856646x`，参数规模校正 Prefill 最低/中位为
`1.037869x/1.351562x`；原始 Decode 最低/中位为
`2.105217x/4.361090x`，参数规模校正 Decode 最低/中位为
`1.692661x/3.301953x`。最弱格是 2.9B/4B B8/P128：RWKV 为
`9,860.98 tok/s`，Qwen 为 `6,659.207 tok/s`，参数校正后仍为
`1.037869x`。

精确形状 FP16 accumulation 的正确性门槛覆盖全部直接调用与分块携带形状，
`15/15` 个 Prompt/缓存交接行都达到 cosine `>=0.9999`，greedy token
完全一致。提升策略只适用于实测 RTX 3090 的模型、Batch 和 token-block
形状。完整数据见
[不可变证据](../bench/3090_g1i_qwen35_prefill_pd_20260812/README.md)。

### RTX 5090 最新检查点严格门槛

最新 RTX 5090 行使用 RWKV-7 g1d 0.4B 和 2026-08-05 g1i
1.5B/2.9B/7.2B，对照官方 Qwen3.5 0.8B/2B/4B/9B。全部 24 个 Qwen
参考单元均验证 FLA、Triton causal convolution、实时 fused bindings 和
full-fused contract。

上面的主表展示每组中位值；严格门槛则逐个检查 B1/B8、P128/P512/P2048。
最终 `24/24` 全部通过：原始 Prefill 最低/中位为
`1.347871x/1.819072x`，参数规模校正 Prefill 最低/中位为
`1.072987x/1.317515x`；原始 Decode 最低/中位为
`2.710952x/6.104568x`，参数规模校正 Decode 最低/中位为
`2.179692x/4.330813x`。原始及参数规模校正 E2E 也在全部 `24/24` 格中
高于 `1.00x`。

0.4B/B1/P2048 达到 `61,343.8 tok/s`，是上一候选行的 `2.2495x`。
P2048 graph 对 eager 的正确性门槛在四组模型、B1/B8 上 `8/8` 通过，
Prompt/缓存交接后 cosine 最低为 `0.99999988/0.99999994`，greedy token
全部一致。移除负收益的 7.2B stacked-RKV 路径后，其候选峰值显存从
`17.4-18.6 GiB` 降到 `14.3-15.5 GiB`。完整数据见
[不可变证据](../bench/5090_g1i_qwen35_prefill_pd_sota_20260811/README.md)。

### Apple M5：全部正式 target-only W4 对照

Apple MLX W4 单独成表，以保持每张表内部的后端和精度一致；这里同样同时给出
原始速度和参数规模校正速度：

| 模型对 | Batch / 形状 | RWKV 活跃参数 | Qwen 活跃参数 | 原始 P / D | 参数规模校正 P / D | E2E 原始 / 校正 | 证据 |
|---|---|---:|---:|---:|---:|---:|---|
| 0.4B / 0.8B | B8，cold，P512 字符/D64 | 0.450768B | 0.752393B | **2.04x / 2.04x** | **1.22x / 1.22x** | **2.02x / 1.21x** | [M5 B8](../bench/apple_bsz8_active_m5_20260714/README.md) |
| 1.5B / 2B | B1，P512 字符/D64 | 1.527405B | 1.881825B | **1.67x / 1.44x** | **1.36x / 1.17x** | **1.45x / 1.17x** | [M5 B1](../bench/apple_bsz1_active_m5_20260715/README.md) |
| 1.5B / 2B | B8，cold，P512 字符/D64 | 1.527405B | 1.881825B | **1.41x / 1.40x** | **1.14x / 1.14x** | **1.37x / 1.11x** | [M5 B8](../bench/apple_bsz8_active_m5_20260714/README.md) |

## AMD 和其他硬件

除上面的 NVIDIA 与 Apple 对照外，仓库还覆盖 AMD ROCm、Turing/Ampere/Hopper
以及国产加速器。这里集中展示已实现能力、已验证路径和可直接执行的入口。

| 平台 | 已实现与已验证能力 | 入口 |
|---|---|---|
| AMD Navi 31 / `gfx1100` | Native HF、缓存、分块 Prefill、PEFT、BF16 Trainer、0.1B–13.3B 融合 Decode、40/40 output-head W8/W4 Decode 行 | [AMD ROCm 验证](validation/AMD_ROCM_HF_VALIDATION.md) |
| AMD MI 系列、`gfx1101/gfx1102` | ROCm/HIP 通用执行路径、架构识别与 portable dispatch | [硬件矩阵](HARDWARE_MATRIX.md) |
| Apple M5 | 0.4B/0.8B 与 1.5B/2B target-only W4 对照，MLX、MPS、CoreML 工作流 | [Apple 指南](APPLE_USAGE.md) |
| NVIDIA T4 | Native HF、量化、训练与 production-close 验证 | [T4 证据](../bench/t4_production_close_20260720/README.md) |
| NVIDIA A100/A800 | Ampere CUDA、训练、并行与 HF 工作流 | [硬件矩阵](HARDWARE_MATRIX.md) |
| NVIDIA H100 | Hopper CUDA、Transformers/HF 与 benchmark 执行路径 | [性能指南](PERFORMANCE.md) |
| Ascend、Biren、MetaX、MUSA | 专用后端、运行时适配和兼容性验证 | [硬件矩阵](HARDWARE_MATRIX.md) |

### AMD `gfx1100` 已有的直观数字

下表是 FP16、P128、cached decode；`加速`列展示同一 RWKV 的融合策略相对
通用策略的提升：

| RWKV-7 | B1 Decode | B8 Aggregate Decode | 融合策略 / 通用策略（B1 / B8） |
|---|---:|---:|---:|
| 0.1B | 347.1 tok/s | 2,666.5 tok/s | `1.88x / 2.04x` |
| 0.4B | 141.8 tok/s | 1,073.2 tok/s | `1.75x / 1.74x` |
| 1.5B | 71.3 tok/s | 514.2 tok/s | `1.40x / 1.47x` |
| 2.9B | 47.7 tok/s | 353.0 tok/s | `1.37x / 1.41x` |
| 7.2B | 29.7 tok/s | 213.9 tok/s | `1.23x / 1.29x` |
| 13.3B | 15.5 tok/s | 113.2 tok/s | `1.21x / 1.29x` |

此外，gfx1100 的 output-head W8/W4 在 0.4B–13.3B、B1/B2/B4/B8 的
40/40 Decode 行中都快于对应 RWKV FP16。详见
[AMD 验证说明](validation/AMD_ROCM_HF_VALIDATION.md)、
[融合 Decode 证据](../bench/amd_gfx1100_fused_decode_20260728/README.md)和
[0.4B–13.3B 回归证据](../bench/amd_gfx1100_rebase_validation_20260728/README.md)。

## 对比口径

- NVIDIA 表中两边使用同一张卡、相同 Batch、Prompt/Decode 长度和 FP16
  精度；Apple 表中两边使用各自正式的 MLX W4 路线。
- NVIDIA 的 Qwen3.5 全部记录并验证 **FLA + Triton causal-conv** 优化路径和
  fused operator 绑定。
- NVIDIA 的 RWKV 使用仓库的 Native prefill 与 native-graph cached decode；
  Apple 两边均为 MLX W4 target-only 路线。
- RTX 4080 采用双方各自已验证的优化运行时：RWKV 使用 PyTorch 2.11 的精确
  形状 FP16 accumulation，Qwen 使用 PyTorch 2.6 的 full-FLA 路线；版本和
  后端均记录在证据中，GPU、形状、Batch 和 FP16 精度保持一致。
- 模型按发布档位配对，例如 7.2B 对 9B；表中同时展示原始 tok/s、精确活跃
  参数、参数规模校正速度和 Prefill + Decode 端到端速度。
- NVIDIA 主表统一使用 dense FP16；Apple 表统一使用双方正式的 MLX W4，
  每张表内部保持一致口径。

## GPU 实测复现

下面的命令会在 GPU 上重新加载 RWKV-7 与 Qwen3.5、执行 warmup 和正式计时，
并重新生成 Prefill、Decode、参数规模校正速度和后端绑定结果。

### 1. 准备环境和模型

先使用对应证据目录里的 `environment.json`/`environment.txt` 对齐 PyTorch、
CUDA、Triton、Transformers、FLA 和 bitsandbytes 版本；每张卡按其已验证版本
创建独立环境。

```bash
git clone https://github.com/rwkv-rs/hf-adapter.git
cd hf-adapter
git checkout 28f724259f8438cfcc71de40cf33889c6cf2396e

python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[cuda,fla-reference,quant,torchao]"
```

准备两个本地目录：

1. 官方 RWKV-7 `.pth` 经
   [`scripts/convert_rwkv7_to_hf.py`](../scripts/convert_rwkv7_to_hf.py)
   转换后的 HF 模型目录；
2. 官方 Qwen3.5 HF 模型目录。

转换方法见[中文用户指南](USER_GUIDE_ZH.md#2-下载并转换模型)。先检查 RWKV：

```bash
python examples/check_environment.py --model /path/to/rwkv7-model-hf
```

必须看到 `RESULT: READY` 和 `[PASS] Model directory`。

### 2. 快速完整实测：RTX 4090、1.5B 对 2B、B8

```bash
OUT=/tmp/rwkv-qwen35-4090-b8

PYTHON_BIN=python \
BATCH_SIZES=8 \
PREFILL_CHUNK_SIZE=512 \
  bench/run_4090_qwen35_pair_acceptance.sh \
  rwkv-1.5b__qwen3.5-2b \
  /path/to/rwkv7-g1h-1.5b-hf \
  /path/to/Qwen3.5-2B \
  "$OUT"

test "$(cat "$OUT/pipeline_exit_code.txt")" = 0
python - "$OUT/summary_active_work.json" <<'PY'
import json, sys
from statistics import median

summary = json.load(open(sys.argv[1], encoding="utf-8"))
speed = summary["speed"]
adjusted = summary["active_parameter_work"]
print(
    "raw prefill/decode median:",
    speed["median_prefill_speedup"],
    speed["median_decode_speedup"],
)
print(
    "parameter-adjusted prefill/decode:",
    adjusted["median_prefill_throughput_ratio"],
    adjusted["median_decode_throughput_ratio"],
)
raw_e2e = []
adjusted_e2e = []
for cell in summary["cells"]:
    batch = cell["batch_size"]
    candidate_s = (
        batch * cell["prompt_tokens"] / cell["candidate_prefill_tokps_total"]
        + batch * cell["decode_tokens"] / cell["candidate_decode_tokps_total"]
    )
    reference_s = (
        batch * cell["prompt_tokens"] / cell["reference_prefill_tokps_total"]
        + batch * cell["decode_tokens"] / cell["reference_decode_tokps_total"]
    )
    raw_ratio = reference_s / candidate_s
    raw_e2e.append(raw_ratio)
    adjusted_e2e.append(raw_ratio * cell["active_parameter_ratio"])
print("raw/parameter-adjusted E2E median:", median(raw_e2e), median(adjusted_e2e))
print("red cells:", len(summary["red_cells"]))
PY
```

成功标准：脚本退出码为 0、`pipeline_exit_code.txt=0`、`red cells: 0`，并且
Qwen 行显示 full-FLA 优化路径；结果按中位值和两位小数进行对照。

### 3. 其他显卡入口

| GPU | 实测入口 | 说明 |
|---|---|---|
| V100 | [V100 证据中的命令](../bench/v100_active_b1b8_20260715/README.md#reproduce) | 1.5B/2B，B1/B8 |
| RTX 3090 最新检查点 | [`bench/run_3090_adjusted_prefill_pd.sh`](../bench/run_3090_adjusted_prefill_pd.sh) | 四个模型对、B1/B8、P128/512/2048、D128；逐格校正 Prefill 门槛与 15 格正确性门禁 |
| RTX 4080 | [`bench/run_4080_adjusted_pd.sh`](../bench/run_4080_adjusted_pd.sh) | 一次运行 3 个模型对、B1/B8 全部 36 格，并强制每格参数校正 P/D 均 `>1.00x` |
| RTX 5070 Laptop | [`bench/run_5070_qwen35_full_fla_bsz8.ps1`](../bench/run_5070_qwen35_full_fla_bsz8.ps1) | Windows PowerShell；通过 `-RwkvModel`、`-QwenModel`、`-OutDir` 传路径 |
| RTX 5090 | [`bench/run_5090_qwen35_full_matrix.sh`](../bench/run_5090_qwen35_full_matrix.sh) | 四个模型对、B1/B8 的完整矩阵 |
| RTX 5090 最新检查点 | [严格门槛证据中的命令](../bench/5090_g1i_qwen35_prefill_pd_sota_20260811/README.md#reproduce-the-gate) | 四个模型对、B1/B8、P128/512/2048、D128 |

这些入口都会检查精确 GPU、后端绑定、矩阵覆盖和验收门槛，并在输出目录生成
`pipeline_exit_code.txt`、`matrix_failures.txt`、`summary*.json` 和完整日志。

### 4. Apple M5 GPU 实测

使用 MLX 环境和本地 W4 模型目录，执行 B8 target-only 正式对照：

```bash
PYTHON_BIN=/path/to/python \
MODEL_ROOT=/path/to/models \
COOLDOWN_SECONDS=30 \
INITIAL_COOLDOWN_SECONDS=60 \
  scripts/run_apple_bsz8_target_only_acceptance.sh
```

脚本会在 Apple GPU 上分别运行 RWKV-7 0.4B/1.5B 与 Qwen3.5 0.8B/2B，输出
原始 Prefill/Decode、参数规模校正口径、峰值内存和 token 一致性结果。

### 5. AMD `gfx1100` GPU 实测

使用 ROCm 7.2.1 对应的 PyTorch 环境，并确认用户可访问 `/dev/kfd` 和
`/dev/dri/render*`：

```bash
OUT=/tmp/rwkv7-amd-gfx1100
mkdir -p "$OUT"
set -o pipefail

bash bench/run_amd_rocm_hf_validation.sh \
  HF_DIR=/path/to/rwkv7-g1d-0.1b-hf \
  OUT_DIR="$OUT" |& tee "$OUT/console.log"

grep -F "AMD ROCm HF VALIDATION PASS" "$OUT/console.log"
```

脚本会验证 HIP 可见性和 `gcnArchName`；精确 `gfx1100` 会启用对应的融合
策略和架构调优。若只复现融合 Decode A/B，使用
[AMD 证据中的精简命令](../bench/amd_gfx1100_fused_decode_20260728/README.md#reproduce)。
