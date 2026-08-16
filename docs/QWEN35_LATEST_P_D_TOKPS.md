# 最新 Qwen3.5 对齐：全卡 Prefill / Decode tok/s

更新日期：**2026-08-16**。

这张表只汇总当前正式严格协议中的五张显卡：V100 32GB、RTX 3090、
RTX 4080、RTX 4090 和 RTX 5090。行顺序固定为：

1. 模型尺寸：0.4B/0.8B → 1.5B/2B → 2.9B/4B → 7.2B/9B；
2. 显卡：V100 → RTX 3090 → RTX 4080 → RTX 4090 → RTX 5090；
3. Batch：B1 → B8。

`P / D` 分别表示 Prefill 和 Decode。吞吐均为对应模型、显卡、Batch
下六个 `P{128,512,2048} × D{128,512}` 单元的中位数；B8 是八条序列的
聚合 tok/s，不是单序列速度。显示规则固定为：`>=100 tok/s` 取整数，
`<100 tok/s` 保留一位小数。完整精度保留在链接的 JSONL artifact 中。

参数调整倍率按每个未舍入单元计算：

```text
(RWKV tok/s / Qwen tok/s) × (RWKV active parameters / Qwen active parameters)
```

因此 RWKV 模型更小时，原始速度比会乘以一个小于 1 的参数量系数。
表内倍率是六个单元倍率的中位数，不一定等于两项显示吞吐中位数直接相除。

| 模型对（RWKV / Qwen） | 显卡 | Batch | RWKV P / D tok/s | Qwen P / D tok/s | 参数调整 P / D | 严格门 | 证据 |
|---|---|---:|---:|---:|---:|---|---|
| 0.4B / 0.8B | V100 32GB | B1 | **17,822 / 434** | **4,117 / 111** | 2.626x / 2.342x | P+D 6/6 PASS | [artifact](../bench/v100_qwen35_paired_pd_v1_20260814/README.md) |
| 0.4B / 0.8B | V100 32GB | B8 | **56,270 / 1,784** | **4,140 / 688** | 8.062x / 1.555x | P+D 6/6 PASS | [artifact](../bench/v100_qwen35_paired_pd_v1_20260814/README.md) |
| 0.4B / 0.8B | RTX 3090 | B1 | **32,871 / 665** | **8,337 / 344** | 2.583x / 1.161x | P+D 6/6 PASS | [artifact](../bench/3090_qwen35_paired_pd_v2_20260816/README.md) |
| 0.4B / 0.8B | RTX 3090 | B8 | **79,132 / 3,980** | **33,331 / 1,857** | 1.435x / 1.291x | P+D 6/6 PASS | [artifact](../bench/3090_qwen35_paired_pd_v2_20260816/README.md) |
| 0.4B / 0.8B | RTX 4080 | B1 | **45,562 / 618** | **22,984 / 310** | 1.238x / 1.197x | P+D 6/6 PASS | [artifact](../bench/4080_qwen35_paired_pd_v1_20260814/README.md) |
| 0.4B / 0.8B | RTX 4080 | B8 | **104,224 / 3,344** | **45,067 / 1,711** | 1.318x / 1.174x | P+D 6/6 PASS | [artifact](../bench/4080_qwen35_paired_pd_v1_20260814/README.md) |
| 0.4B / 0.8B | RTX 4090 | B1 | **62,950 / 788** | **9,311 / 407** | 4.306x / 1.161x | P+D 6/6 PASS | [artifact](../bench/4090_qwen35_paired_pd_v2_20260815/README.md) |
| 0.4B / 0.8B | RTX 4090 | B8 | **146,273 / 4,839** | **65,924 / 2,252** | 1.347x / 1.290x | P+D 6/6 PASS | [artifact](../bench/4090_qwen35_paired_pd_v2_20260815/README.md) |
| 0.4B / 0.8B | RTX 5090 | B1 | **58,550 / 1,126** | **14,467 / 559** | 2.484x / 1.211x | P+D 6/6 PASS* | [artifact](../bench/5090_qwen35_paired_decode_v1_20260813/README.md) |
| 0.4B / 0.8B | RTX 5090 | B8 | **206,607 / 6,465** | **93,375 / 3,180** | 1.345x / 1.225x | P+D 6/6 PASS* | [artifact](../bench/5090_qwen35_paired_decode_v1_20260813/README.md) |
| 1.5B / 2B | V100 32GB | B1 | **11,167 / 230** | **3,672 / 83.3** | 2.479x / 2.241x | P+D 6/6 PASS | [artifact](../bench/v100_qwen35_paired_pd_v1_20260814/README.md) |
| 1.5B / 2B | V100 32GB | B8 | **20,861 / 841** | **3,816 / 517** | 4.438x / 1.323x | P+D 6/6 PASS | [artifact](../bench/v100_qwen35_paired_pd_v1_20260814/README.md) |
| 1.5B / 2B | RTX 3090 | B1 | **19,032 / 231** | **8,243 / 175** | 1.877x / 1.073x | P+D 6/6 PASS | [artifact](../bench/3090_qwen35_paired_pd_v2_20260816/README.md) |
| 1.5B / 2B | RTX 3090 | B8 | **29,163 / 1,619** | **16,910 / 1,132** | 1.305x / 1.164x | P+D 6/6 PASS | [artifact](../bench/3090_qwen35_paired_pd_v2_20260816/README.md) |
| 1.5B / 2B | RTX 4080 | B1 | **30,683 / 207** | **19,692 / 154** | 1.323x / 1.092x | P+D 6/6 PASS | [artifact](../bench/4080_qwen35_paired_pd_v1_20260814/README.md) |
| 1.5B / 2B | RTX 4080 | B8 | **39,234 / 1,360** | **22,896 / 960** | 1.339x / 1.150x | P+D 6/6 PASS | [artifact](../bench/4080_qwen35_paired_pd_v1_20260814/README.md) |
| 1.5B / 2B | RTX 4090 | B1 | **36,222 / 349** | **9,220 / 212** | 3.264x / 1.337x | P+D 6/6 PASS | [artifact](../bench/4090_qwen35_paired_pd_v2_20260815/README.md) |
| 1.5B / 2B | RTX 4090 | B8 | **57,929 / 1,909** | **36,953 / 1,302** | 1.272x / 1.191x | P+D 6/6 PASS | [artifact](../bench/4090_qwen35_paired_pd_v2_20260815/README.md) |
| 1.5B / 2B | RTX 5090 | B1 | **33,656 / 548** | **14,177 / 325** | 1.948x / 1.370x | P+D 6/6 PASS* | [artifact](../bench/5090_qwen35_paired_decode_v1_20260813/README.md) |
| 1.5B / 2B | RTX 5090 | B8 | **80,434 / 3,107** | **50,778 / 2,058** | 1.286x / 1.226x | P+D 6/6 PASS* | [artifact](../bench/5090_qwen35_paired_decode_v1_20260813/README.md) |
| 2.9B / 4B | V100 32GB | B1 | **7,066 / 124** | **1,382 / 46.0** | 3.581x / 1.889x | P+D 6/6 PASS | [artifact](../bench/v100_qwen35_paired_pd_v1_20260814/README.md) |
| 2.9B / 4B | V100 32GB | B8 | **10,711 / 536** | **1,505 / 275** | 5.007x / 1.366x | P+D 6/6 PASS | [artifact](../bench/v100_qwen35_paired_pd_v1_20260814/README.md) |
| 2.9B / 4B | RTX 3090 | B1 | **11,970 / 122** | **6,227 / 68.2** | 1.353x / 1.253x | P+D 6/6 PASS | [artifact](../bench/3090_qwen35_paired_pd_v2_20260816/README.md) |
| 2.9B / 4B | RTX 3090 | B8 | **16,889 / 843** | **7,129 / 421** | 1.607x / 1.409x | P+D 6/6 PASS | [artifact](../bench/3090_qwen35_paired_pd_v2_20260816/README.md) |
| 2.9B / 4B | RTX 4080 | B1 | **14,264 / 108** | **8,956 / 63.4** | 1.167x / 1.193x | P+D 6/6 PASS | [artifact](../bench/4080_qwen35_paired_pd_v1_20260814/README.md) |
| 2.9B / 4B | RTX 4080 | B8 | **19,533 / 729** | **9,866 / 422** | 1.393x / 1.211x | P+D 6/6 PASS | [artifact](../bench/4080_qwen35_paired_pd_v1_20260814/README.md) |
| 2.9B / 4B | RTX 4090 | B1 | **19,115 / 191** | **6,946 / 84.5** | 1.951x / 1.588x | P+D 6/6 PASS | [artifact](../bench/4090_qwen35_paired_pd_v2_20260815/README.md) |
| 2.9B / 4B | RTX 4090 | B8 | **28,183 / 948** | **14,458 / 518** | 1.361x / 1.285x | P+D 6/6 PASS | [artifact](../bench/4090_qwen35_paired_pd_v2_20260815/README.md) |
| 2.9B / 4B | RTX 5090 | B1 | **21,706 / 309** | **10,042 / 120** | 1.544x / 1.802x | P+D 6/6 PASS* | [artifact](../bench/5090_qwen35_paired_decode_v1_20260813/README.md) |
| 2.9B / 4B | RTX 5090 | B8 | **36,488 / 1,248** | **21,808 / 731** | 1.195x / 1.197x | P+D 6/6 PASS* | [artifact](../bench/5090_qwen35_paired_decode_v1_20260813/README.md) |
| 7.2B / 9B | V100 32GB | B1 | **3,758 / 56.1** | **1,174 / 31.2** | 2.526x / 1.448x | P+D 6/6 PASS | [artifact](../bench/v100_qwen35_paired_pd_v1_20260814/README.md) |
| 7.2B / 9B | V100 32GB | B8 | **4,708 / 267** | **1,283 / 164** | 2.937x / 1.312x | P+D 6/6 PASS | [artifact](../bench/v100_qwen35_paired_pd_v1_20260814/README.md) |
| 7.2B / 9B | RTX 3090 | B1 | **5,978 / 59.2** | **3,670 / 42.4** | 1.310x / 1.123x | P+D 6/6 PASS | [artifact](../bench/3090_qwen35_paired_pd_v2_20260816/README.md) |
| 7.2B / 9B | RTX 3090 | B8 | **7,402 / 393** | **4,187 / 284** | 1.421x / 1.113x | P+D 6/6 PASS | [artifact](../bench/3090_qwen35_paired_pd_v2_20260816/README.md) |
| 7.2B / 9B | RTX 4090 | B1 | **10,736 / 85.2** | **6,801 / 50.7** | 1.273x / 1.349x | P+D 6/6 PASS | [artifact](../bench/4090_qwen35_paired_pd_v2_20260815/README.md) |
| 7.2B / 9B | RTX 4090 | B8 | **13,658 / 449** | **8,434 / 331** | 1.301x / 1.090x | P+D 6/6 PASS | [artifact](../bench/4090_qwen35_paired_pd_v2_20260815/README.md) |
| 7.2B / 9B | RTX 5090 | B1 | **14,775 / 146** | **10,461 / 79.2** | 1.143x / 1.481x | P+D 6/6 PASS* | [artifact](../bench/5090_qwen35_paired_decode_v1_20260813/README.md) |
| 7.2B / 9B | RTX 5090 | B8 | **19,053 / 867** | **12,199 / 518** | 1.208x / 1.345x | P+D 6/6 PASS* | [artifact](../bench/5090_qwen35_paired_decode_v1_20260813/README.md) |

## 验收边界

- V100、RTX 3090、RTX 4080、RTX 4090：表内每行的六个单元均通过原始与
  参数调整后的 Prefill、Decode 严格门。
- RTX 4080 受 16 GiB 容量限制，只覆盖前三个模型对；缺少 7.2B/9B
  不是失败单元。
- RTX 5090 的 Decode 是原 `paired_decode_v1` validator 的正式严格 48/48
  PASS。标有 `*` 的 Prefill 使用同一批正式 RWKV/Qwen 行进行全精度逐格
  复核：原始 Prefill 48/48、参数调整 Prefill 48/48 均严格 `>1.0x`；参数
  调整最小值/中位数/最大值为 `1.089713x/1.354606x/4.590900x`。原
  Decode-v1 validator 没有输出 Prefill gate 字段，因此此处明确标成复核结果，
  不伪装成原 validator 字段。
- RTX 5070 Laptop、A6000、A800、1080 Ti 等卡没有完成这套当前统一矩阵，
  其历史结果不混入本表。
- 这是推理吞吐比较，不代表模型质量、连续 E2E、TTFT 或 cache-handoff
  latency 结论。
