# Documentation map

本页是仓库文档总入口。整理原则是:**状态看 `HF_STATUS.md`,执行看 `HF_TODO.md`,数字看 `BENCHMARK.md`,原始证据看 `bench/`,规则看 `docs/reference` / `docs/BACKENDS.md`。**

如果文档之间出现口径冲突,优先级如下:

1. 真实 benchmark / validation 原始记录: [`../bench/README.md`](../bench/README.md), [`../bench/INDEX.md`](../bench/INDEX.md),各 evidence 目录的 `README.md` 与 `.jsonl`。
2. 当前数字汇总与验收合同: [`../BENCHMARK.md`](../BENCHMARK.md)。
3. 当前完成度与缺口: [`../HF_STATUS.md`](../HF_STATUS.md), [`../HF_TODO.md`](../HF_TODO.md), [`reference/HF_CRITERIA.md`](reference/HF_CRITERIA.md)。
4. 工程路线与规则: [`performance/FUSED_BACKEND.md`](performance/FUSED_BACKEND.md), [`native_fused_roadmap.md`](native_fused_roadmap.md), [`BACKENDS.md`](BACKENDS.md)。
5. 历史计划 / 旧结论: [`archive/NEXT_STEPS.md`](archive/NEXT_STEPS.md) 和旧 bench evidence,只作为上下文。

## 先读哪几个

| 你要做什么 | 阅读顺序 |
|---|---|
| 快速了解项目 | [`../README.md`](../README.md) → [`../HF_STATUS.md`](../HF_STATUS.md) → [`../HF_TODO.md`](../HF_TODO.md) |
| 做 HF 适配贡献 | [`../CONTRIBUTING.md`](../CONTRIBUTING.md) → [`reference/HF_CRITERIA.md`](reference/HF_CRITERIA.md) → [`../HF_TODO.md`](../HF_TODO.md) |
| 跑 benchmark / 整理证据 | [`../bench/README.md`](../bench/README.md) → [`../bench/INDEX.md`](../bench/INDEX.md) → [`../BENCHMARK.md`](../BENCHMARK.md) |
| 做 Albatross / fused 性能 | [`performance/FUSED_BACKEND.md`](performance/FUSED_BACKEND.md) → [`native_fused_roadmap.md`](native_fused_roadmap.md) → [`../BENCHMARK.md`](../BENCHMARK.md) |
| 做 Apple / MLX / Qwen3.5 | [`hardware/APPLE_SILICON.md`](hardware/APPLE_SILICON.md) → [`hardware/QWEN35_APPLE_BASELINE.md`](hardware/QWEN35_APPLE_BASELINE.md) → [`../bench/INDEX.md`](../bench/INDEX.md) 中 `apple_*` evidence |
| 做新卡适配 | [`BACKENDS.md`](BACKENDS.md) → 对应 [`validation/`](validation/) 或 [`hardware/`](hardware/) 文档 → [`../bench/README.md`](../bench/README.md) |
| 做 MATH500 / 质量对齐 | [`validation/math500_acceptance.md`](validation/math500_acceptance.md) → [`validation/math500_accuracy_parity.md`](validation/math500_accuracy_parity.md) → [`../bench/INDEX.md`](../bench/INDEX.md) 中 `math500_*` evidence |
| 做发布 / 贡献归因 | [`../CONTRIBUTORS.md`](../CONTRIBUTORS.md) → [`../CONTRIBUTIONS.md`](../CONTRIBUTIONS.md) → [`../CONTRIBUTING.md`](../CONTRIBUTING.md) |

## 顶层文档

| 文档 | 作用 |
|---|---|
| [`../README.md`](../README.md) | 项目概览、安装/转换/推理入口、当前主要能力摘要。 |
| [`../HF_STATUS.md`](../HF_STATUS.md) | HF 适配当前状态快照、硬件矩阵、主要缺口入口。 |
| [`../HF_TODO.md`](../HF_TODO.md) | 贡献者 TODO:训练矩阵、ZeRO、卡适配、Apple、性能、量化。 |
| [`../BENCHMARK.md`](../BENCHMARK.md) | 数字汇总和 benchmark contract:速度、显存、精度、硬件 evidence。 |
| [`../CONTRIBUTING.md`](../CONTRIBUTING.md) | 如何贡献、最小测试、硬件报告模板、PR checklist。 |
| [`../CONTRIBUTORS.md`](../CONTRIBUTORS.md) | 贡献者身份映射和 attribution block。 |
| [`../CONTRIBUTIONS.md`](../CONTRIBUTIONS.md) | 贡献 ledger、评分/归因说明、关键复现门禁。 |
| [`../AGENTS.md`](../AGENTS.md) | agent / 自动化开发约束、当前目标、常用命令。 |
| [`../TODO_DPLR_WY.md`](../TODO_DPLR_WY.md) | DPLR/WY compiled prefill 临时任务记录;性能分支上下文。 |

## 规则 / 路线文档

| 文档 | 作用 |
|---|---|
| [`reference/HF_CRITERIA.md`](reference/HF_CRITERIA.md) | HF 适配验收准则:最终门禁、已验证进展、最大缺口、优化流程。 |
| [`BACKENDS.md`](BACKENDS.md) | backend 边界和硬件特化规则:哪些代码能放卡特化,哪些不能污染核心路径。 |
| [`performance/FUSED_BACKEND.md`](performance/FUSED_BACKEND.md) | native graph / fused fp16 / fused W8-W4 的性能路线和目标 ladder。 |
| [`native_fused_roadmap.md`](native_fused_roadmap.md) | native fused backend 的 kernel 边界、Albatross layout/autotune、DPLR/chunked prefill 路线。 |
| [`archive/NEXT_STEPS.md`](archive/NEXT_STEPS.md) | 历史下一步计划;仅作背景,不要覆盖当前 TODO/STATUS。 |

## 硬件 / 平台文档

| 文档 | 作用 | 对应 bench evidence |
|---|---|---|
| [`hardware/APPLE_SILICON.md`](hardware/APPLE_SILICON.md) | Apple Silicon / MPS / MLX 适配计划、smoke 命令、已知限制。 | [`../bench/INDEX.md`](../bench/INDEX.md) 中 `apple_*` |
| [`hardware/QWEN35_APPLE_BASELINE.md`](hardware/QWEN35_APPLE_BASELINE.md) | Qwen3.5 Apple/mobile baseline、同机比较指标、runner 和 gate。 | `apple_qwen35_*`, `apple_scan_prefill_*`, `apple_e2e_scan_prefill_*` |
| [`hardware/APPLE_QWEN35_LIVE_EVIDENCE_20260707.md`](hardware/APPLE_QWEN35_LIVE_EVIDENCE_20260707.md) | 2026-07-07 Apple/Qwen3.5 live evidence notes。 | `apple_qwen35_live_m5_20260707`, `apple_qwen35_2b_tokenonly_m5_20260707` |
| [`hardware/BLACKWELL_50SERIES.md`](hardware/BLACKWELL_50SERIES.md) | RTX 50 系 / Blackwell 兼容、5090/5070 实测和问题记录。 | `5090_blackwell_*` |
| [`validation/V100_HF_VALIDATION.md`](validation/V100_HF_VALIDATION.md) | V100 HF 训练/量化/ZeRO 验证矩阵。 | `results_v100_zero3_resume_2gpu_20260703.jsonl` 等 |
| [`validation/A100_HF_VALIDATION.md`](validation/A100_HF_VALIDATION.md) | A100 40GB HF 训练/量化/ZeRO 验证矩阵。 | `BENCHMARK.md` A100 段 |
| [`validation/A800_HF_VALIDATION.md`](validation/A800_HF_VALIDATION.md) | A800 80GB 适配、native mm8/mm4、ZeRO 和大模型 smoke。 | `BENCHMARK.md` A800 段 |

## Benchmark / evidence 文档

| 文档 | 作用 |
|---|---|
| [`../bench/README.md`](../bench/README.md) | bench 目录规则:命名、README 必填项、promotion 规则、validation 命令。 |
| [`../bench/INDEX.md`](../bench/INDEX.md) | 自动/半自动整理的 evidence 目录和 top-level bench 脚本索引。 |
| [`../bench/4090_validation_summary.md`](../bench/4090_validation_summary.md) | RTX 4090 fused backend validation summary。 |
| 各 `../bench/<topic>_<hardware>_<date>/README.md` | 每次硬件/性能/质量实验的原始记录入口。 |

Bench evidence 的整理原则:

- `BENCHMARK.md` 只写已经有 evidence 支撑的数字摘要。
- 新实验先进入 `bench/<topic>_<hardware>_<date>/`,写 README + JSONL。
- 正收益可以在 `BENCHMARK.md` 或平台文档升格;负收益也保留,防止重复试错。
- Apple/Qwen3.5、Albatross、MATH500 这种 comparison 必须同时保留 raw rows 和 compare/gate rows。

## MATH500 / 质量文档

| 文档 | 作用 | 对应 evidence |
|---|---|---|
| [`validation/math500_acceptance.md`](validation/math500_acceptance.md) | MATH500 avg@64 接受流程、HF run、compare gate、final runner。 | `math500_final_acceptance_*`, `math500_hf_seed43_*` |
| [`validation/math500_accuracy_parity.md`](validation/math500_accuracy_parity.md) | 质量 parity 探索:gap 报告、logits parity、RNG/refill、seed sweep。 | `math500_gap_*`, `math500_logits_parity_*`, `math500_rng_*`, `math500_stratified64_*` |

## 文档更新规则

| 改动类型 | 必须更新 |
|---|---|
| 新 benchmark / 新证据 | `bench/<...>/README.md` + `.jsonl`;必要时更新 [`../bench/INDEX.md`](../bench/INDEX.md)。 |
| 新硬件通过 | 对应 `docs/hardware` 或 `docs/validation` 文档 + [`../BENCHMARK.md`](../BENCHMARK.md) 摘要。 |
| HF 功能完成度变化 | [`../HF_STATUS.md`](../HF_STATUS.md) 和 [`../HF_TODO.md`](../HF_TODO.md)。 |
| 性能路线变化 | [`performance/FUSED_BACKEND.md`](performance/FUSED_BACKEND.md) 和/或 [`native_fused_roadmap.md`](native_fused_roadmap.md)。 |
| 验收标准变化 | [`reference/HF_CRITERIA.md`](reference/HF_CRITERIA.md)。 |
| 贡献者/归因变化 | [`../CONTRIBUTORS.md`](../CONTRIBUTORS.md), [`../CONTRIBUTIONS.md`](../CONTRIBUTIONS.md)。 |

## 当前主线摘要

- **HF 适配**:加载、生成、Trainer/PEFT/TRL/DeepSpeed、state cache、量化功能已有 smoke/矩阵证据;继续补更大模型、更多卡、更长训练和 production 证据。
- **性能主线**:`native_graph -> fused fp16 WKV/prefill/decode -> fused quant W8/W4`,不是继续堆 wrapper。
- **Apple 主线**:MLX/CoreML 作为移动端路线;当前 scan-prefill 有实测提升,但相对 Qwen3.5 仍有 prefill/decode gap;所有“超过”只能基于 same-device comparison gate。
- **量化主线**:功能和显存下降已证明;稳定端到端快过 fp16 仍依赖 fused quant kernel / projection 路线。
- **质量主线**:MATH500 通过固定 shape、seed、avg@64 和 logits/compression alignment 保证性能优化不偷换质量。
