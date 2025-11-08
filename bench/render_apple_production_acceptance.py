#!/usr/bin/env python3
"""Render the complete Apple production gate manifest as a Markdown report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .check_apple_production_acceptance import DEFAULT_MANIFEST, audit
except ImportError:  # Direct `python bench/render_...py` execution.
    from check_apple_production_acceptance import DEFAULT_MANIFEST, audit


def _cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def render(manifest: dict[str, object], *, root: Path) -> str:
    rows = audit(manifest, root=root)
    by_id = {row["gate_id"]: row for row in rows}
    required = [row for row in rows if row["required"]]
    passed = sum(row["status"] == "pass" for row in required)
    complete = bool(required) and passed == len(required)
    categories = {item["id"]: item["title"] for item in manifest.get("categories", [])}
    symbols = {"pass": "✅ PASS", "fail": "❌ FAIL", "missing": "⬜ MISSING", "unknown": "❓ UNKNOWN"}

    out = [
        "# Apple Silicon 生产级硬门清单",
        "",
        "> **结论：%s。** 当前通过 **%d / %d** 个必选硬门。任何 `FAIL`、`MISSING` 或 `UNKNOWN` 都禁止声明 Apple 生产级完成。"
        % ("全部通过" if complete else "尚未达到生产级", passed, len(required)),
        "",
        "本清单覆盖 RWKV-7 Hugging Face 适配中的 Apple 路线：HF/MPS 兼容与训练、MLX/Metal 生产推理、CoreML/ANE 部署。"
        "CUDA、vLLM、SGLang、DeepSpeed ZeRO 属于其他验收路线，不用 Apple 兼容结果替代。",
        "",
        "## 完成规则",
        "",
        "1. 只有真实机器、真实 checkpoint、可复现命令和 JSONL 数据才算证据；源码里存在某个函数不算通过。",
        "2. 冒烟、单模型、单 bsz、单芯片结果只关闭对应原子门，不能替代完整矩阵。",
        "3. 性能通过必须在同一配置同时通过正确性/质量门，并记录冷/热、真实峰值内存、重复次数和精确硬件。",
        "4. W8/W4 必须实际降低峰值内存，并在所有声明支持的 Apple 卡型/模型/bsz 上不慢于 W16；否则只是功能完成。",
        "5. CoreML 只有实际运行证据；设置 `compute_units` 不能替代 ANE 落核证明。",
        "6. 运行严格审计：`STRICT=1 scripts/run_apple_production_acceptance.sh`。任一硬门未通过时命令必须非零退出。",
        "",
        "## 分类状态",
        "",
        "| 分类 | PASS | 未完成 | 总数 |",
        "|---|---:|---:|---:|",
    ]
    for category_id, title in categories.items():
        subset = [row for row in required if row["category"] == category_id]
        category_pass = sum(row["status"] == "pass" for row in subset)
        out.append(f"| {_cell(title)} (`{category_id}`) | {category_pass} | {len(subset) - category_pass} | {len(subset)} |")

    out.extend(["", "## 全部硬门", ""])
    manifest_gates = manifest.get("gates", [])
    for category_id, title in categories.items():
        out.extend(
            [
                f"### {title}",
                "",
                "| 状态 | Gate ID | 验收点 | 硬判据 | 当前证据 |",
                "|---|---|---|---|---|",
            ]
        )
        for gate in manifest_gates:
            if gate["category"] != category_id:
                continue
            result = by_id[gate["id"]]
            out.append(
                "| %s | `%s` | %s | %s | %s |"
                % (
                    symbols[result["status"]],
                    _cell(gate["id"]),
                    _cell(gate["title"]),
                    _cell(gate["criterion"]),
                    _cell(result["reason"]),
                )
            )
        out.append("")

    out.extend(
        [
            "## 当前实施顺序",
            "",
            "1. **MLX 自动策略与真实动态批**：按芯片/模型/dtype/quant/bsz 安全选择 DPLR、norm、compiled decode，并补 bsz 1/2/4/8。",
            "2. **W8/W4 性能闭环**：packed Metal fused kernel；所有矩阵内存下降且 prefill/decode 不慢于 W16。",
            "3. **Qwen3.5 全矩阵**：补真实 Qwen 峰值内存、1k/4k/8k、0.8B/2B/4B/9B 配对和质量。",
            "4. **CoreML/ANE**：真实 export/runtime、HF/state/chunk parity、INT8/INT4、Instruments/compute-plan 落核证据。",
            "5. **训练与可靠性**：1000 步 Trainer/SFT/DPO/GRPO、断点续训、24h/10k 请求、泄漏/OOM/热稳态。",
            "6. **跨设备**：M1/M2/M3/M4/M5、8GB/16GB/32GB+ 和 iPhone/iPad ANE 真机矩阵。",
            "",
            "## 证据文件",
            "",
            "- 机器可读清单：`bench/apple_production_gates.json`",
            "- 严格审计器：`bench/check_apple_production_acceptance.py`",
            "- 一键入口：`scripts/run_apple_production_acceptance.sh`",
            "- 审计输出：`bench/results_apple_production_acceptance.jsonl`（默认 append-only）",
            "",
            "本文件由 `bench/render_apple_production_acceptance.py` 从 manifest 和已提交证据生成；新增或修改硬门后必须重新生成。",
            "",
        ]
    )
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output", default="docs/hardware/APPLE_PRODUCTION_ACCEPTANCE.md")
    parser.add_argument("--root", default="")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = Path(args.root).resolve() if args.root else (manifest_path.parents[1] if manifest_path.parent.name == "bench" else manifest_path.parent)
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render(manifest, root=root), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
