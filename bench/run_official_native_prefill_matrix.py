#!/usr/bin/env python3
# coding=utf-8
"""Run isolated Native-vs-official FP16-state prefill comparison cases."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
COMPARE = REPO_ROOT / "scripts" / "compare_official_native_prefill.py"


def parse_cases(value: str) -> list[tuple[int, int]]:
    cases = []
    for item in value.replace(",", " ").split():
        batch, tokens = item.lower().split("x", 1)
        pair = (int(batch), int(tokens))
        if pair[0] <= 0 or pair[1] <= 0:
            raise ValueError(f"matrix shape must be positive: {item}")
        if pair not in cases:
            cases.append(pair)
    return cases


def run(command: list[str], log_path: Path) -> int:
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return int(completed.returncode)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--official-dir", required=True)
    ap.add_argument("--official-model", required=True)
    ap.add_argument("--official-source-manifest", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--cases", default="1x128,1x512,1x2048,8x128,8x512,8x2048")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--native-source-revision", default="working-tree")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for batch_size, prompt_tokens in parse_cases(args.cases):
        name = f"b{batch_size}_t{prompt_tokens}"
        native_capture = output_dir / f"{name}_native.pt"
        official_capture = output_dir / f"{name}_official.pt"
        report = output_dir / f"{name}_report.json"
        row = {
            "batch_size": batch_size,
            "prompt_tokens": prompt_tokens,
            "native_capture": native_capture.name,
            "official_capture": official_capture.name,
            "report": report.name,
        }

        native_code = 0
        if not (args.skip_existing and native_capture.exists()):
            native_code = run(
                [
                    sys.executable,
                    str(COMPARE),
                    "--mode",
                    "capture-native",
                    "--hf-dir",
                    args.hf_dir,
                    "--batch-size",
                    str(batch_size),
                    "--prompt-tokens",
                    str(prompt_tokens),
                    "--warmup",
                    str(args.warmup),
                    "--repeats",
                    str(args.repeats),
                    "--native-source-revision",
                    args.native_source_revision,
                    "--output",
                    str(native_capture),
                ],
                output_dir / f"{name}_native.log",
            )
        row["native_exit_code"] = native_code

        official_code = 0
        if native_capture.exists() and not (args.skip_existing and official_capture.exists()):
            official_code = run(
                [
                    sys.executable,
                    str(COMPARE),
                    "--mode",
                    "capture-official",
                    "--hf-dir",
                    args.hf_dir,
                    "--batch-size",
                    str(batch_size),
                    "--prompt-tokens",
                    str(prompt_tokens),
                    "--warmup",
                    str(args.warmup),
                    "--repeats",
                    str(args.repeats),
                    "--official-dir",
                    args.official_dir,
                    "--official-model",
                    args.official_model,
                    "--official-source-manifest",
                    args.official_source_manifest,
                    "--output",
                    str(official_capture),
                ],
                output_dir / f"{name}_official.log",
            )
        row["official_exit_code"] = official_code

        compare_code = 0
        if native_capture.exists() and official_capture.exists():
            if not (args.skip_existing and report.exists()):
                compare_code = run(
                    [
                        sys.executable,
                        str(COMPARE),
                        "--mode",
                        "compare",
                        "--native-capture",
                        str(native_capture),
                        "--official-capture",
                        str(official_capture),
                        "--output",
                        str(report),
                    ],
                    output_dir / f"{name}_compare.log",
                )
            if report.exists():
                comparison = json.loads(report.read_text(encoding="utf-8"))
                row.update(
                    {
                        "status": comparison.get("status"),
                        "quality_pass": comparison.get("quality_pass"),
                        "performance_gate_pass": comparison.get("performance_gate_pass"),
                        "native_over_official_tokps": comparison.get("native_over_official_tokps"),
                        "native_median_ms": comparison.get("native", {}).get("timing", {}).get("median_ms"),
                        "official_median_ms": comparison.get("official", {}).get("timing", {}).get("median_ms"),
                        "native_peak_vram_mb": comparison.get("native", {}).get("peak_vram_mb"),
                        "official_peak_vram_mb": comparison.get("official", {}).get("peak_vram_mb"),
                    }
                )
        else:
            compare_code = 2
        row["compare_exit_code"] = compare_code
        row.setdefault("status", "incomplete")
        rows.append(row)
        (output_dir / "summary.json").write_text(
            json.dumps({"axis": "official_native_prefill_matrix", "rows": rows}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(row), flush=True)

    complete = all(
        row.get("status") == "pass"
        and row.get("quality_pass") is True
        and row.get("performance_gate_pass") is True
        for row in rows
    )
    summary = {
        "axis": "official_native_prefill_matrix",
        "status": "pass" if complete else "incomplete",
        "cases": len(rows),
        "quality_pass_cases": sum(row.get("quality_pass") is True for row in rows),
        "performance_pass_cases": sum(row.get("performance_gate_pass") is True for row in rows),
        "rows": rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
