#!/usr/bin/env python3
"""Run the restartable exact-V100 MM4 production acceptance matrix."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


CELLS = (
    (1, 128, 128),
    (2, 128, 128),
    (4, 128, 128),
    (8, 128, 128),
    (1, 512, 128),
    (1, 2048, 128),
    (1, 128, 512),
)

# Full-memory MM4 is the default acceptance lane. Exact cells may use the
# smaller speed/balanced policy only after a paired fp16 row closes every production
# gate; keeping this table explicit prevents accidental policy promotion.
CELL_POLICY_OVERRIDES = {
    ("2.9b", 4, 128, 128): "speed",
}


def parse_model(raw: str) -> tuple[str, str]:
    label, separator, path = raw.partition("=")
    if not separator or not label or not path:
        raise argparse.ArgumentTypeError("model must be LABEL=/absolute/hf/path")
    return label, path


def read_row(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    return json.loads(lines[-1]) if lines else None


def cell_slug(label: str, batch: int, prompt: int, decode: int) -> str:
    return f"{label}_b{batch}_p{prompt}_d{decode}"


def cell_policy(label: str, batch: int, prompt: int, decode: int) -> str:
    return CELL_POLICY_OVERRIDES.get((label, batch, prompt, decode), "memory")


def row_matches_configuration(
    row: dict[str, object],
    policy: str,
    group_size: int,
    group_policy: str,
    fused_epilogue: bool,
) -> bool:
    return (
        row.get("native_mm_policy") == policy
        and int(row.get("native_mm4_group_size") or 0) == int(group_size)
        and row.get("native_mm4_group_policy") == group_policy
        and row.get("sm70_w4_fused_epilogue") is fused_epilogue
    )


def acceptance_failures(row: dict[str, object]) -> list[str]:
    failures = []
    if row.get("sm70_extension_build_error"):
        failures.append("sm70_extension")
    if float(row.get("decode_speed_ratio_vs_fp16") or 0.0) < 1.0:
        failures.append("decode")
    if float(row.get("footprint_ratio_vs_fp16") or 1.0) >= 1.0:
        failures.append("footprint")
    if float(row.get("final_logits_cos_vs_fp16") or 0.0) < 0.998:
        failures.append("logits")
    if row.get("same_greedy_tokens_as_fp16") is not True:
        failures.append("greedy")
    if row.get("greedy_repeat_deterministic") is not True:
        failures.append("repeat_determinism")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", type=parse_model, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--min-params", type=int, default=8_000_000)
    parser.add_argument("--group-size", type=int, choices=(128, 256), default=128)
    parser.add_argument("--group-policy", default="lm_head", choices=("lm_head",))
    parser.add_argument(
        "--policy",
        choices=("auto", "memory", "speed", "balanced"),
        default="memory",
        help=(
            "MM4 deployment policy. memory, speed, or balanced validates one configuration "
            "across the whole matrix; auto is an explicit per-workload evidence "
            "route and starts a fresh model process for every cell."
        ),
    )
    parser.add_argument(
        "--fused-epilogue",
        choices=("false", "true"),
        default="false",
        help="Enable the exact-sm70 rowwise FFN ReLU2/residual epilogues.",
    )
    args = parser.parse_args()
    if args.timing_repeats < 1 or args.warmup < 0:
        parser.error("timing repeats must be positive and warmup non-negative")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fused_epilogue = args.fused_epilogue == "true"
    worker = Path(__file__).with_name("bench_native_quant_e2e_decode.py")
    failures = 0
    rows = []
    for label, model_path in args.model:
        for batch, prompt, decode in CELLS:
            slug = cell_slug(label, batch, prompt, decode)
            policy = (
                cell_policy(label, batch, prompt, decode)
                if args.policy == "auto"
                else args.policy
            )
            result_path = args.output_dir / f"{slug}.jsonl"
            log_path = args.output_dir / f"{slug}.log"
            row = read_row(result_path)
            if (
                row is None
                or row.get("status") != "pass"
                or not row_matches_configuration(
                    row,
                    policy,
                    args.group_size,
                    args.group_policy,
                    fused_epilogue,
                )
            ):
                command = [
                    sys.executable,
                    str(worker),
                    "--hf-dir",
                    model_path,
                    "--code-source",
                    "repo",
                    "--model-size-label",
                    label,
                    "--dtype",
                    "fp16",
                    "--device",
                    "cuda",
                    "--attn-mode",
                    "fused_recurrent",
                    "--fast-token-backend",
                    "native_graph",
                    "--single-quantization",
                    "mm4",
                    "--paired-baseline",
                    "--min-params",
                    str(args.min_params),
                    "--mm4-group-size",
                    str(args.group_size),
                    "--mm4-group-policy",
                    args.group_policy,
                    "--policy",
                    policy,
                    "--batch-size",
                    str(batch),
                    "--prompt-tokens",
                    str(prompt),
                    "--decode-tokens",
                    str(decode),
                    "--warmup",
                    str(args.warmup),
                    "--timing-repeats",
                    str(args.timing_repeats),
                    "--results",
                    str(result_path),
                ]
                env = os.environ.copy()
                env.setdefault("RWKV_V7_ON", "1")
                env.setdefault("RWKV7_FAST_TOKEN_BACKEND", "native_graph")
                env["RWKV7_SM70_W4_FUSED_EPILOGUE"] = (
                    "1" if fused_epilogue else "0"
                )
                with log_path.open("w", encoding="utf-8") as log:
                    completed = subprocess.run(
                        command,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=False,
                    )
                row = read_row(result_path)
                if completed.returncode or row is None or row.get("status") != "pass":
                    failures += 1
                    print(f"FAIL {slug}: exit={completed.returncode} log={log_path}")
                    continue
            rows.append(row)
            gate_failures = acceptance_failures(row)
            outcome = "PASS" if not gate_failures else "GATE-FAIL"
            print(
                f"{outcome} {slug}: decode={row.get('decode_speed_ratio_vs_fp16')}x "
                f"footprint={row.get('footprint_ratio_vs_fp16')}x "
                f"cos={row.get('final_logits_cos_vs_fp16')} "
                f"greedy={row.get('same_greedy_tokens_as_fp16')} "
                f"policy={policy} "
                f"failed={','.join(gate_failures) or 'none'}",
                flush=True,
            )

    aggregate = args.output_dir / "results.jsonl"
    aggregate.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary = {
        "expected": len(args.model) * len(CELLS),
        "completed": len(rows),
        "failures": failures,
        "decode_speed_pass": sum(
            float(row.get("decode_speed_ratio_vs_fp16") or 0.0) >= 1.0
            for row in rows
        ),
        "footprint_pass": sum(
            float(row.get("footprint_ratio_vs_fp16") or 1.0) < 1.0 for row in rows
        ),
        "logits_pass": sum(
            float(row.get("final_logits_cos_vs_fp16") or 0.0) >= 0.998
            for row in rows
        ),
        "greedy_pass": sum(
            row.get("same_greedy_tokens_as_fp16") is True for row in rows
        ),
        "repeat_determinism_pass": sum(
            row.get("greedy_repeat_deterministic") is True for row in rows
        ),
        "memory_policy_cells": sum(
            row.get("native_mm_policy") == "memory" for row in rows
        ),
        "speed_policy_cells": sum(
            row.get("native_mm_policy") == "speed" for row in rows
        ),
        "balanced_policy_cells": sum(
            row.get("native_mm_policy") == "balanced" for row in rows
        ),
        "requested_policy": args.policy,
        "fused_epilogue": fused_epilogue,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, sort_keys=True), flush=True)
    gates_pass = all(
        int(summary[name]) == int(summary["expected"])
        for name in (
            "decode_speed_pass",
            "footprint_pass",
            "logits_pass",
            "greedy_pass",
            "repeat_determinism_pass",
        )
    )
    return 0 if not failures and len(rows) == summary["expected"] and gates_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
