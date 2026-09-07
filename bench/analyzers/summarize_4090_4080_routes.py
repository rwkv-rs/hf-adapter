#!/usr/bin/env python3
"""Gate the exact RTX 4090 routes reproduced from the RTX 4080 work."""
from __future__ import annotations

# Support direct ``python bench/<category>/<script>.py`` execution.
if __package__ in {None, ""}:
    import sys as _sys
    from pathlib import Path as _Path

    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any


DEVICE = "NVIDIA GeForce RTX 4090"
MODELS = {
    (1024, 24): "0.4b",
    (2048, 24): "1.5b",
    (2560, 32): "2.9b",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def rounded(value: float) -> float:
    return round(float(value), 6)


def model_label(row: dict[str, Any]) -> str | None:
    reported = str(row.get("model_size_label", ""))
    if reported in MODELS.values():
        return reported
    return MODELS.get(
        (int(row.get("hidden_size", -1)), int(row.get("num_hidden_layers", -1)))
    )


def summarize(
    bmm_path: Path,
    accum_path: Path,
    policy_prefill_path: Path,
    *,
    min_cosine: float = 0.9999,
) -> dict[str, Any]:
    bmm = load_jsonl(bmm_path)
    accum = load_jsonl(accum_path)
    policy = load_jsonl(policy_prefill_path)
    errors: list[str] = []

    for label, rows, expected in (
        ("BMM", bmm, 9),
        ("accumulation", accum, 108),
        ("default-policy Prefill", policy, 18),
    ):
        if len(rows) != expected:
            errors.append(f"{label} has {len(rows)} rows, expected {expected}")
        bad_devices = Counter(row.get("device") for row in rows if row.get("device") != DEVICE)
        if bad_devices:
            errors.append(f"{label} contains non-4090 devices: {dict(bad_devices)}")

    bmm_groups: list[dict[str, Any]] = []
    for shape, name in MODELS.items():
        rows = [row for row in bmm if model_label(row) == name]
        if len(rows) != 3:
            errors.append(f"BMM {name} has {len(rows)} rows, expected 3")
            continue
        passed = all(
            row.get("status") == "pass"
            and row.get("correctness_pass") is True
            and int(row.get("batch_size", -1)) == 8
            and int(row.get("greedy_match", -1)) == int(row.get("greedy_total", -2))
            and float(row.get("min_cosine_first_step", 0.0)) >= min_cosine
            and float(row.get("speedup", 0.0)) > 1.0
            for row in rows
        )
        if not passed:
            errors.append(f"BMM {name} failed speed/correctness gate")
        speedups = [float(row["speedup"]) for row in rows]
        bmm_groups.append(
            {
                "model": name,
                "hidden_size": shape[0],
                "runs": 3,
                "speedup_median": rounded(median(speedups)),
                "speedup_min": rounded(min(speedups)),
                "speedup_max": rounded(max(speedups)),
                "vram_delta_mb": rounded(median(float(row["vram_delta_mb"]) for row in rows)),
                "greedy_match": sum(int(row["greedy_match"]) for row in rows),
                "greedy_total": sum(int(row["greedy_total"]) for row in rows),
                "minimum_cosine": rounded(min(float(row["min_cosine_first_step"]) for row in rows)),
                "pass": passed,
            }
        )

    accum_by_key: dict[tuple[str, int, int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in accum:
        name = model_label(row)
        if name is None:
            errors.append("accumulation row has an unknown model shape")
            continue
        accum_by_key[
            (
                name,
                int(row.get("batch_size", -1)),
                int(row.get("prompt_tokens", -1)),
                int(row.get("order_index", -1)),
                str(row.get("mode")),
            )
        ].append(row)

    accum_shapes: list[dict[str, Any]] = []
    for name in MODELS.values():
        for batch in (1, 8):
            for prompt in (128, 512, 2048):
                rows = [
                    accum_by_key.get((name, batch, prompt, order, mode), [])
                    for order in (0, 1)
                    for mode in ("off", "global", "block")
                ]
                if any(len(group) != 1 for group in rows):
                    errors.append(f"accumulation {name} B{batch} P{prompt} coverage drifted")
                    continue
                flat = [group[0] for group in rows]
                block = [row for row in flat if row["mode"] == "block"]
                passed = all(
                    row.get("status") == "pass"
                    and row.get("route_effective_match") is True
                    and row.get("prompt_greedy_match") is True
                    and row.get("decode_greedy_match") is True
                    for row in flat
                ) and all(
                    float(row["speedup_vs_off"]) > 1.0
                    and float(row["prompt_min_cosine"]) >= min_cosine
                    and float(row["decode_min_cosine"]) >= min_cosine
                    for row in block
                )
                if not passed:
                    errors.append(f"accumulation {name} B{batch} P{prompt} failed")
                accum_shapes.append(
                    {
                        "model": name,
                        "batch_size": batch,
                        "prompt_tokens": prompt,
                        "block_speedup_median": rounded(
                            median(float(row["speedup_vs_off"]) for row in block)
                        ),
                        "block_speedup_min": rounded(
                            min(float(row["speedup_vs_off"]) for row in block)
                        ),
                        "minimum_prompt_cosine": rounded(
                            min(float(row["prompt_min_cosine"]) for row in block)
                        ),
                        "minimum_decode_cosine": rounded(
                            min(float(row["decode_min_cosine"]) for row in block)
                        ),
                        "pass": passed,
                    }
                )

    expected_policy = {
        (name, batch, prompt)
        for name in MODELS.values()
        for batch in (1, 8)
        for prompt in (128, 512, 2048)
    }
    actual_policy = {
        (model_label(row), int(row.get("batch_size", -1)), int(row.get("prompt_tokens", -1)))
        for row in policy
    }
    if actual_policy != expected_policy:
        errors.append("default-policy Prefill coverage drifted")
    policy_passed = all(
        row.get("status") == "pass"
        and row.get("prefill_block_fp16_accum_effective") is True
        and row.get("prefill_global_fp16_accum_effective") is False
        and row.get("greedy_match") is True
        and row.get("decode_after_prefill_greedy_match") is True
        and float(row.get("min_cosine", 0.0)) >= min_cosine
        and float(row.get("decode_after_prefill_min_cosine", 0.0)) >= min_cosine
        for row in policy
    )
    if not policy_passed:
        errors.append("default-policy Prefill route/correctness gate failed")

    return {
        "axis": "rtx4090_rtx4080_route_transfer",
        "status": "pass" if not errors else "fail",
        "device": DEVICE,
        "min_cosine_gate": min_cosine,
        "bmm": {
            "rows": len(bmm),
            "groups": bmm_groups,
            "pass": all(group["pass"] for group in bmm_groups) and len(bmm_groups) == 3,
        },
        "block_fp16_accumulation": {
            "rows": len(accum),
            "shapes": accum_shapes,
            "pass": all(row["pass"] for row in accum_shapes) and len(accum_shapes) == 18,
        },
        "default_policy_prefill": {
            "rows": len(policy),
            "pass": policy_passed and actual_policy == expected_policy,
            "minimum_prefill_cosine": rounded(
                min((float(row.get("min_cosine", 0.0)) for row in policy), default=0.0)
            ),
            "minimum_decode_cosine": rounded(
                min(
                    (float(row.get("decode_after_prefill_min_cosine", 0.0)) for row in policy),
                    default=0.0,
                )
            ),
        },
        "errors": errors,
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# RTX 4090 reproduction of RTX 4080 routes",
        "",
        f"Status: **{report['status']}**",
        "",
        "## B8 grouped W/A/V BMM",
        "",
        "| Model | Median speedup | Range | VRAM delta | Greedy |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in report["bmm"]["groups"]:
        lines.append(
            f"| {row['model']} | **{row['speedup_median']:.4f}x** | "
            f"{row['speedup_min']:.4f}x-{row['speedup_max']:.4f}x | "
            f"{row['vram_delta_mb']:+.1f} MiB | "
            f"{row['greedy_match']}/{row['greedy_total']} |"
        )
    shapes = report["block_fp16_accumulation"]["shapes"]
    lines.extend(
        [
            "",
            "## Block-scoped FP16 Prefill accumulation",
            "",
            f"All {len(shapes)} exact shapes pass. Minimum per-order speedup: "
            f"**{min(row['block_speedup_min'] for row in shapes):.4f}x**; "
            f"largest shape median: **{max(row['block_speedup_median'] for row in shapes):.4f}x**.",
            "",
            "## Default policy",
            "",
            f"All {report['default_policy_prefill']['rows']} exact Prefill rows: "
            f"**{'PASS' if report['default_policy_prefill']['pass'] else 'FAIL'}**.",
        ]
    )
    if report["errors"]:
        lines.extend(["", "## Errors", "", *[f"- {error}" for error in report["errors"]]])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bmm", type=Path)
    parser.add_argument("accum", type=Path)
    parser.add_argument("policy_prefill", type=Path)
    parser.add_argument("--min-cosine", type=float, default=0.9999)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()
    report = summarize(
        args.bmm,
        args.accum,
        args.policy_prefill,
        min_cosine=args.min_cosine,
    )
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    if args.markdown_output:
        args.markdown_output.write_text(markdown(report), encoding="utf-8")
    print(text, end="")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
