#!/usr/bin/env python3
"""Fail-closed summary for the canonical V100 evidence lanes."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Direct execution places ``bench/`` ahead of the repository root and can load
# the legacy ``bench.py`` module instead of the benchmark package.
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path = [entry for entry in sys.path if Path(entry or ".").resolve() != SCRIPT_DIR]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bench.compare_qwen35_speed_matrix import compare, load_rows


PRODUCTION_DIR = Path("bench/v100_production_close_20260711")
FULL_FLA_DIR = Path("bench/v100_active_b1b8_20260715")
TORCH_MATRIX_DIR = Path("bench/v100_qwen35_full_matrix_20260713")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} must contain a JSON object")
        rows.append(value)
    return rows


def _validate_production_close(repo_root: Path) -> tuple[dict[str, Any], list[str]]:
    directory = repo_root / PRODUCTION_DIR
    errors: list[str] = []
    try:
        summary = _load_json(directory / "summary.json")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {}, [f"production close summary is invalid: {exc}"]

    expected = {
        "status": "pass",
        "dense_decode_rows": 12,
        "dense_prefill_rows": 12,
        "quant_decode_rows": 24,
        "quant_prefill_rows": 24,
        "device_map_2gpu_status": "pass",
        "zero_resume_stages": [2, 3],
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            errors.append(f"production close {key} must be {value!r}")

    numeric_floors = {
        "dense_decode_ratio_min": 0.90,
        "dense_prefill_ratio_min": 0.90,
        "quant_decode_ratio_min": 1.0,
        "quant_prefill_ratio_min": 0.99,
    }
    for key, floor in numeric_floors.items():
        value = float(summary.get(key, 0.0))
        if value < floor:
            errors.append(f"production close {key}={value} is below {floor}")

    raw_files = {
        "dense_decode.jsonl": 12,
        "dense_prefill.jsonl": 12,
        "quant_decode_acceptance.jsonl": 24,
        "quant_prefill_acceptance.jsonl": 24,
        "training_regression.jsonl": 2,
        "zero2_resume_regression.jsonl": 2,
        "zero3_resume_regression.jsonl": 2,
        "device_map_2gpu.jsonl": 1,
    }
    for name, expected_rows in raw_files.items():
        try:
            rows = _load_jsonl(directory / name)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"production close {name} is invalid: {exc}")
            continue
        if len(rows) != expected_rows:
            errors.append(f"production close {name} has {len(rows)}/{expected_rows} rows")
        if any(row.get("status") not in (None, "pass") for row in rows):
            errors.append(f"production close {name} contains a non-pass row")

    return {
        "status": "pass" if not errors else "fail",
        "source": PRODUCTION_DIR.as_posix(),
        "scope": "0.1B/0.4B/1.5B dense, selected-module W8/W4, serving and training smokes",
        "dense_decode_ratio_min": summary.get("dense_decode_ratio_min"),
        "dense_prefill_ratio_min": summary.get("dense_prefill_ratio_min"),
        "quant_decode_ratio_min": summary.get("quant_decode_ratio_min"),
        "quant_prefill_ratio_min": summary.get("quant_prefill_ratio_min"),
        "quant_payload_ratio_range": [
            summary.get("quant_payload_ratio_min"),
            summary.get("quant_payload_ratio_max"),
        ],
        "chunk512_speed_ratio_vs_full": summary.get("chunk512_speed_ratio_vs_full"),
        "chunk512_vram_ratio_vs_full": summary.get("chunk512_vram_ratio_vs_full"),
        "training_backends": summary.get("training_backends"),
        "zero_resume_stages": summary.get("zero_resume_stages"),
        "boundary": (
            "The W8/W4 speed lane quantizes selected modules; it is not the "
            "unaccepted full-memory native MM8/MM4 path."
        ),
    }, errors


def _correctness_report(path: Path, minimum: float) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    try:
        report = _load_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {}, [f"invalid correctness report {path.name}: {exc}"]
    if report.get("status") != "pass" or report.get("greedy_tokens_match") is not True:
        errors.append(f"correctness report {path.name} did not pass greedy equality")
    for key in ("prompt_logits_cosine", "final_logits_cosine"):
        if float(report.get(key, 0.0)) < minimum:
            errors.append(f"correctness report {path.name} {key} is below {minimum}")
    return {
        "status": report.get("status"),
        "greedy_tokens": len(report.get("greedy_tokens") or []),
        "prompt_logits_cosine": report.get("prompt_logits_cosine"),
        "final_logits_cosine": report.get("final_logits_cosine"),
    }, errors


def _validate_full_fla(repo_root: Path) -> tuple[dict[str, Any], list[str]]:
    directory = repo_root / FULL_FLA_DIR
    errors: list[str] = []
    try:
        rows = load_rows(directory / "results_dense.jsonl")
        summary = compare(
            rows,
            expected_cells=2,
            min_prefill_speedup=1.0,
            min_decode_speedup=1.0,
            require_qwen_fast_path=True,
            require_qwen_full_fused=True,
            required_reference_backend="fla",
            min_prefill_active_parameter_throughput_ratio=1.0,
            min_decode_active_parameter_throughput_ratio=1.0,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {}, [f"full-FLA matrix is invalid: {exc}"]

    if not (summary.get("gates") or {}).get("overall_pass"):
        errors.append("full-FLA matrix no longer passes the current comparator")
    roles = {role: [row for row in rows if row.get("model_role") == role] for role in ("candidate", "reference")}
    if len(roles["candidate"]) != 2 or len(roles["reference"]) != 2:
        errors.append("full-FLA matrix must contain two candidate and two reference rows")
    if any("V100" not in str(row.get("device")) for row in rows):
        errors.append("full-FLA matrix contains a non-V100 row")

    qwen_correctness, qwen_errors = _correctness_report(
        directory / "qwen-full-fla-vs-oracle.json", 0.999
    )
    rwkv_correctness, rwkv_errors = _correctness_report(
        directory / "rwkv-native-graph-vs-fla.json", 0.9999
    )
    errors.extend(qwen_errors)
    errors.extend(rwkv_errors)

    return {
        "status": "pass" if not errors else "fail",
        "source": FULL_FLA_DIR.as_posix(),
        "scope": {
            "model_pair": "RWKV-7 1.5B / Qwen3.5-2B",
            "batch_sizes": [1, 8],
            "prompt_tokens": 512,
            "decode_tokens": 64,
            "dtype": "fp16",
            "quantization": "none",
        },
        "coverage": summary.get("coverage"),
        "reference_backend": summary.get("reference_backend"),
        "speed": summary.get("speed"),
        "active_parameter_work": summary.get("active_parameter_work"),
        "memory": summary.get("memory"),
        "correctness": {
            "qwen_full_fla_vs_conv_oracle": qwen_correctness,
            "rwkv_native_graph_vs_fla": rwkv_correctness,
        },
        "boundary": (
            "This is the only promoted V100 full-FLA Qwen lane. It does not "
            "cover larger pairs, other lengths, quantization, or model quality."
        ),
    }, errors


def _validate_torch_matrix(repo_root: Path) -> tuple[dict[str, Any], list[str]]:
    directory = repo_root / TORCH_MATRIX_DIR
    errors: list[str] = []
    try:
        rows = load_rows(directory / "results.jsonl")
        summary = compare(
            rows,
            expected_cells=216,
            min_prefill_speedup=1.05,
            min_decode_speedup=1.05,
            min_quant_prefill_speedup=1.0,
            min_quant_decode_speedup=1.0,
            required_reference_backend="torch",
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {}, [f"torch-fallback matrix is invalid: {exc}"]

    if not (summary.get("gates") or {}).get("overall_pass"):
        errors.append("torch-fallback matrix no longer passes its explicit historical gate")
    references = [row for row in rows if row.get("model_role") == "reference"]
    candidates = [row for row in rows if row.get("model_role") == "candidate"]
    if len(references) != 216 or len(candidates) != 219:
        errors.append(
            f"torch-fallback matrix raw role counts are candidate={len(candidates)}, reference={len(references)}"
        )
    if any(
        row.get("qwen_backend_requested") != "torch"
        or row.get("effective_backend") != "transformers_torch_fallback"
        or row.get("qwen_force_torch") is not True
        for row in references
    ):
        errors.append("torch-fallback matrix contains a reference row from another backend")

    return {
        "status": "pass" if not errors else "fail",
        "role": "historical diagnostic",
        "source": TORCH_MATRIX_DIR.as_posix(),
        "scope": "3 pairs x 3 prompts x 2 decode lengths x B1/B2/B4/B8 x fp16/W8/W4",
        "raw_rows": len(rows),
        "coverage": summary.get("coverage"),
        "reference_backend": summary.get("reference_backend"),
        "speed": summary.get("speed"),
        "memory": summary.get("memory"),
        "boundary": (
            "All 216 Qwen references are Transformers torch fallback. This "
            "matrix is not evidence against optimized FLA Qwen. Memory was "
            "not a gate and is not lower in every final rerun cell."
        ),
    }, errors


def summarize(repo_root: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    production, production_errors = _validate_production_close(repo_root)
    full_fla, full_fla_errors = _validate_full_fla(repo_root)
    torch_matrix, torch_errors = _validate_torch_matrix(repo_root)
    errors = production_errors + full_fla_errors + torch_errors
    return {
        "axis": "v100_acceptance_manifest",
        "date": "2026-07-16",
        "validation_status": "pass" if not errors else "fail",
        "project_status": "partial_gpu_followups_required",
        "device": "Tesla V100-PCIE-32GB (sm_70)",
        "lanes": {
            "production_close": production,
            "qwen_full_fla": full_fla,
            "qwen_torch_fallback": torch_matrix,
        },
        "open_gpu_gates": [
            {
                "id": "qwen_full_fla_expansion",
                "status": "requires_v100",
                "detail": "Expand beyond 1.5B/2B, prompt512/decode64 and B1/B8.",
            },
            {
                "id": "full_memory_native_quant",
                "status": "requires_v100_and_kernel_work",
                "detail": (
                    "Draft PR #21 records MM4 speed with greedy mismatches and MM8 at "
                    "0/21 speed cells per model; no full-memory path is promoted."
                ),
            },
            {
                "id": "large_training_and_zero",
                "status": "requires_v100_or_larger_gpu",
                "detail": "Longer training and ZeRO resume beyond the promoted smoke/model sizes remain open.",
            },
        ],
        "not_claimed": [
            "model-quality superiority over Qwen3.5",
            "full-FLA coverage for the historical 216-cell torch matrix",
            "lower peak VRAM in every full-FLA or historical cell",
            "accepted full-memory native MM8/MM4 on Volta",
            "universal Albatross P2/P3 parity",
        ],
        "errors": errors,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lanes = report["lanes"]
    production = lanes["production_close"]
    full_fla = lanes["qwen_full_fla"]
    torch_matrix = lanes["qwen_torch_fallback"]
    speed = full_fla.get("speed") or {}
    active = full_fla.get("active_parameter_work") or {}
    memory = full_fla.get("memory") or {}
    torch_speed = torch_matrix.get("speed") or {}
    lines = [
        "# V100 acceptance summary",
        "",
        f"Evidence validation: **{str(report['validation_status']).upper()}**.",
        "",
        "| Lane | Coverage | Result | Boundary |",
        "|---|---:|---|---|",
        (
            "| Production close | 12 dense decode + 12 dense prefill + 48 selected-module quant rows "
            f"| Albatross minima `{production.get('dense_decode_ratio_min')}x` decode / "
            f"`{production.get('dense_prefill_ratio_min')}x` prefill | Selected-module W8/W4, not full-memory quant |"
        ),
        (
            f"| Full-FLA Qwen | {(full_fla.get('coverage') or {}).get('joined_cells')}/2 cells "
            f"| raw prefill/decode min `{speed.get('min_prefill_speedup')}x/"
            f"{speed.get('min_decode_speedup')}x`; active-work min "
            f"`{active.get('min_prefill_throughput_ratio')}x/"
            f"{active.get('min_decode_throughput_ratio')}x` | Only 1.5B/2B, P512/D64, B1/B8, dense fp16 |"
        ),
        (
            f"| Torch-fallback Qwen diagnostic | {(torch_matrix.get('coverage') or {}).get('joined_cells')}/216 cells "
            f"| prefill/decode min `{torch_speed.get('min_prefill_speedup')}x/"
            f"{torch_speed.get('min_decode_speedup')}x` | Not an optimized-Qwen comparison |"
        ),
        "",
        "## Disclosed boundaries",
        "",
        f"- Full-FLA B1 peak VRAM ratio is `{memory.get('max_peak_vram_ratio')}x`; only 1/2 cells use no more peak VRAM.",
        "- The historical 216-cell matrix is pinned to `--required-reference-backend torch`.",
        "- The production W8/W4 speed lane quantizes selected modules. Full-memory native MM8/MM4 remains open.",
        "- Inference speed does not establish instruction, reasoning, math, code or multilingual quality.",
        "",
        "## GPU-required follow-ups",
        "",
    ]
    for item in report["open_gpu_gates"]:
        lines.append(f"- `{item['id']}`: {item['detail']}")
    if report["errors"]:
        lines.extend(["", "## Validation errors", ""])
        lines.extend(f"- {error}" for error in report["errors"])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()
    report = summarize(args.repo_root)
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(encoded, encoding="utf-8", newline="\n")
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_markdown(report), encoding="utf-8", newline="\n")
    print(encoded, end="")
    return 0 if report["validation_status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
