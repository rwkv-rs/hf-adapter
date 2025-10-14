#!/usr/bin/env python3
# coding=utf-8
"""Audit RWKV-7 Apple/Qwen3.5 goal coverage from JSONL evidence.

This is intentionally stricter than the pairwise speed comparator.  The active
Apple/mobile goal is not only "one row was faster"; it needs coverage across
public Qwen3.5 size classes, same-prompt MLX rows, quantized RWKV rows, state
cache/chunked-prefill evidence, CoreML/ANE export/runtime status, and quality
rows.  This script turns a result JSONL into explicit audit rows so missing
items remain visible and cannot be summarized as completed work by accident.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
# When this file is executed as ``python bench/audit_qwen35_apple_goal.py``,
# Python puts ``bench/`` itself on sys.path.  That directory also contains
# ``bench.py``, which would shadow the namespace package needed for
# ``bench.compare_qwen35_apple_baseline`` imports.
sys.path = [entry for entry in sys.path if Path(entry or ".").resolve() != SCRIPT_DIR]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bench.compare_qwen35_apple_baseline import BASELINE_AXIS, COMPARISON_AXIS, memory_bytes, requested_tokens, row_model_id, safe_float, safe_int
from bench.score_qwen35_quality import COMPARISON_AXIS as QUALITY_COMPARISON_AXIS

AUDIT_AXIS = "qwen35_apple_goal_audit"
SUMMARY_AXIS = "qwen35_apple_goal_audit_summary"
COREML_EXPORT_AXIS = "rwkv7_coreml_export"
COREML_PLAN_AXIS = "rwkv7_coreml_runtime_plan"

DEFAULT_TIERS = [
    "qwen3.5:0.8b-mlx|mlx-community/Qwen3.5-0.8B-MLX-4bit=rwkv7-g1d-0.4b-hf",
    "qwen3.5:2b-mlx|mlx-community/Qwen3.5-2B-MLX-4bit=rwkv7-g1g-1.5b-hf",
    "qwen3.5:4b-mlx|mlx-community/Qwen3.5-4B-MLX-4bit=rwkv7-g1g-2.9b-hf",
    "qwen3.5:9b-mlx|mlx-community/Qwen3.5-9B-MLX-4bit=rwkv7-g1g-7.2b-hf|rwkv7-g1g-13.3b-hf",
]
DEFAULT_REQUIRED_SHAPES = ["chars1024:128", "chars1024:512", "chars4096:128", "chars4096:512"]
GOAL_METRICS = ("decode", "prefill", "ttft", "memory", "quant", "state_cache", "comparison", "quality", "coreml")
PASS_STATUSES = {"pass"}
NON_PASS_STATUSES = {"missing", "unknown", "fail", "prototype"}


@dataclass(frozen=True)
class Tier:
    qwen_aliases: tuple[str, ...]
    rwkv_aliases: tuple[str, ...]

    @property
    def qwen_label(self) -> str:
        return self.qwen_aliases[0]

    @property
    def rwkv_label(self) -> str:
        return self.rwkv_aliases[0]

    @property
    def tier_id(self) -> str:
        return f"{self.qwen_label}={self.rwkv_label}"


@dataclass(frozen=True)
class Shape:
    prompt_case: str
    requested_generated_tokens: int

    @property
    def label(self) -> str:
        return f"{self.prompt_case}:{self.requested_generated_tokens}"


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected object row")
            rows.append(row)
    return rows


def load_evidence(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            for child in sorted(path.glob("*.jsonl")):
                rows.extend(load_jsonl(child))
        else:
            rows.extend(load_jsonl(path))
    return rows


def append_jsonl(path: str | Path | None, rows: Iterable[dict[str, Any]]) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_tier(raw: str) -> Tier:
    if "=" not in raw:
        raise ValueError(f"--tier must use qwen_alias[|...]=rwkv_alias[|...], got {raw!r}")
    left, right = raw.split("=", 1)
    qwen = tuple(item.strip() for item in left.split("|") if item.strip())
    rwkv = tuple(item.strip() for item in right.split("|") if item.strip())
    if not qwen or not rwkv:
        raise ValueError(f"invalid --tier {raw!r}")
    return Tier(qwen_aliases=qwen, rwkv_aliases=rwkv)


def parse_shape(raw: str) -> Shape:
    if ":" not in raw:
        raise ValueError(f"--required-shape must use charsN:TOKENS or N:TOKENS, got {raw!r}")
    left, right = raw.split(":", 1)
    left = left.strip()
    if left.isdigit():
        left = f"chars{left}"
    tokens = int(right.strip())
    if not left or tokens <= 0:
        raise ValueError(f"invalid --required-shape {raw!r}")
    return Shape(prompt_case=left, requested_generated_tokens=tokens)


def normalized_model_id(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", Path(str(value or "")).name.lower())


def model_matches(value: Any, aliases: Iterable[str]) -> bool:
    text = str(value or "")
    if not text:
        return False
    base = Path(text).name
    norm_text = normalized_model_id(text)
    for alias in aliases:
        if text == alias or base == alias:
            return True
        if norm_text and norm_text == normalized_model_id(alias):
            return True
    return False


def row_matches_model(row: dict[str, Any], aliases: Iterable[str]) -> bool:
    values = [row_model_id(row), row.get("model_path"), row.get("coreml_package"), row.get("manifest")]
    return any(model_matches(value, aliases) for value in values)


def row_shape(row: dict[str, Any]) -> Shape | None:
    prompt_case = str(row.get("prompt_case") or "")
    tokens = requested_tokens(row)
    if not prompt_case or tokens is None:
        return None
    return Shape(prompt_case=prompt_case, requested_generated_tokens=int(tokens))


def pass_baseline_rows(rows: list[dict[str, Any]], aliases: Iterable[str], shape: Shape | None = None, *, runtime: str | None = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if row.get("axis") != BASELINE_AXIS or row.get("status") != "pass":
            continue
        if runtime is not None and row.get("runtime") != runtime:
            continue
        if not row_matches_model(row, aliases):
            continue
        if shape is not None and row_shape(row) != shape:
            continue
        out.append(row)
    return out


def has_metric(rows: list[dict[str, Any]], metric: str) -> bool:
    if metric == "decode":
        return any(safe_float(row.get("decode_tok_s")) is not None for row in rows)
    if metric == "prefill":
        return any(safe_float(row.get("prefill_tok_s")) is not None for row in rows)
    if metric == "ttft":
        return any(safe_float(row.get("ttft_s")) is not None for row in rows)
    if metric == "memory":
        return any(memory_bytes(row) is not None for row in rows)
    raise ValueError(metric)


def quantized_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if str(row.get("quantization") or "none") in {"mm8", "mm4", "w8", "w4", "int8", "int4", "lut8", "lut6", "lut4"}]


def state_cache_rows(rows: list[dict[str, Any]], *, tolerance: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        chunk_diff = safe_float(row.get("chunked_prefill_max_abs"))
        seen = safe_int(row.get("seen_tokens_after_generate"))
        expected = safe_int(row.get("expected_seen_tokens"))
        chunk_ok = chunk_diff is not None and chunk_diff <= tolerance
        seen_ok = seen is not None and expected is not None and seen == expected
        if chunk_ok and seen_ok:
            out.append(row)
    return out


def comparison_rows(rows: list[dict[str, Any]], tier: Tier, shape: Shape) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        if row.get("axis") != COMPARISON_AXIS:
            continue
        if not model_matches(row.get("qwen_model"), tier.qwen_aliases):
            continue
        if not model_matches(row.get("rwkv_model"), tier.rwkv_aliases):
            continue
        if str(row.get("prompt_case") or "") != shape.prompt_case:
            continue
        if safe_int(row.get("requested_generated_tokens")) != shape.requested_generated_tokens:
            continue
        out.append(row)
    return out


def quality_rows(rows: list[dict[str, Any]], tier: Tier, shape: Shape) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        if row.get("axis") != QUALITY_COMPARISON_AXIS:
            continue
        if not model_matches(row.get("qwen_model"), tier.qwen_aliases):
            continue
        if not model_matches(row.get("rwkv_model"), tier.rwkv_aliases):
            continue
        if str(row.get("prompt_case") or "") != shape.prompt_case:
            continue
        if safe_int(row.get("requested_generated_tokens")) != shape.requested_generated_tokens:
            continue
        out.append(row)
    return out


def coreml_status(rows: list[dict[str, Any]], tier: Tier) -> tuple[str, list[str], dict[str, Any]]:
    runtime_pass = [
        row
        for row in rows
        if row.get("axis") == BASELINE_AXIS
        and row.get("runtime") == "coreml"
        and row.get("status") == "pass"
        and row_matches_model(row, tier.rwkv_aliases)
    ]
    if runtime_pass:
        return "pass", [], {"runtime_pass_rows": len(runtime_pass)}
    partial_runtime = [
        row
        for row in rows
        if row.get("axis") == BASELINE_AXIS
        and row.get("runtime") == "coreml"
        and row.get("status") == "partial"
        and row_matches_model(row, tier.rwkv_aliases)
    ]
    export_pass = [
        row
        for row in rows
        if row.get("axis") == COREML_EXPORT_AXIS
        and row.get("status") in {"pass", "plan"}
        and row_matches_model(row, tier.rwkv_aliases)
    ]
    plan_rows = [row for row in rows if row.get("axis") == COREML_PLAN_AXIS and row_matches_model(row, tier.rwkv_aliases)]
    if partial_runtime or export_pass or plan_rows:
        return (
            "prototype",
            ["CoreML evidence is export/plan/full-logits only; stateful decode/prefill runtime row is missing"],
            {"partial_runtime_rows": len(partial_runtime), "export_or_plan_rows": len(export_pass) + len(plan_rows)},
        )
    return "missing", ["missing CoreML export/runtime evidence for this RWKV tier"], {}


def status_from_bool(ok: bool, missing_reason: str) -> tuple[str, list[str]]:
    if ok:
        return "pass", []
    return "missing", [missing_reason]


def audit_shape(rows: list[dict[str, Any]], tier: Tier, shape: Shape, *, state_tolerance: float, require_quality: bool) -> dict[str, Any]:
    qwen = pass_baseline_rows(rows, tier.qwen_aliases, shape)
    rwkv = pass_baseline_rows(rows, tier.rwkv_aliases, shape)
    checks: dict[str, dict[str, Any]] = {}
    actions: list[dict[str, Any]] = []

    q_status, q_reasons = status_from_bool(bool(qwen), "missing Qwen3.5 same-prompt baseline row")
    checks["qwen_row"] = {"status": q_status, "count": len(qwen), "reasons": q_reasons}
    if q_status != "pass":
        actions.append({"action": "collect_qwen_baseline_rows", "metric": "coverage", "severity": "blocker"})

    r_status, r_reasons = status_from_bool(bool(rwkv), "missing RWKV-7 same-prompt MLX row")
    checks["rwkv_mlx_row"] = {"status": r_status, "count": len(rwkv), "reasons": r_reasons}
    if r_status != "pass":
        actions.append({"action": "collect_rwkv_mlx_rows", "metric": "coverage", "severity": "blocker"})

    for metric in ("decode", "prefill", "ttft", "memory"):
        q_ok = has_metric(qwen, metric)
        r_ok = has_metric(rwkv, metric)
        status = "pass" if q_ok and r_ok else "missing"
        reasons = [] if status == "pass" else [f"missing {metric} metric on {'qwen' if not q_ok else ''}{' and ' if not q_ok and not r_ok else ''}{'rwkv' if not r_ok else ''} rows"]
        checks[metric] = {"status": status, "qwen_has_metric": q_ok, "rwkv_has_metric": r_ok, "reasons": reasons}
        if status != "pass":
            actions.append({"action": "collect_required_metric", "metric": metric, "severity": "blocker"})

    qrows = quantized_rows(rwkv)
    q_status, q_reasons = status_from_bool(bool(qrows), "missing RWKV W8/W4 quantized row for this shape")
    checks["quant"] = {"status": q_status, "count": len(qrows), "quantizations": sorted({str(row.get("quantization")) for row in qrows}), "reasons": q_reasons}
    if q_status != "pass":
        actions.append({"action": "collect_quantized_rwkv_rows", "metric": "quant", "severity": "blocker"})

    sc_rows = state_cache_rows(rwkv, tolerance=state_tolerance)
    sc_status, sc_reasons = status_from_bool(bool(sc_rows), "missing chunked-prefill/state-cache correctness evidence")
    checks["state_cache"] = {"status": sc_status, "count": len(sc_rows), "max_abs_tolerance": state_tolerance, "reasons": sc_reasons}
    if sc_status != "pass":
        actions.append({"action": "collect_state_cache_or_chunked_prefill_rows", "metric": "state_cache", "severity": "blocker"})

    comps = comparison_rows(rows, tier, shape)
    passing_comp = [row for row in comps if row.get("status") == "pass"]
    if passing_comp:
        comp_status = "pass"
        comp_reasons: list[str] = []
    elif comps:
        comp_status = "fail" if any(row.get("status") == "fail" for row in comps) else "unknown"
        comp_reasons = ["comparison gate exists but is not passing"]
    else:
        comp_status = "missing"
        comp_reasons = ["missing qwen35_apple_baseline_comparison row"]
    checks["comparison"] = {"status": comp_status, "count": len(comps), "reasons": comp_reasons}
    if comp_status == "missing":
        actions.append({"action": "run_comparison_gates", "metric": "comparison", "severity": "blocker"})
    elif comp_status != "pass":
        actions.append({"action": "close_speed_latency_memory_gap", "metric": "comparison", "severity": "fail"})

    quals = quality_rows(rows, tier, shape)
    passing_quality = [row for row in quals if row.get("status") == "pass"]
    if passing_quality:
        quality_status = "pass"
        quality_reasons: list[str] = []
    elif quals:
        quality_status = "fail" if any(row.get("status") == "fail" for row in quals) else "unknown"
        quality_reasons = ["quality comparison exists but is not passing"]
    else:
        quality_status = "missing" if require_quality else "unknown"
        quality_reasons = ["missing qwen35_apple_quality_comparison row"]
    checks["quality"] = {"status": quality_status, "count": len(quals), "required": bool(require_quality), "reasons": quality_reasons}
    if require_quality and quality_status == "missing":
        actions.append({"action": "collect_quality_rows_with_store_responses", "metric": "quality", "severity": "blocker"})
    elif require_quality and quality_status != "pass":
        actions.append({"action": "improve_or_distill_quality", "metric": "quality", "severity": "fail"})

    statuses = [check["status"] for check in checks.values() if check.get("status")]
    row_status = "pass" if statuses and all(status == "pass" or (status == "unknown" and not require_quality) for status in statuses) else "fail" if any(status == "fail" for status in statuses) else "missing"
    return {
        "axis": AUDIT_AXIS,
        "status": row_status,
        "tier": tier.tier_id,
        "qwen_model": tier.qwen_label,
        "qwen_aliases": list(tier.qwen_aliases),
        "rwkv_model": tier.rwkv_label,
        "rwkv_aliases": list(tier.rwkv_aliases),
        "prompt_case": shape.prompt_case,
        "requested_generated_tokens": int(shape.requested_generated_tokens),
        "scope": "shape",
        "checks": checks,
        "actions": actions,
    }


def audit_tier(rows: list[dict[str, Any]], tier: Tier, shapes: list[Shape], *, long_context_chars: int, require_coreml: bool) -> dict[str, Any]:
    rwkv_all = pass_baseline_rows(rows, tier.rwkv_aliases)
    long_rows = [row for row in rwkv_all if safe_int(row.get("prompt_target_chars")) is not None and int(row.get("prompt_target_chars")) >= int(long_context_chars)]
    long_status = "pass" if long_rows else "missing"
    core_status, core_reasons, core_extra = coreml_status(rows, tier)
    checks = {
        "long_context": {
            "status": long_status,
            "threshold_prompt_target_chars": int(long_context_chars),
            "count": len(long_rows),
            "reasons": [] if long_rows else ["missing RWKV long-context Apple row"],
        },
        "coreml_stateful_runtime": {
            "status": core_status if require_coreml else (core_status if core_status == "pass" else "unknown"),
            "required": bool(require_coreml),
            "reasons": core_reasons,
            **core_extra,
        },
    }
    actions: list[dict[str, Any]] = []
    if long_status != "pass":
        actions.append({"action": "collect_long_context_rows", "metric": "long_context", "severity": "blocker"})
    if require_coreml and core_status != "pass":
        actions.append({"action": "add_stateful_coreml_decode_prefill_runtime", "metric": "coreml", "severity": "blocker"})
    statuses = [check["status"] for check in checks.values()]
    row_status = "pass" if all(status == "pass" or (status == "unknown" and not require_coreml) for status in statuses) else "missing" if not any(status == "fail" for status in statuses) else "fail"
    return {
        "axis": AUDIT_AXIS,
        "status": row_status,
        "tier": tier.tier_id,
        "qwen_model": tier.qwen_label,
        "qwen_aliases": list(tier.qwen_aliases),
        "rwkv_model": tier.rwkv_label,
        "rwkv_aliases": list(tier.rwkv_aliases),
        "scope": "tier",
        "required_shapes": [shape.label for shape in shapes],
        "checks": checks,
        "actions": actions,
    }


def summarize(audits: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    action_counts: dict[str, int] = {}
    tiers: dict[str, dict[str, int]] = {}
    for row in audits:
        status = str(row.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
        tier = str(row.get("tier") or "")
        if tier:
            tier_counts = tiers.setdefault(tier, {})
            tier_counts[status] = tier_counts.get(status, 0) + 1
        for action in row.get("actions") or []:
            key = str(action.get("action") or "unknown")
            action_counts[key] = action_counts.get(key, 0) + 1
    status = "pass" if audits and set(counts) <= PASS_STATUSES else "fail" if counts.get("fail") else "missing" if counts.get("missing") else "unknown"
    return {
        "axis": SUMMARY_AXIS,
        "status": status,
        "rows": len(audits),
        "status_counts": dict(sorted(counts.items())),
        "tiers": tiers,
        "action_counts": dict(sorted(action_counts.items())),
        "top_actions": [
            {"action": action, "count": count}
            for action, count in sorted(action_counts.items(), key=lambda item: (-item[1], item[0]))[:10]
        ],
    }


def run_audit(
    rows: list[dict[str, Any]],
    *,
    tiers: list[Tier],
    shapes: list[Shape],
    state_tolerance: float,
    long_context_chars: int,
    require_quality: bool,
    require_coreml: bool,
) -> list[dict[str, Any]]:
    audits: list[dict[str, Any]] = []
    for tier in tiers:
        for shape in shapes:
            audits.append(audit_shape(rows, tier, shape, state_tolerance=state_tolerance, require_quality=require_quality))
        audits.append(audit_tier(rows, tier, shapes, long_context_chars=long_context_chars, require_coreml=require_coreml))
    audits.append(summarize(audits))
    return audits


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit RWKV-7 Apple/Qwen3.5 goal coverage from JSONL evidence.")
    ap.add_argument("--results", action="append", required=True, help="Input JSONL evidence file or directory; repeatable.")
    ap.add_argument("--tier", action="append", default=[], help="Tier mapping qwen_alias[|...]=rwkv_alias[|...]. Defaults to 0.8B/2B/4B/9B.")
    ap.add_argument("--required-shape", action="append", default=[], help="Required shape charsN:TOKENS; repeatable.")
    ap.add_argument("--state-cache-tolerance", type=float, default=1e-4)
    ap.add_argument("--long-context-chars", type=int, default=4096)
    ap.add_argument("--require-quality", action="store_true", help="Treat missing/failing quality comparison rows as gate failures.")
    ap.add_argument("--require-coreml", action="store_true", help="Require stateful CoreML runtime pass rows, not only export prototypes.")
    ap.add_argument("--append", default="", help="Optional JSONL path to append audit rows and summary.")
    ap.add_argument("--fail-on-gate", action="store_true", help="Exit 1 unless the audit summary passes.")
    args = ap.parse_args()

    rows = load_evidence(args.results)
    tiers = [parse_tier(item) for item in (args.tier or DEFAULT_TIERS)]
    shapes = [parse_shape(item) for item in (args.required_shape or DEFAULT_REQUIRED_SHAPES)]
    audits = run_audit(
        rows,
        tiers=tiers,
        shapes=shapes,
        state_tolerance=float(args.state_cache_tolerance),
        long_context_chars=int(args.long_context_chars),
        require_quality=bool(args.require_quality),
        require_coreml=bool(args.require_coreml),
    )
    for row in audits:
        print(json.dumps(row, ensure_ascii=False))
    append_jsonl(args.append, audits)
    if args.fail_on_gate and audits[-1].get("status") != "pass":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
