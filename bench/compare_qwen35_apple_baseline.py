#!/usr/bin/env python3
# coding=utf-8
"""Compare RWKV-7 Apple rows against Qwen3.5 Apple baseline rows.

Input rows are produced by :mod:`bench.run_qwen35_apple_baseline`.  The script
uses conservative aggregation by default:

* speed gates use the minimum observed tok/s for each model/prompt/decode group;
* TTFT latency uses the maximum observed TTFT;
* memory uses the maximum observed runtime memory when both sides expose it.

Rows with missing required fields are kept out of pass/fail claims and reported
as ``unknown`` so PRs cannot accidentally turn a missing metric into a win.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

BASELINE_AXIS = "qwen35_apple_baseline"
COMPARISON_AXIS = "qwen35_apple_baseline_comparison"
SUMMARY_AXIS = "qwen35_apple_baseline_comparison_summary"
DIAGNOSTIC_AXIS = "qwen35_apple_baseline_gap_diagnostic"


@dataclass(frozen=True)
class Pair:
    qwen_model: str
    rwkv_model: str


def safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


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


def append_jsonl(path: str | Path | None, rows: Iterable[dict[str, Any]]) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_pair(raw: str) -> Pair:
    if "=" not in raw:
        raise ValueError(f"--pair must use qwen_model=rwkv_model, got {raw!r}")
    left, right = raw.split("=", 1)
    left = left.strip()
    right = right.strip()
    if not left or not right:
        raise ValueError(f"invalid --pair {raw!r}")
    return Pair(qwen_model=left, rwkv_model=right)


def row_model_id(row: dict[str, Any]) -> str:
    return str(row.get("model") or row.get("model_path") or "")


def requested_tokens(row: dict[str, Any]) -> int | None:
    return safe_int(row.get("requested_generated_tokens", row.get("generated_tokens")))


def memory_bytes(row: dict[str, Any]) -> int | None:
    for key in (
        "peak_memory_bytes",
        "mlx_peak_memory_bytes",
        "ollama_peak_memory_bytes",
        "rss_peak_memory_bytes",
    ):
        value = safe_int(row.get(key))
        if value is not None:
            return value
    value_mb = safe_float(row.get("peak_memory_mb"))
    if value_mb is not None:
        return int(value_mb * 1024 * 1024)
    return None


def group_key(row: dict[str, Any]) -> tuple[str, str, int] | None:
    model = row_model_id(row)
    prompt_case = str(row.get("prompt_case") or "")
    tokens = requested_tokens(row)
    if not model or not prompt_case or tokens is None:
        return None
    return model, prompt_case, int(tokens)


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    decode_values = [value for value in (safe_float(row.get("decode_tok_s")) for row in rows) if value is not None]
    prefill_values = [value for value in (safe_float(row.get("prefill_tok_s")) for row in rows) if value is not None]
    ttft_values = [value for value in (safe_float(row.get("ttft_s")) for row in rows) if value is not None]
    memory_values = [value for value in (memory_bytes(row) for row in rows) if value is not None]
    generated_values = [value for value in (safe_int(row.get("generated_tokens")) for row in rows) if value is not None]
    public_package = [value for value in (safe_float(row.get("public_package_gb")) for row in rows) if value is not None]
    return {
        "count": len(rows),
        "min_decode_tok_s": round(min(decode_values), 6) if decode_values else None,
        "min_prefill_tok_s": round(min(prefill_values), 6) if prefill_values else None,
        "max_ttft_s": round(max(ttft_values), 6) if ttft_values else None,
        "max_memory_bytes": max(memory_values) if memory_values else None,
        "min_generated_tokens": min(generated_values) if generated_values else None,
        "public_package_gb": max(public_package) if public_package else None,
    }


def ratio(numerator: Any, denominator: Any) -> float | None:
    n = safe_float(numerator)
    d = safe_float(denominator)
    if n is None or d is None or d == 0.0:
        return None
    return round(float(n) / float(d), 6)


def gate_at_least(value: float | None, threshold: float) -> bool | None:
    if value is None:
        return None
    return bool(value >= float(threshold))


def gate_at_most(value: float | None, threshold: float) -> bool | None:
    if value is None:
        return None
    return bool(value <= float(threshold))


def bool_status(values: list[bool | None]) -> tuple[str, list[str]]:
    unknown = [idx for idx, value in enumerate(values) if value is None]
    if any(value is False for value in values):
        return "fail", []
    if unknown:
        return "unknown", ["one or more required comparison metrics are missing"]
    return "pass", []


def _action(
    *,
    action: str,
    metric: str,
    severity: str,
    reason: str,
    current: Any = None,
    target: Any = None,
    ratio_value: Any = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "action": action,
        "metric": metric,
        "severity": severity,
        "reason": reason,
    }
    if current is not None:
        row["current"] = current
    if target is not None:
        row["target"] = target
    if ratio_value is not None:
        row["ratio"] = ratio_value
    if extra:
        row.update(extra)
    return row


def _required_speed(qwen_speed: Any, threshold: Any) -> float | None:
    q = safe_float(qwen_speed)
    t = safe_float(threshold)
    if q is None or t is None:
        return None
    return round(q * t, 6)


def _required_latency(qwen_latency: Any, threshold: Any) -> float | None:
    q = safe_float(qwen_latency)
    t = safe_float(threshold)
    if q is None or t is None:
        return None
    return round(q * t, 6)


def _required_memory(qwen_memory: Any, threshold: Any) -> int | None:
    q = safe_float(qwen_memory)
    t = safe_float(threshold)
    if q is None or t is None:
        return None
    return int(q * t)


def comparison_gap_actions(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Return concrete next actions for a comparison row.

    The comparison gate is intentionally conservative: missing metrics become
    ``unknown`` instead of pass.  These actions make that conservative result
    operational by spelling out whether the next run should collect data, tune
    decode/prefill speed, reduce TTFT, or lower memory.
    """

    actions: list[dict[str, Any]] = []
    status = str(row.get("status") or "unknown")
    qwen = row.get("qwen") if isinstance(row.get("qwen"), dict) else {}
    rwkv = row.get("rwkv") if isinstance(row.get("rwkv"), dict) else {}
    gates = row.get("gates") if isinstance(row.get("gates"), dict) else {}

    if status == "pass":
        return actions

    if status == "missing" or qwen.get("count", 0) == 0:
        actions.append(
            _action(
                action="collect_qwen_baseline_rows",
                metric="coverage",
                severity="blocker",
                reason="missing Qwen3.5 same-prompt baseline rows for this model/prompt/decode shape",
            )
        )
    if status == "missing" or rwkv.get("count", 0) == 0:
        actions.append(
            _action(
                action="collect_rwkv_mlx_or_coreml_rows",
                metric="coverage",
                severity="blocker",
                reason="missing RWKV-7 same-prompt comparison rows for this model/prompt/decode shape",
            )
        )

    decode_threshold = safe_float(gates.get("min_decode_ratio"))
    decode_ratio = safe_float(row.get("decode_ratio_rwkv_over_qwen"))
    if decode_threshold is not None:
        if decode_ratio is None:
            actions.append(
                _action(
                    action="collect_decode_tok_s",
                    metric="decode",
                    severity="blocker",
                    reason="decode gate requested but decode_tok_s is missing on one or both sides",
                    extra={"qwen_min_decode_tok_s": qwen.get("min_decode_tok_s"), "rwkv_min_decode_tok_s": rwkv.get("min_decode_tok_s")},
                )
            )
        elif decode_ratio < decode_threshold:
            target = _required_speed(qwen.get("min_decode_tok_s"), decode_threshold)
            current = rwkv.get("min_decode_tok_s")
            extra: dict[str, Any] = {"qwen_min_decode_tok_s": qwen.get("min_decode_tok_s")}
            current_f = safe_float(current)
            if target is not None and current_f and current_f > 0:
                extra["needed_speedup_over_current"] = round(target / current_f, 6)
            actions.append(
                _action(
                    action="optimize_decode_kernel_or_batching",
                    metric="decode",
                    severity="fail",
                    reason="RWKV decode tok/s is below the configured Qwen3.5 ratio gate",
                    current=current,
                    target=target,
                    ratio_value=decode_ratio,
                    extra=extra,
                )
            )

    prefill_threshold = safe_float(gates.get("min_prefill_ratio"))
    prefill_ratio = safe_float(row.get("prefill_ratio_rwkv_over_qwen"))
    if prefill_threshold is not None:
        if prefill_ratio is None:
            actions.append(
                _action(
                    action="collect_prefill_tok_s",
                    metric="prefill",
                    severity="blocker",
                    reason="prefill gate requested but prefill_tok_s is missing on one or both sides",
                    extra={"qwen_min_prefill_tok_s": qwen.get("min_prefill_tok_s"), "rwkv_min_prefill_tok_s": rwkv.get("min_prefill_tok_s")},
                )
            )
        elif prefill_ratio < prefill_threshold:
            target = _required_speed(qwen.get("min_prefill_tok_s"), prefill_threshold)
            current = rwkv.get("min_prefill_tok_s")
            extra = {"qwen_min_prefill_tok_s": qwen.get("min_prefill_tok_s")}
            current_f = safe_float(current)
            if target is not None and current_f and current_f > 0:
                extra["needed_speedup_over_current"] = round(target / current_f, 6)
            actions.append(
                _action(
                    action="optimize_prefill_or_chunked_prefill",
                    metric="prefill",
                    severity="fail",
                    reason="RWKV prefill tok/s is below the configured Qwen3.5 ratio gate",
                    current=current,
                    target=target,
                    ratio_value=prefill_ratio,
                    extra=extra,
                )
            )

    ttft_threshold = safe_float(gates.get("max_ttft_ratio"))
    ttft_ratio = safe_float(row.get("ttft_ratio_rwkv_over_qwen"))
    if ttft_threshold is not None:
        if ttft_ratio is None:
            actions.append(
                _action(
                    action="collect_ttft_s",
                    metric="ttft",
                    severity="blocker",
                    reason="TTFT gate requested but ttft_s is missing on one or both sides",
                    extra={"qwen_max_ttft_s": qwen.get("max_ttft_s"), "rwkv_max_ttft_s": rwkv.get("max_ttft_s")},
                )
            )
        elif ttft_ratio > ttft_threshold:
            target = _required_latency(qwen.get("max_ttft_s"), ttft_threshold)
            current = rwkv.get("max_ttft_s")
            extra = {"qwen_max_ttft_s": qwen.get("max_ttft_s")}
            current_f = safe_float(current)
            if target is not None and current_f and current_f > 0:
                extra["needed_latency_ratio_over_current"] = round(target / current_f, 6)
            actions.append(
                _action(
                    action="reduce_ttft_load_prefill_or_first_token",
                    metric="ttft",
                    severity="fail",
                    reason="RWKV TTFT is above the configured Qwen3.5 ratio gate",
                    current=current,
                    target=target,
                    ratio_value=ttft_ratio,
                    extra=extra,
                )
            )

    memory_threshold = safe_float(gates.get("max_memory_ratio"))
    memory_ratio_value = safe_float(row.get("memory_ratio_rwkv_over_qwen"))
    if memory_threshold is not None:
        if memory_ratio_value is None:
            actions.append(
                _action(
                    action="collect_memory_telemetry",
                    metric="memory",
                    severity="blocker",
                    reason="memory gate requested but peak memory telemetry is missing on one or both sides",
                    extra={"qwen_max_memory_bytes": qwen.get("max_memory_bytes"), "rwkv_max_memory_bytes": rwkv.get("max_memory_bytes")},
                )
            )
        elif memory_ratio_value > memory_threshold:
            target = _required_memory(qwen.get("max_memory_bytes"), memory_threshold)
            current = rwkv.get("max_memory_bytes")
            extra = {"qwen_max_memory_bytes": qwen.get("max_memory_bytes")}
            current_f = safe_float(current)
            if target is not None and current_f and current_f > 0:
                extra["needed_memory_ratio_over_current"] = round(target / current_f, 6)
                extra["needed_memory_reduction_bytes"] = int(current_f - target)
            actions.append(
                _action(
                    action="reduce_peak_memory_or_quantize_more",
                    metric="memory",
                    severity="fail",
                    reason="RWKV peak memory is above the configured Qwen3.5 ratio gate",
                    current=current,
                    target=target,
                    ratio_value=memory_ratio_value,
                    extra=extra,
                )
            )

    return actions


def gap_diagnostic_rows(comparisons: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for comparison in comparisons:
        actions = comparison_gap_actions(comparison)
        if not actions:
            continue
        rows.append(
            {
                "axis": DIAGNOSTIC_AXIS,
                "status": comparison.get("status", "unknown"),
                "qwen_model": comparison.get("qwen_model"),
                "rwkv_model": comparison.get("rwkv_model"),
                "prompt_case": comparison.get("prompt_case"),
                "requested_generated_tokens": comparison.get("requested_generated_tokens"),
                "actions": actions,
                "action_count": len(actions),
            }
        )
    return rows


def summarize_gap_actions(comparisons: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for comparison in comparisons:
        for action in comparison_gap_actions(comparison):
            key = str(action.get("action") or "unknown")
            counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def compare_group(
    *,
    pair: Pair,
    prompt_case: str,
    requested_generated_tokens: int,
    qwen_rows: list[dict[str, Any]],
    rwkv_rows: list[dict[str, Any]],
    min_decode_ratio: float,
    min_prefill_ratio: float,
    max_ttft_ratio: float,
    max_memory_ratio: float,
    require_prefill: bool,
    require_ttft: bool,
    require_memory: bool,
) -> dict[str, Any]:
    qwen = aggregate(qwen_rows)
    rwkv = aggregate(rwkv_rows)

    decode_ratio = ratio(rwkv.get("min_decode_tok_s"), qwen.get("min_decode_tok_s"))
    prefill_ratio = ratio(rwkv.get("min_prefill_tok_s"), qwen.get("min_prefill_tok_s"))
    ttft_ratio = ratio(rwkv.get("max_ttft_s"), qwen.get("max_ttft_s"))
    memory_ratio = ratio(rwkv.get("max_memory_bytes"), qwen.get("max_memory_bytes"))
    package_ratio = None
    if rwkv.get("max_memory_bytes") is not None and qwen.get("public_package_gb") is not None:
        package_ratio = round((float(rwkv["max_memory_bytes"]) / (1024 ** 3)) / float(qwen["public_package_gb"]), 6)

    decode_gate = gate_at_least(decode_ratio, min_decode_ratio)
    prefill_gate = gate_at_least(prefill_ratio, min_prefill_ratio) if require_prefill else None
    ttft_gate = gate_at_most(ttft_ratio, max_ttft_ratio) if require_ttft else None
    memory_gate = gate_at_most(memory_ratio, max_memory_ratio) if require_memory else None
    gates = [decode_gate]
    if require_prefill:
        gates.append(prefill_gate)
    if require_ttft:
        gates.append(ttft_gate)
    if require_memory:
        gates.append(memory_gate)
    status, unknown_reasons = bool_status(gates)

    if not qwen_rows:
        status = "missing"
        unknown_reasons.append("missing qwen baseline rows")
    if not rwkv_rows:
        status = "missing"
        unknown_reasons.append("missing rwkv comparison rows")

    return {
        "axis": COMPARISON_AXIS,
        "status": status,
        "qwen_model": pair.qwen_model,
        "rwkv_model": pair.rwkv_model,
        "prompt_case": prompt_case,
        "requested_generated_tokens": int(requested_generated_tokens),
        "qwen": qwen,
        "rwkv": rwkv,
        "decode_ratio_rwkv_over_qwen": decode_ratio,
        "prefill_ratio_rwkv_over_qwen": prefill_ratio,
        "ttft_ratio_rwkv_over_qwen": ttft_ratio,
        "memory_ratio_rwkv_over_qwen": memory_ratio,
        "rwkv_memory_gb_over_qwen_public_package_gb": package_ratio,
        "gates": {
            "min_decode_ratio": float(min_decode_ratio),
            "decode_gate_pass": decode_gate,
            "min_prefill_ratio": float(min_prefill_ratio) if require_prefill else None,
            "prefill_gate_pass": prefill_gate,
            "max_ttft_ratio": float(max_ttft_ratio) if require_ttft else None,
            "ttft_gate_pass": ttft_gate,
            "max_memory_ratio": float(max_memory_ratio) if require_memory else None,
            "memory_gate_pass": memory_gate,
        },
        "unknown_reasons": unknown_reasons,
    }


def compare_rows(
    rows: list[dict[str, Any]],
    *,
    pairs: list[Pair],
    min_decode_ratio: float = 1.0,
    min_prefill_ratio: float = 1.0,
    max_ttft_ratio: float = 1.1,
    max_memory_ratio: float = 1.0,
    require_prefill: bool = False,
    require_ttft: bool = False,
    require_memory: bool = False,
) -> list[dict[str, Any]]:
    pass_rows = [row for row in rows if row.get("axis") == BASELINE_AXIS and row.get("status") == "pass"]
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in pass_rows:
        key = group_key(row)
        if key is not None:
            grouped.setdefault(key, []).append(row)

    comparisons: list[dict[str, Any]] = []
    for pair in pairs:
        qwen_shapes = {(prompt, tokens) for model, prompt, tokens in grouped if model == pair.qwen_model}
        rwkv_shapes = {(prompt, tokens) for model, prompt, tokens in grouped if model == pair.rwkv_model}
        shapes = sorted(qwen_shapes | rwkv_shapes)
        if not shapes:
            comparisons.append(
                compare_group(
                    pair=pair,
                    prompt_case="",
                    requested_generated_tokens=0,
                    qwen_rows=[],
                    rwkv_rows=[],
                    min_decode_ratio=min_decode_ratio,
                    min_prefill_ratio=min_prefill_ratio,
                    max_ttft_ratio=max_ttft_ratio,
                    max_memory_ratio=max_memory_ratio,
                    require_prefill=require_prefill,
                    require_ttft=require_ttft,
                    require_memory=require_memory,
                )
            )
            continue
        for prompt_case, tokens in shapes:
            comparisons.append(
                compare_group(
                    pair=pair,
                    prompt_case=prompt_case,
                    requested_generated_tokens=tokens,
                    qwen_rows=grouped.get((pair.qwen_model, prompt_case, tokens), []),
                    rwkv_rows=grouped.get((pair.rwkv_model, prompt_case, tokens), []),
                    min_decode_ratio=min_decode_ratio,
                    min_prefill_ratio=min_prefill_ratio,
                    max_ttft_ratio=max_ttft_ratio,
                    max_memory_ratio=max_memory_ratio,
                    require_prefill=require_prefill,
                    require_ttft=require_ttft,
                    require_memory=require_memory,
                )
            )
    return comparisons


def summarize_comparisons(comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for row in comparisons:
        counts[str(row.get("status", "unknown"))] = counts.get(str(row.get("status", "unknown")), 0) + 1
    if counts.get("fail", 0) or counts.get("missing", 0):
        status = "fail"
    elif counts.get("unknown", 0):
        status = "unknown"
    else:
        status = "pass"
    gap_action_counts = summarize_gap_actions(comparisons)
    return {
        "axis": SUMMARY_AXIS,
        "status": status,
        "comparisons": len(comparisons),
        "status_counts": counts,
        "gap_action_counts": gap_action_counts,
        "top_gap_actions": [
            {"action": action, "count": count}
            for action, count in sorted(gap_action_counts.items(), key=lambda item: (-item[1], item[0]))[:8]
        ],
        "min_decode_ratio": min(
            (float(row["decode_ratio_rwkv_over_qwen"]) for row in comparisons if row.get("decode_ratio_rwkv_over_qwen") is not None),
            default=None,
        ),
        "min_prefill_ratio": min(
            (float(row["prefill_ratio_rwkv_over_qwen"]) for row in comparisons if row.get("prefill_ratio_rwkv_over_qwen") is not None),
            default=None,
        ),
        "max_ttft_ratio": max(
            (float(row["ttft_ratio_rwkv_over_qwen"]) for row in comparisons if row.get("ttft_ratio_rwkv_over_qwen") is not None),
            default=None,
        ),
        "max_memory_ratio": max(
            (float(row["memory_ratio_rwkv_over_qwen"]) for row in comparisons if row.get("memory_ratio_rwkv_over_qwen") is not None),
            default=None,
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare RWKV-7 Apple baseline rows against Qwen3.5 rows.")
    ap.add_argument("--results", required=True, help="Input JSONL from bench/run_qwen35_apple_baseline.py")
    ap.add_argument(
        "--pair",
        action="append",
        default=[],
        help="Comparison pair in the form qwen_model=rwkv_model, e.g. qwen3.5:2b-mlx=rwkv7-g1g-1.5b-hf",
    )
    ap.add_argument("--min-decode-ratio", type=float, default=1.0)
    ap.add_argument("--min-prefill-ratio", type=float, default=1.0)
    ap.add_argument("--max-ttft-ratio", type=float, default=1.1)
    ap.add_argument("--max-memory-ratio", type=float, default=1.0)
    ap.add_argument("--require-prefill", action="store_true")
    ap.add_argument("--require-ttft", action="store_true")
    ap.add_argument("--require-memory", action="store_true")
    ap.add_argument("--append", default="", help="Optional JSONL path to append comparison rows and summary.")
    ap.add_argument("--diagnostics", action="store_true", help="Emit gap-diagnostic rows with concrete next actions for missing/failing gates.")
    ap.add_argument("--fail-on-gate", action="store_true", help="Exit 1 if summary is fail/unknown.")
    args = ap.parse_args()

    pairs = [parse_pair(raw) for raw in args.pair]
    if not pairs:
        raise ValueError("at least one --pair is required")
    rows = load_jsonl(args.results)
    comparisons = compare_rows(
        rows,
        pairs=pairs,
        min_decode_ratio=float(args.min_decode_ratio),
        min_prefill_ratio=float(args.min_prefill_ratio),
        max_ttft_ratio=float(args.max_ttft_ratio),
        max_memory_ratio=float(args.max_memory_ratio),
        require_prefill=bool(args.require_prefill),
        require_ttft=bool(args.require_ttft),
        require_memory=bool(args.require_memory),
    )
    diagnostics = gap_diagnostic_rows(comparisons) if args.diagnostics else []
    summary = summarize_comparisons(comparisons)
    for row in comparisons:
        print(json.dumps(row, ensure_ascii=False))
    for row in diagnostics:
        print(json.dumps(row, ensure_ascii=False))
    print(json.dumps(summary, ensure_ascii=False))
    append_jsonl(args.append, [*comparisons, *diagnostics, summary])
    if args.fail_on_gate and summary.get("status") != "pass":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
