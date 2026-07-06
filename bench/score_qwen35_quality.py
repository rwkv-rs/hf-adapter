#!/usr/bin/env python3
# coding=utf-8
"""Score same-prompt Qwen3.5/RWKV Apple quality rows.

The speed gate is not enough for the Apple/mobile goal: we also need a quality
matrix that shows where RWKV can claim parity and where it still needs
model-side work (fine-tuning, distillation, prompt format, tokenizer fixes).

This script consumes rows from ``bench/run_qwen35_apple_baseline.py``.  It only
scores rows with stored response text by default, so a PR cannot infer quality
from a truncated preview.  Use ``--allow-preview`` only for local smoke rows.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

BASELINE_AXIS = "qwen35_apple_baseline"
QUALITY_AXIS = "qwen35_apple_quality"
COMPARISON_AXIS = "qwen35_apple_quality_comparison"
SUMMARY_AXIS = "qwen35_apple_quality_summary"
DEFAULT_FORBIDDEN = [
    "i cannot",
    "i can't",
    "as an ai language model",
    "无法回答",
    "不能回答",
]


@dataclass(frozen=True)
class Pair:
    qwen_model: str
    rwkv_model: str


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


def load_rubric(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {"tasks": []}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("rubric must be a JSON object")
    tasks = data.get("tasks", [])
    if not isinstance(tasks, list):
        raise ValueError("rubric.tasks must be a list")
    return data


def parse_pair(raw: str) -> Pair:
    if "=" not in raw:
        raise ValueError(f"--pair must use qwen_model=rwkv_model, got {raw!r}")
    left, right = raw.split("=", 1)
    left = left.strip()
    right = right.strip()
    if not left or not right:
        raise ValueError(f"invalid --pair {raw!r}")
    return Pair(left, right)


def safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def row_model_id(row: dict[str, Any]) -> str:
    return str(row.get("model") or row.get("model_path") or "")


def requested_tokens(row: dict[str, Any]) -> int | None:
    return safe_int(row.get("requested_generated_tokens", row.get("generated_tokens")))


def row_key(row: dict[str, Any]) -> tuple[str, str, int] | None:
    model = row_model_id(row)
    prompt_case = str(row.get("prompt_case") or "")
    tokens = requested_tokens(row)
    if not model or not prompt_case or tokens is None:
        return None
    return model, prompt_case, int(tokens)


def task_id(prompt_case: str, tokens: int) -> str:
    return f"{prompt_case}/decode{int(tokens)}"


def task_lookup(rubric: dict[str, Any]) -> dict[tuple[str, int | None], dict[str, Any]]:
    out: dict[tuple[str, int | None], dict[str, Any]] = {}
    for raw in rubric.get("tasks", []):
        if not isinstance(raw, dict):
            raise ValueError("each rubric task must be an object")
        prompt_case = str(raw.get("prompt_case") or "")
        if not prompt_case:
            raise ValueError("rubric task missing prompt_case")
        tokens = safe_int(raw.get("requested_generated_tokens"))
        out[(prompt_case, tokens)] = raw
    return out


def response_text(row: dict[str, Any], *, allow_preview: bool) -> tuple[str, bool]:
    text = row.get("response_text")
    if isinstance(text, str):
        return text, False
    if allow_preview and isinstance(row.get("response_preview"), str):
        return str(row.get("response_preview")), True
    return "", False


def score_row(row: dict[str, Any], task: dict[str, Any] | None, *, allow_preview: bool) -> dict[str, Any]:
    model = row_model_id(row)
    prompt_case = str(row.get("prompt_case") or "")
    tokens = int(requested_tokens(row) or 0)
    text, used_preview = response_text(row, allow_preview=allow_preview)
    lower = text.lower()
    required = [str(x) for x in (task or {}).get("required_substrings", [])]
    forbidden = [str(x) for x in (task or {}).get("forbidden_substrings", DEFAULT_FORBIDDEN)]
    min_chars = safe_int((task or {}).get("min_response_chars"))
    if min_chars is None:
        min_chars = 1

    required_hits = [needle for needle in required if needle.lower() in lower]
    forbidden_hits = [needle for needle in forbidden if needle.lower() in lower]
    missing_required = [needle for needle in required if needle not in required_hits]
    reasons: list[str] = []
    status = "pass"
    if not text:
        status = "unknown"
        reasons.append("missing response_text; rerun baseline with --store-responses")
    if used_preview:
        status = "unknown"
        reasons.append("scored from truncated response_preview")
    if len(text) < int(min_chars):
        status = "fail" if text else status
        reasons.append(f"response shorter than min_response_chars={min_chars}")
    if missing_required:
        status = "fail" if text else status
        reasons.append("missing required substrings")
    if forbidden_hits:
        status = "fail" if text else status
        reasons.append("forbidden substrings present")

    required_score = len(required_hits) / len(required) if required else 1.0
    length_score = min(1.0, len(text) / float(max(int(min_chars), 1))) if text else 0.0
    forbidden_score = 0.0 if forbidden_hits else 1.0
    score = round((required_score * 0.6) + (length_score * 0.2) + (forbidden_score * 0.2), 6)
    return {
        "axis": QUALITY_AXIS,
        "status": status,
        "model": model,
        "engine": row.get("engine"),
        "runtime": row.get("runtime"),
        "prompt_case": prompt_case,
        "requested_generated_tokens": tokens,
        "task_id": str((task or {}).get("id") or task_id(prompt_case, tokens)),
        "score": score,
        "required_score": round(required_score, 6),
        "length_score": round(length_score, 6),
        "forbidden_score": round(forbidden_score, 6),
        "response_chars": len(text),
        "used_preview": bool(used_preview),
        "required_substrings": required,
        "required_hits": required_hits,
        "missing_required": missing_required,
        "forbidden_hits": forbidden_hits,
        "reasons": reasons,
    }


def score_rows(rows: list[dict[str, Any]], rubric: dict[str, Any], *, allow_preview: bool = False) -> list[dict[str, Any]]:
    tasks = task_lookup(rubric)
    out: list[dict[str, Any]] = []
    for row in rows:
        if row.get("axis") != BASELINE_AXIS or row.get("status") != "pass":
            continue
        key = row_key(row)
        if key is None:
            continue
        _model, prompt_case, tokens = key
        task = tasks.get((prompt_case, tokens)) or tasks.get((prompt_case, None))
        out.append(score_row(row, task, allow_preview=allow_preview))
    return out


def compare_quality(scored: list[dict[str, Any]], pairs: list[Pair]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in scored:
        key = (str(row.get("model") or ""), str(row.get("prompt_case") or ""), int(row.get("requested_generated_tokens") or 0))
        grouped.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    for pair in pairs:
        q_shapes = {(prompt, tokens) for model, prompt, tokens in grouped if model == pair.qwen_model}
        r_shapes = {(prompt, tokens) for model, prompt, tokens in grouped if model == pair.rwkv_model}
        for prompt_case, tokens in sorted(q_shapes | r_shapes):
            q_rows = grouped.get((pair.qwen_model, prompt_case, tokens), [])
            r_rows = grouped.get((pair.rwkv_model, prompt_case, tokens), [])
            q_score = min((float(row["score"]) for row in q_rows if row.get("status") == "pass"), default=None)
            r_score = min((float(row["score"]) for row in r_rows if row.get("status") == "pass"), default=None)
            if not q_rows or not r_rows:
                status = "missing"
                reasons = ["missing qwen or rwkv quality rows"]
            elif q_score is None or r_score is None:
                status = "unknown"
                reasons = ["one side has no passing quality row"]
            elif r_score >= q_score:
                status = "pass"
                reasons = []
            else:
                status = "fail"
                reasons = ["rwkv quality score below qwen baseline"]
            out.append(
                {
                    "axis": COMPARISON_AXIS,
                    "status": status,
                    "qwen_model": pair.qwen_model,
                    "rwkv_model": pair.rwkv_model,
                    "prompt_case": prompt_case,
                    "requested_generated_tokens": int(tokens),
                    "qwen_min_score": q_score,
                    "rwkv_min_score": r_score,
                    "score_delta_rwkv_minus_qwen": round(float(r_score) - float(q_score), 6) if q_score is not None and r_score is not None else None,
                    "reasons": reasons,
                }
            )
    return out


def summarize(rows: list[dict[str, Any]], comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    comparison_counts: dict[str, int] = {}
    for row in rows:
        counts[str(row.get("status", "unknown"))] = counts.get(str(row.get("status", "unknown")), 0) + 1
    for row in comparisons:
        comparison_counts[str(row.get("status", "unknown"))] = comparison_counts.get(str(row.get("status", "unknown")), 0) + 1
    if comparison_counts.get("fail") or comparison_counts.get("missing"):
        status = "fail"
    elif comparison_counts.get("unknown"):
        status = "unknown"
    elif comparisons:
        status = "pass"
    elif counts.get("fail"):
        status = "fail"
    elif counts.get("unknown") or not rows:
        status = "unknown"
    else:
        status = "pass"
    return {
        "axis": SUMMARY_AXIS,
        "status": status,
        "quality_rows": len(rows),
        "quality_status_counts": counts,
        "comparison_rows": len(comparisons),
        "comparison_status_counts": comparison_counts,
        "min_score": min((float(row["score"]) for row in rows if row.get("score") is not None), default=None),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Score Apple/Qwen3.5 same-prompt quality rows.")
    ap.add_argument("--results", required=True, help="Input JSONL from bench/run_qwen35_apple_baseline.py")
    ap.add_argument("--rubric", default="", help="Optional quality rubric JSON")
    ap.add_argument("--pair", action="append", default=[], help="qwen_model=rwkv_model comparison pair")
    ap.add_argument("--allow-preview", action="store_true", help="Allow truncated response_preview rows for smoke scoring")
    ap.add_argument("--append", default="", help="Optional JSONL path to append quality rows/comparisons/summary")
    ap.add_argument("--fail-on-gate", action="store_true", help="Exit 1 if summary is fail/unknown")
    args = ap.parse_args()

    rows = load_jsonl(args.results)
    rubric = load_rubric(args.rubric or None)
    quality_rows = score_rows(rows, rubric, allow_preview=bool(args.allow_preview))
    pairs = [parse_pair(raw) for raw in args.pair]
    comparisons = compare_quality(quality_rows, pairs) if pairs else []
    summary = summarize(quality_rows, comparisons)
    output = [*quality_rows, *comparisons, summary]
    for row in output:
        print(json.dumps(row, ensure_ascii=False))
    append_jsonl(args.append, output)
    if args.fail_on_gate and summary.get("status") != "pass":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
