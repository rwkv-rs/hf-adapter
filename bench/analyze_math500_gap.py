#!/usr/bin/env python3
# coding=utf-8
"""Analyze per-row and per-task MATH500 gaps between two rollout runs.

This is intended for the MATH500 avg@64 parity loop.  It compares two
`generations.jsonl` files emitted by the HF adapter runner and the Albatross
runner, keyed by `(task_index, sample_id)`, and emits a compact report that can
be committed without storing the full generations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()


def _snippet(text: str, limit: int) -> str:
    text = text.replace("\r\n", "\n")
    if len(text) <= limit:
        return text
    head = max(0, (limit - 15) // 2)
    tail = max(0, limit - 15 - head)
    return text[:head] + "\n...<snip>...\n" + text[-tail:]


def _as_bool(value: Any) -> bool:
    return bool(value)


def load_rows(path: str | Path, *, snippet_chars: int) -> tuple[dict[tuple[int, int], dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    rows: dict[tuple[int, int], dict[str, Any]] = {}
    by_task: dict[int, list[dict[str, Any]]] = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            task_index = int(raw["task_index"])
            sample_id = int(raw["sample_id"])
            completion = str(raw.get("completion", ""))
            row = {
                "task_index": task_index,
                "sample_id": sample_id,
                "problem": str(raw.get("problem", "")),
                "answer": str(raw.get("answer", "")),
                "subject": str(raw.get("subject", "")),
                "level": str(raw.get("level", "")),
                "unique_id": str(raw.get("unique_id", task_index)),
                "prompt_tokens": int(raw.get("prompt_tokens", 0)),
                "generated_tokens": int(raw.get("generated_tokens", 0)),
                "tokens_including_stop": int(raw.get("tokens_including_stop", raw.get("tokens_including_eod", raw.get("generated_tokens", 0)))),
                "ended_eod": _as_bool(raw.get("ended_eod", False)),
                "ended_user_stop": _as_bool(raw.get("ended_user_stop", False)),
                "stop_reason": str(raw.get("stop_reason", "")),
                "truncated": _as_bool(raw.get("truncated", False)),
                "correct": _as_bool(raw.get("correct", False)),
                "verify_error": str(raw.get("verify_error", "")),
                "completion_sha1": _sha1(completion),
                "completion_snippet": _snippet(completion, snippet_chars),
                "line_no": line_no,
            }
            key = (task_index, sample_id)
            if key in rows:
                raise ValueError(f"duplicate key {key} in {path} at line {line_no}")
            rows[key] = row
            by_task[task_index].append(row)
    return rows, by_task


def _rate(n: int | float, d: int | float) -> float:
    return float(n) / float(d) if d else 0.0


def _stop_counts(rows: dict[tuple[int, int], dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(r.get("stop_reason", "")) for r in rows.values()))


def _verify_counts(rows: dict[tuple[int, int], dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(r.get("verify_error", "")) for r in rows.values()).most_common(20))


def task_summary(by_task: dict[int, list[dict[str, Any]]]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for task_index, rows in by_task.items():
        rows_sorted = sorted(rows, key=lambda r: int(r["sample_id"]))
        correct = sum(int(r["correct"]) for r in rows_sorted)
        truncated = sum(int(r["truncated"]) for r in rows_sorted)
        generated = [int(r["generated_tokens"]) for r in rows_sorted]
        out[task_index] = {
            "task_index": task_index,
            "problem": rows_sorted[0].get("problem", ""),
            "answer": rows_sorted[0].get("answer", ""),
            "prompt_tokens": rows_sorted[0].get("prompt_tokens", 0),
            "rows": len(rows_sorted),
            "correct": correct,
            "pass": correct > 0,
            "truncated": truncated,
            "mean_generated_tokens": sum(generated) / max(len(generated), 1),
        }
    return out


def compare(hf: dict[tuple[int, int], dict[str, Any]], alb: dict[tuple[int, int], dict[str, Any]], *, top_n: int, examples_per_bucket: int) -> dict[str, Any]:
    hf_keys = set(hf)
    alb_keys = set(alb)
    common = sorted(hf_keys & alb_keys)
    missing_in_hf = sorted(alb_keys - hf_keys)
    missing_in_albatross = sorted(hf_keys - alb_keys)

    row_counts = Counter()
    first_completion_diffs: list[dict[str, Any]] = []
    prompt_token_diffs: list[dict[str, Any]] = []
    row_correct_examples: dict[str, list[dict[str, Any]]] = {"hf_only_correct": [], "albatross_only_correct": []}

    for key in common:
        a = hf[key]
        b = alb[key]
        hf_correct = bool(a["correct"])
        alb_correct = bool(b["correct"])
        same_completion = a["completion_sha1"] == b["completion_sha1"]
        same_tokens = (
            int(a["generated_tokens"]) == int(b["generated_tokens"])
            and int(a["tokens_including_stop"]) == int(b["tokens_including_stop"])
        )
        if same_completion:
            row_counts["same_completion"] += 1
        else:
            row_counts["different_completion"] += 1
            if len(first_completion_diffs) < examples_per_bucket:
                first_completion_diffs.append(_row_example(a, b))
        if same_tokens:
            row_counts["same_token_counts"] += 1
        else:
            row_counts["different_token_counts"] += 1
        if int(a["prompt_tokens"]) != int(b["prompt_tokens"]):
            row_counts["prompt_token_diffs"] += 1
            if len(prompt_token_diffs) < examples_per_bucket:
                prompt_token_diffs.append(
                    {
                        "task_index": key[0],
                        "sample_id": key[1],
                        "hf_prompt_tokens": a["prompt_tokens"],
                        "albatross_prompt_tokens": b["prompt_tokens"],
                    }
                )
        if hf_correct and alb_correct:
            row_counts["both_correct"] += 1
        elif hf_correct and not alb_correct:
            row_counts["hf_only_correct"] += 1
            if len(row_correct_examples["hf_only_correct"]) < examples_per_bucket:
                row_correct_examples["hf_only_correct"].append(_row_example(a, b))
        elif alb_correct and not hf_correct:
            row_counts["albatross_only_correct"] += 1
            if len(row_correct_examples["albatross_only_correct"]) < examples_per_bucket:
                row_correct_examples["albatross_only_correct"].append(_row_example(a, b))
        else:
            row_counts["both_wrong"] += 1

    hf_by_task = task_summary(_group_by_task(hf))
    alb_by_task = task_summary(_group_by_task(alb))
    common_tasks = sorted(set(hf_by_task) & set(alb_by_task))
    task_deltas: list[dict[str, Any]] = []
    hf_only_pass_tasks: list[int] = []
    alb_only_pass_tasks: list[int] = []
    for task_index in common_tasks:
        h = hf_by_task[task_index]
        a = alb_by_task[task_index]
        if h["pass"] and not a["pass"]:
            hf_only_pass_tasks.append(task_index)
        if a["pass"] and not h["pass"]:
            alb_only_pass_tasks.append(task_index)
        delta = int(h["correct"]) - int(a["correct"])
        if delta != 0:
            task_deltas.append(
                {
                    "task_index": task_index,
                    "hf_correct": int(h["correct"]),
                    "albatross_correct": int(a["correct"]),
                    "hf_minus_albatross_correct": delta,
                    "albatross_minus_hf_correct": -delta,
                    "hf_pass": bool(h["pass"]),
                    "albatross_pass": bool(a["pass"]),
                    "hf_truncated": int(h["truncated"]),
                    "albatross_truncated": int(a["truncated"]),
                    "problem": h["problem"],
                    "answer": h["answer"],
                }
            )

    task_deltas_hf_adv = sorted(task_deltas, key=lambda x: x["hf_minus_albatross_correct"], reverse=True)[:top_n]
    task_deltas_alb_adv = sorted(task_deltas, key=lambda x: x["albatross_minus_hf_correct"], reverse=True)[:top_n]

    total = len(common)
    hf_correct = sum(int(r["correct"]) for r in hf.values())
    alb_correct = sum(int(r["correct"]) for r in alb.values())
    hf_pass = sum(1 for t in hf_by_task.values() if t["pass"])
    alb_pass = sum(1 for t in alb_by_task.values() if t["pass"])
    task_count = len(common_tasks)

    return {
        "shape": {
            "hf_rows": len(hf),
            "albatross_rows": len(alb),
            "common_rows": total,
            "missing_in_hf": missing_in_hf[:top_n],
            "missing_in_albatross": missing_in_albatross[:top_n],
            "hf_tasks": len(hf_by_task),
            "albatross_tasks": len(alb_by_task),
            "common_tasks": task_count,
        },
        "summary": {
            "hf_correct_generations": hf_correct,
            "albatross_correct_generations": alb_correct,
            "correct_generation_delta_hf_minus_albatross": hf_correct - alb_correct,
            "hf_rollout_accuracy": _rate(hf_correct, len(hf)),
            "albatross_rollout_accuracy": _rate(alb_correct, len(alb)),
            "rollout_accuracy_delta_hf_minus_albatross": _rate(hf_correct, len(hf)) - _rate(alb_correct, len(alb)),
            "hf_pass_at_rollout": _rate(hf_pass, max(len(hf_by_task), 1)),
            "albatross_pass_at_rollout": _rate(alb_pass, max(len(alb_by_task), 1)),
            "pass_at_rollout_delta_hf_minus_albatross": _rate(hf_pass, max(len(hf_by_task), 1)) - _rate(alb_pass, max(len(alb_by_task), 1)),
            "hf_pass_tasks": hf_pass,
            "albatross_pass_tasks": alb_pass,
            "hf_only_pass_task_count": len(hf_only_pass_tasks),
            "albatross_only_pass_task_count": len(alb_only_pass_tasks),
            "hf_only_pass_tasks": hf_only_pass_tasks,
            "albatross_only_pass_tasks": alb_only_pass_tasks,
        },
        "row_diff": dict(row_counts),
        "rates": {
            "completion_diff_rate": _rate(row_counts["different_completion"], total),
            "token_count_diff_rate": _rate(row_counts["different_token_counts"], total),
            "row_correct_disagreement_rate": _rate(row_counts["hf_only_correct"] + row_counts["albatross_only_correct"], total),
            "prompt_token_diff_rate": _rate(row_counts["prompt_token_diffs"], total),
        },
        "stops": {
            "hf": _stop_counts(hf),
            "albatross": _stop_counts(alb),
        },
        "verify_errors": {
            "hf": _verify_counts(hf),
            "albatross": _verify_counts(alb),
        },
        "top_task_advantages": {
            "hf": task_deltas_hf_adv,
            "albatross": task_deltas_alb_adv,
        },
        "examples": {
            "completion_diffs": first_completion_diffs,
            "prompt_token_diffs": prompt_token_diffs,
            **row_correct_examples,
        },
    }


def _group_by_task(rows: dict[tuple[int, int], dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows.values():
        grouped[int(row["task_index"])].append(row)
    return grouped


def _row_example(hf_row: dict[str, Any], alb_row: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_index": hf_row["task_index"],
        "sample_id": hf_row["sample_id"],
        "problem": hf_row.get("problem", ""),
        "answer": hf_row.get("answer", ""),
        "hf": {
            "correct": hf_row["correct"],
            "generated_tokens": hf_row["generated_tokens"],
            "stop_reason": hf_row["stop_reason"],
            "truncated": hf_row["truncated"],
            "completion_sha1": hf_row["completion_sha1"],
            "completion_snippet": hf_row["completion_snippet"],
        },
        "albatross": {
            "correct": alb_row["correct"],
            "generated_tokens": alb_row["generated_tokens"],
            "stop_reason": alb_row["stop_reason"],
            "truncated": alb_row["truncated"],
            "completion_sha1": alb_row["completion_sha1"],
            "completion_snippet": alb_row["completion_snippet"],
        },
    }


def render_markdown(report: dict[str, Any], *, title: str) -> str:
    s = report["summary"]
    r = report["row_diff"]
    rates = report["rates"]
    shape = report["shape"]
    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append("## Shape")
    lines.append("")
    lines.append(f"- Rows: HF `{shape['hf_rows']}`, Albatross `{shape['albatross_rows']}`, common `{shape['common_rows']}`")
    lines.append(f"- Tasks: HF `{shape['hf_tasks']}`, Albatross `{shape['albatross_tasks']}`, common `{shape['common_tasks']}`")
    lines.append("")
    lines.append("## Accuracy summary")
    lines.append("")
    lines.append("| Metric | HF | Albatross | HF - Albatross |")
    lines.append("|---|---:|---:|---:|")
    lines.append(
        f"| Correct generations | {s['hf_correct_generations']} | {s['albatross_correct_generations']} | {s['correct_generation_delta_hf_minus_albatross']} |"
    )
    lines.append(
        f"| Rollout accuracy | {s['hf_rollout_accuracy']:.8f} | {s['albatross_rollout_accuracy']:.8f} | {s['rollout_accuracy_delta_hf_minus_albatross']:+.8f} |"
    )
    lines.append(
        f"| Pass@rollout | {s['hf_pass_at_rollout']:.6f} | {s['albatross_pass_at_rollout']:.6f} | {s['pass_at_rollout_delta_hf_minus_albatross']:+.6f} |"
    )
    lines.append(
        f"| Pass tasks | {s['hf_pass_tasks']} | {s['albatross_pass_tasks']} | {s['hf_pass_tasks'] - s['albatross_pass_tasks']} |"
    )
    lines.append("")
    lines.append("## Row-level disagreement")
    lines.append("")
    lines.append(f"- Completion differs: `{r.get('different_completion', 0)}` / `{shape['common_rows']}` (`{rates['completion_diff_rate']:.4%}`)")
    lines.append(f"- Token counts differ: `{r.get('different_token_counts', 0)}` / `{shape['common_rows']}` (`{rates['token_count_diff_rate']:.4%}`)")
    lines.append(f"- Correctness disagreement rows: `{r.get('hf_only_correct', 0) + r.get('albatross_only_correct', 0)}` / `{shape['common_rows']}` (`{rates['row_correct_disagreement_rate']:.4%}`)")
    lines.append(f"- Prompt token diffs: `{r.get('prompt_token_diffs', 0)}` / `{shape['common_rows']}` (`{rates['prompt_token_diff_rate']:.4%}`)")
    lines.append(f"- Both correct: `{r.get('both_correct', 0)}`; HF-only correct: `{r.get('hf_only_correct', 0)}`; Albatross-only correct: `{r.get('albatross_only_correct', 0)}`; both wrong: `{r.get('both_wrong', 0)}`")
    lines.append("")
    lines.append("## Pass-task deltas")
    lines.append("")
    lines.append(f"- HF-only pass tasks ({s['hf_only_pass_task_count']}): `{s['hf_only_pass_tasks']}`")
    lines.append(f"- Albatross-only pass tasks ({s['albatross_only_pass_task_count']}): `{s['albatross_only_pass_tasks']}`")
    lines.append("")
    lines.append("## Top Albatross task advantages")
    lines.append("")
    lines.extend(_task_table(report["top_task_advantages"]["albatross"], advantage_key="albatross_minus_hf_correct"))
    lines.append("")
    lines.append("## Top HF task advantages")
    lines.append("")
    lines.extend(_task_table(report["top_task_advantages"]["hf"], advantage_key="hf_minus_albatross_correct"))
    lines.append("")
    lines.append("## Stop reasons")
    lines.append("")
    lines.append(f"- HF: `{report['stops']['hf']}`")
    lines.append(f"- Albatross: `{report['stops']['albatross']}`")
    lines.append("")
    lines.append("## Verify errors")
    lines.append("")
    lines.append(f"- HF: `{report['verify_errors']['hf']}`")
    lines.append(f"- Albatross: `{report['verify_errors']['albatross']}`")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    if r.get("prompt_token_diffs", 0) == 0:
        lines.append("- Prompt token counts match for all common rows; the gap is unlikely to be caused by prompt length/BOS truncation differences.")
    if report["verify_errors"].get("hf", {}).get("", 0) == shape["hf_rows"] and report["verify_errors"].get("albatross", {}).get("", 0) == shape["albatross_rows"]:
        lines.append("- Both runs report empty verifier errors for all rows; the gap is unlikely to be caused by verifier exceptions.")
    lines.append("- Large completion divergence means the next probe should compare logits/state parity, not just final verifier outputs.")
    lines.append("")
    return "\n".join(lines)


def _task_table(rows: list[dict[str, Any]], *, advantage_key: str) -> list[str]:
    lines = ["| Task | Advantage | HF correct | Albatross correct | Problem |", "|---:|---:|---:|---:|---|"]
    for row in rows:
        problem = str(row.get("problem", "")).replace("\n", " ")
        if len(problem) > 100:
            problem = problem[:97] + "..."
        # Problem snippets can contain LaTeX display delimiters followed
        # immediately by multiple-choice labels, e.g. ``\\[...\\](A)``.  The
        # repository markdown-link checker intentionally uses a simple regex and
        # otherwise sees this as a local link to ``A``.  Add a display-safe
        # separator and escape table pipes without changing the underlying JSON.
        problem = problem.replace("](", "] (").replace("|", "\\|")
        lines.append(
            f"| {row['task_index']} | {row[advantage_key]} | {row['hf_correct']} | {row['albatross_correct']} | {problem} |"
        )
    if not rows:
        lines.append("| n/a | 0 | 0 | 0 | n/a |")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hf-generations", required=True, help="HF generations.jsonl")
    ap.add_argument("--albatross-generations", required=True, help="Albatross generations.jsonl")
    ap.add_argument("--json-output", default="", help="Optional path for JSON report")
    ap.add_argument("--markdown-output", default="", help="Optional path for Markdown report")
    ap.add_argument("--title", default="MATH500 HF vs Albatross gap analysis")
    ap.add_argument("--top-n", type=int, default=12)
    ap.add_argument("--examples", type=int, default=8, help="Number of examples to retain per bucket")
    ap.add_argument("--snippet-chars", type=int, default=700)
    args = ap.parse_args()

    hf, _ = load_rows(args.hf_generations, snippet_chars=args.snippet_chars)
    alb, _ = load_rows(args.albatross_generations, snippet_chars=args.snippet_chars)
    report = compare(hf, alb, top_n=args.top_n, examples_per_bucket=args.examples)
    report["inputs"] = {
        "hf_generations": str(args.hf_generations),
        "albatross_generations": str(args.albatross_generations),
    }

    if args.json_output:
        Path(args.json_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_output).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md = render_markdown(report, title=args.title)
    if args.markdown_output:
        Path(args.markdown_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.markdown_output).write_text(md, encoding="utf-8")
    if not args.json_output and not args.markdown_output:
        print(md)


if __name__ == "__main__":
    main()
