#!/usr/bin/env python3
# coding=utf-8
"""Build a reproducible stratified MATH500 subset from full rollout artifacts.

The selection is intended for fast accuracy-parity probes before launching a
full avg@64 run.  It samples four high-signal buckets:

- Albatross-only pass tasks.
- HF-only pass tasks.
- Both-pass tasks where Albatross has more correct generations.
- Both-pass tasks where HF has more correct generations.

It writes a subset dataset JSONL plus remapped reference generations for the
selected tasks, so existing comparison tools can operate on local subset task
indices.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_generation_rows(path: str | Path) -> dict[int, list[dict[str, Any]]]:
    by_task: dict[int, list[dict[str, Any]]] = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            task = int(row["task_index"])
            row["_line_no"] = line_no
            by_task[task].append(row)
    for rows in by_task.values():
        rows.sort(key=lambda r: int(r.get("sample_id", 0)))
    return dict(by_task)


def load_dataset(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def task_stats(hf: dict[int, list[dict[str, Any]]], alb: dict[int, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for task in sorted(set(hf) & set(alb)):
        hf_rows = hf[task]
        alb_rows = alb[task]
        hf_correct = sum(int(bool(r.get("correct"))) for r in hf_rows)
        alb_correct = sum(int(bool(r.get("correct"))) for r in alb_rows)
        out.append(
            {
                "original_task_index": task,
                "hf_correct": hf_correct,
                "albatross_correct": alb_correct,
                "hf_pass": hf_correct > 0,
                "albatross_pass": alb_correct > 0,
                "hf_minus_albatross_correct": hf_correct - alb_correct,
                "albatross_minus_hf_correct": alb_correct - hf_correct,
                "rollout": min(len(hf_rows), len(alb_rows)),
            }
        )
    return out


def _take(bucket: list[dict[str, Any]], n: int, selected: set[int], label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in bucket:
        task = int(item["original_task_index"])
        if task in selected:
            continue
        row = dict(item)
        row["selection_bucket"] = label
        row["selection_rank"] = len(rows)
        rows.append(row)
        selected.add(task)
        if len(rows) >= n:
            break
    return rows


def select_tasks(stats: list[dict[str, Any]], per_bucket: int, total: int) -> list[dict[str, Any]]:
    selected: set[int] = set()
    chosen: list[dict[str, Any]] = []

    alb_only = [s for s in stats if s["albatross_pass"] and not s["hf_pass"]]
    alb_only.sort(key=lambda s: (s["albatross_correct"], s["albatross_minus_hf_correct"]), reverse=True)
    hf_only = [s for s in stats if s["hf_pass"] and not s["albatross_pass"]]
    hf_only.sort(key=lambda s: (s["hf_correct"], s["hf_minus_albatross_correct"]), reverse=True)
    both_alb_adv = [s for s in stats if s["hf_pass"] and s["albatross_pass"] and s["albatross_correct"] > s["hf_correct"]]
    both_alb_adv.sort(key=lambda s: (s["albatross_minus_hf_correct"], s["albatross_correct"]), reverse=True)
    both_hf_adv = [s for s in stats if s["hf_pass"] and s["albatross_pass"] and s["hf_correct"] > s["albatross_correct"]]
    both_hf_adv.sort(key=lambda s: (s["hf_minus_albatross_correct"], s["hf_correct"]), reverse=True)

    for label, bucket in [
        ("albatross_only_pass", alb_only),
        ("hf_only_pass", hf_only),
        ("both_pass_albatross_adv", both_alb_adv),
        ("both_pass_hf_adv", both_hf_adv),
    ]:
        chosen.extend(_take(bucket, per_bucket, selected, label))

    if len(chosen) < total:
        fill = [s for s in stats if int(s["original_task_index"]) not in selected]
        fill.sort(key=lambda s: (abs(int(s["hf_minus_albatross_correct"])), max(s["hf_correct"], s["albatross_correct"])), reverse=True)
        chosen.extend(_take(fill, total - len(chosen), selected, "fill_abs_delta"))

    return chosen[:total]


def write_subset_dataset(dataset: list[dict[str, Any]], selected: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for local_index, meta in enumerate(selected):
            original = int(meta["original_task_index"])
            item = dict(dataset[original])
            item["original_task_index"] = original
            item["selection_bucket"] = meta["selection_bucket"]
            item["selection_rank"] = int(meta["selection_rank"])
            item["selection_local_index"] = local_index
            item["selection_hf_correct"] = int(meta["hf_correct"])
            item["selection_albatross_correct"] = int(meta["albatross_correct"])
            item.setdefault("unique_id", f"math500_{original}")
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def write_remapped_generations(
    generations: dict[int, list[dict[str, Any]]],
    selected: list[dict[str, Any]],
    path: Path,
) -> None:
    original_to_local = {int(meta["original_task_index"]): i for i, meta in enumerate(selected)}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for original, local in sorted(original_to_local.items(), key=lambda x: x[1]):
            for row in generations.get(original, []):
                out = {k: v for k, v in row.items() if not k.startswith("_")}
                out["source_task_index"] = original
                out["task_index"] = local
                out["local_task_index"] = local
                f.write(json.dumps(out, ensure_ascii=False) + "\n")


def summarize_selected(selected: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, int] = defaultdict(int)
    hf_correct = 0
    alb_correct = 0
    hf_pass = 0
    alb_pass = 0
    for row in selected:
        buckets[str(row["selection_bucket"])] += 1
        hf_correct += int(row["hf_correct"])
        alb_correct += int(row["albatross_correct"])
        hf_pass += int(bool(row["hf_pass"]))
        alb_pass += int(bool(row["albatross_pass"]))
    rollout = int(selected[0]["rollout"]) if selected else 0
    total_rows = len(selected) * rollout
    return {
        "selected_tasks": len(selected),
        "rollout": rollout,
        "total_generations_per_run": total_rows,
        "buckets": dict(sorted(buckets.items())),
        "full_run_reference_on_selected_tasks": {
            "hf_correct_generations": hf_correct,
            "albatross_correct_generations": alb_correct,
            "hf_minus_albatross_correct": hf_correct - alb_correct,
            "hf_pass_tasks": hf_pass,
            "albatross_pass_tasks": alb_pass,
            "hf_minus_albatross_pass_tasks": hf_pass - alb_pass,
            "hf_pass_at_rollout": hf_pass / max(len(selected), 1),
            "albatross_pass_at_rollout": alb_pass / max(len(selected), 1),
        },
    }


def render_readme(summary: dict[str, Any], selected: list[dict[str, Any]], *, title: str) -> str:
    ref = summary["full_run_reference_on_selected_tasks"]
    lines = [f"# {title}", ""]
    lines.append("## Selection summary")
    lines.append("")
    lines.append(f"- Selected tasks: `{summary['selected_tasks']}`")
    lines.append(f"- Rollout per task in source runs: `{summary['rollout']}`")
    lines.append(f"- Buckets: `{summary['buckets']}`")
    lines.append("")
    lines.append("## Full-run reference restricted to selected tasks")
    lines.append("")
    lines.append("| Metric | HF full run | Albatross full run | HF - Albatross |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| Correct generations | {ref['hf_correct_generations']} | {ref['albatross_correct_generations']} | {ref['hf_minus_albatross_correct']} |")
    lines.append(f"| Pass tasks | {ref['hf_pass_tasks']} | {ref['albatross_pass_tasks']} | {ref['hf_minus_albatross_pass_tasks']} |")
    lines.append(f"| Pass@64 | {ref['hf_pass_at_rollout']:.6f} | {ref['albatross_pass_at_rollout']:.6f} | {ref['hf_pass_at_rollout'] - ref['albatross_pass_at_rollout']:+.6f} |")
    lines.append("")
    lines.append("## Selected original task IDs")
    lines.append("")
    for bucket in sorted({str(s["selection_bucket"]) for s in selected}):
        ids = [str(s["original_task_index"]) for s in selected if s["selection_bucket"] == bucket]
        lines.append(f"- `{bucket}`: `{', '.join(ids)}`")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append("- `dataset.jsonl`: subset dataset for fresh reruns")
    lines.append("- `subset_tasks.json`: local->original mapping and source full-run counts")
    lines.append("- `hf_full_reference_generations.jsonl`: source HF full-run rows remapped to local subset IDs")
    lines.append("- `albatross_full_reference_generations.jsonl`: source Albatross full-run rows remapped to local subset IDs")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--hf-generations", required=True)
    ap.add_argument("--albatross-generations", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--per-bucket", type=int, default=16)
    ap.add_argument("--total", type=int, default=64)
    ap.add_argument("--title", default="MATH500 stratified accuracy-parity subset")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(args.dataset)
    hf = load_generation_rows(args.hf_generations)
    alb = load_generation_rows(args.albatross_generations)
    stats = task_stats(hf, alb)
    selected = select_tasks(stats, per_bucket=args.per_bucket, total=args.total)
    write_subset_dataset(dataset, selected, out_dir / "dataset.jsonl")
    write_remapped_generations(hf, selected, out_dir / "hf_full_reference_generations.jsonl")
    write_remapped_generations(alb, selected, out_dir / "albatross_full_reference_generations.jsonl")
    summary = summarize_selected(selected)
    (out_dir / "subset_tasks.json").write_text(json.dumps({"summary": summary, "tasks": selected}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "README.md").write_text(render_readme(summary, selected, title=args.title), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
